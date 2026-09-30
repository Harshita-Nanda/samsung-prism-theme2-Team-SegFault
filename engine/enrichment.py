"""Phase [0] Query Enrichment.

Input : raw complaint
Output: canonical query, domain, intent_key, deterministic semantic_key and
        8-10 distinct, URL-free query_variations.

Important latency note: this module calls an LLM, so it runs ONLY on the cold
path (cache miss). The cache fast path (<300 ms) must never wait on it; it uses
`semantic_key()` (local, <1 ms) and Phase 3 embeddings of the variations this
module produces.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .llm_client import LLMClient, LLMParseError, Usage, parse_llm_json
from .prompts import (ENRICHMENT_SYSTEM_PROMPT, ENRICHMENT_TOPUP_TEMPLATE,
                      ENRICHMENT_USER_TEMPLATE, REPAIR_TEMPLATE)
from .sanitizers import normalize_query, scrub_urls, semantic_key

log = logging.getLogger(__name__)

VARIATIONS_MIN, VARIATIONS_MAX = 8, 10
DOMAINS = {"battery", "display", "camera", "performance", "connectivity", "other"}
_INTENT_RE = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+){1,3}$")
_NEAR_DUP_RATIO = 0.90


class _EnrichmentLLMOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    canonical_query: str = ""
    # json_schema_extra only shapes the schema sent to Gemini; validation stays lenient
    # so a slightly-off value is repaired by code instead of failing the request.
    domain: str = Field("other", json_schema_extra={"enum": sorted(["battery", "display", "camera",
                                                                   "performance", "connectivity", "other"])})
    intent_key: str = ""
    variations: List[str] = Field(default_factory=list, json_schema_extra={"minItems": 8, "maxItems": 10})

    @field_validator("variations", mode="before")
    @classmethod
    def _coerce(cls, v):
        if isinstance(v, str):
            return [line for line in v.splitlines() if line.strip()]
        return [str(x) for x in (v or []) if isinstance(x, (str, int, float))]


@dataclass
class EnrichmentResult:
    query: str
    canonical_query: str
    domain: str
    intent_key: str
    semantic_key: str
    query_variations: List[str]
    usage: Usage
    degraded: bool = False                  # True => LLM failed, deterministic fallback used
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _norm_for_dedupe(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def clean_variations(candidates: List[str], original: str, limit: int = VARIATIONS_MAX,
                     near_dup_ratio: float = _NEAR_DUP_RATIO) -> List[str]:
    """Scrub URLs, strip numbering/quotes, drop empties, the original, and near-duplicates."""
    out: List[str] = []
    seen_norm: List[str] = [_norm_for_dedupe(original)]
    for c in candidates:
        v = scrub_urls(c)
        v = re.sub(r"^\s*(?:[-*•]+|\d{1,2}[.)])\s*", "", v).strip().strip('"\'“”').strip()
        v = re.sub(r"\s+", " ", v)
        if len(v) < 3 or len(re.findall(r"[A-Za-z]{2,}", v)) < 2:
            continue
        n = _norm_for_dedupe(v)
        if any(n == s or difflib.SequenceMatcher(None, n, s).ratio() >= near_dup_ratio for s in seen_norm):
            continue
        seen_norm.append(n)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _typo(text: str) -> str:
    """Deterministic typo: swap two middle letters of the longest word."""
    words = text.split()
    if not words:
        return text
    idx = max(range(len(words)), key=lambda i: len(words[i]))
    w = words[idx]
    if len(w) >= 5:
        mid = len(w) // 2
        words[idx] = w[:mid - 1] + w[mid] + w[mid - 1] + w[mid + 1:]
    return " ".join(words)


def template_variations(query: str, canonical: str) -> List[str]:
    """Last-resort paraphrases so the contract (8-10 items) is ALWAYS met."""
    c = canonical.rstrip(".?! ")
    c_low = c[:1].lower() + c[1:]
    kw = " ".join(t for t in semantic_key(query).split("|") if t)[:60] or c_low
    return [
        f"How do I fix this on my Galaxy phone: {c_low}?",
        f"My phone has a problem where {c_low}.",
        f"{kw} fix",
        f"Samsung {kw} problem",
        f"What should I do when {c_low}?",
        f"Please help, {c_low} and it is really frustrating.",
        f"Galaxy device issue: {c_low}.",
        f"why is this happening {c_low}",
        _typo(f"my phone {c_low}"),
        f"Troubleshooting steps needed: {c_low}.",
        f"{kw} not working properly",
    ]


def _sanitize_intent_key(key: str, domain: str, query: str) -> str:
    k = re.sub(r"[^a-z0-9_.]", "_", (key or "").strip().lower()).strip("._")
    k = re.sub(r"_+", "_", k)
    k = re.sub(r"\.+", ".", k)
    if _INTENT_RE.match(k):
        return k
    toks = [t for t in semantic_key(query).split("|") if t][:3] or ["general"]
    return f"{domain}.{'_'.join(toks)}"


def _call_json(llm: LLMClient, usage: Usage, system: str, user: str, model_cls, max_tokens: int):
    """One call + one repair retry on parse/validation failure."""
    history = None
    last_err: Optional[Exception] = None
    for _attempt in range(2):
        reply = llm.complete(system, user if history is None else REPAIR_TEMPLATE.format(error=last_err),
                             max_tokens=max_tokens, history=history, response_schema=model_cls)
        usage.add(reply)
        try:
            return model_cls.model_validate(parse_llm_json(reply.text))
        except (LLMParseError, ValidationError) as e:
            last_err = str(e)[:300]
            history = [{"role": "user", "content": user}, {"role": "assistant", "content": reply.text}]
    raise LLMParseError(f"invalid JSON after repair: {last_err}")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def enrich_query(query: str, llm: Optional[LLMClient]) -> EnrichmentResult:
    query = (query or "").strip()
    usage = Usage(model=getattr(llm, "model", "") if llm else "")
    warnings: List[str] = []
    normalized = normalize_query(query)
    skey = semantic_key(query)

    out: Optional[_EnrichmentLLMOut] = None
    if llm is not None and query:
        try:
            out = _call_json(llm, usage, ENRICHMENT_SYSTEM_PROMPT,
                             ENRICHMENT_USER_TEMPLATE.format(query=query), _EnrichmentLLMOut, 900)
        except Exception as e:                      # never let enrichment kill the request
            warnings.append(f"enrichment_llm_failed: {type(e).__name__}: {str(e)[:120]}")
            log.warning("enrichment LLM failed: %s", e)

    degraded = out is None
    canonical = scrub_urls(out.canonical_query) if out else ""
    if not canonical or len(canonical.split()) < 3:
        canonical = query or normalized
    domain = (out.domain.strip().lower() if out else "other")
    domain = domain if domain in DOMAINS else "other"
    intent_key = _sanitize_intent_key(out.intent_key if out else "", domain, query)

    variations = clean_variations(out.variations if out else [], query)

    # Top-up call if the model returned too few distinct items.
    if len(variations) < VARIATIONS_MIN and llm is not None and not degraded:
        need = VARIATIONS_MAX - len(variations)
        try:
            class _TopUp(BaseModel):
                model_config = ConfigDict(extra="ignore")
                variations: List[str] = []
            extra = _call_json(llm, usage, ENRICHMENT_SYSTEM_PROMPT,
                               ENRICHMENT_TOPUP_TEMPLATE.format(query=query, existing=json.dumps(variations),
                                                                n=need), _TopUp, 600)
            variations = clean_variations(variations + extra.variations, query)
        except Exception as e:
            warnings.append(f"variation_topup_failed: {type(e).__name__}")

    # Deterministic guarantee.
    if len(variations) < VARIATIONS_MIN:
        warnings.append(f"template_variations_used: had {len(variations)}")
        variations = clean_variations(variations + template_variations(query, canonical), query)
        if len(variations) < VARIATIONS_MIN:
            # Long queries make wrapper templates look alike; relax to exact-duplicate removal only.
            variations = clean_variations(variations + template_variations(query, canonical), query,
                                          near_dup_ratio=1.01)

    return EnrichmentResult(
        query=query, canonical_query=canonical, domain=domain, intent_key=intent_key,
        semantic_key=skey, query_variations=variations[:VARIATIONS_MAX], usage=usage,
        degraded=degraded, warnings=warnings,
    )
