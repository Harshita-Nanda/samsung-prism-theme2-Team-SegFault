"""Gemini LLM client for Phase 1, with native structured output, cost tracking and retries.

SDK: `google-genai` (pip install google-genai).
    NOT `google-generativeai`: that package reached end of life on 30 Nov 2025
    (repo archived Dec 2025, no new features, no new models).

How the Pydantic schema is enforced (three layers, strongest last)
  1. The draft Pydantic model (e.g. _DraftExtraction) is converted to a Gemini-safe
     JSON schema and sent as structured output, so Gemini's decoder is constrained to
     emit JSON of that shape (right keys, types, enums).
  2. The reply is parsed and validated with `model_cls.model_validate(...)`; on failure
     the caller sends one repair turn.
  3. extraction.py then validates the final ContextDeeplinkResponse from schema.py.
     Gemini is deliberately NOT asked for the final schema: goal sentences, deeplinks
     and ordering are built by code, so the model can never get them wrong.

Env vars
  GEMINI_API_KEY (or GOOGLE_API_KEY)   required for live calls
  GEMINI_MODEL                          default gemini-3.5-flash-lite
  GEMINI_THINKING_LEVEL                 minimal | low | medium | high (Gemini 3 only), default low
  GEMINI_TEMPERATURE                    unset by default (Google recommends 1.0 for Gemini 3)
  GEMINI_RPM                            client-side requests/minute cap for free-tier quotas
  LLM_BILLING                           free (default) -> cost_usd = 0 ; paid -> list price
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type, Union

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens, paid Standard tier, text). Source: ai.google.dev pricing
# page, checked Sept 2026. Thinking tokens are billed as output tokens.
# ---------------------------------------------------------------------------
PRICING_PER_M: Dict[str, Tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def price_for(model: str) -> Tuple[float, float]:
    env_in, env_out = os.getenv("LLM_PRICE_IN_PER_M"), os.getenv("LLM_PRICE_OUT_PER_M")
    if env_in and env_out:
        return float(env_in), float(env_out)
    for name in sorted(PRICING_PER_M, key=len, reverse=True):     # longest prefix wins
        if model.startswith(name):
            return PRICING_PER_M[name]
    return (0.0, 0.0)


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0          # includes thinking tokens (billed as output)
    latency_ms: float = 0.0
    finish_reason: str = ""


@dataclass
class Usage:
    """Accumulates tokens and cost across every LLM call for one request (thread-safe)."""

    model: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, reply: LLMReply) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += reply.prompt_tokens
            self.completion_tokens += reply.completion_tokens
            self.llm_latency_ms += reply.latency_ms

    def merge(self, other: "Usage") -> None:
        with self._lock:
            self.calls += other.calls
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens
            self.llm_latency_ms += other.llm_latency_ms
            self.model = self.model or other.model

    @property
    def list_price_usd(self) -> float:
        """What this request would cost on the paid tier (report this in metrics.md)."""
        p_in, p_out = price_for(self.model)
        return round((self.prompt_tokens * p_in + self.completion_tokens * p_out) / 1_000_000, 8)

    @property
    def cost_usd(self) -> float:
        """What you are actually billed: 0 on the free tier."""
        return 0.0 if os.getenv("LLM_BILLING", "free").lower() == "free" else self.list_price_usd


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, *, max_tokens: int = 1500,
                 history: Optional[List[Dict[str, str]]] = None,
                 response_schema: Optional[Type[BaseModel]] = None) -> LLMReply: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LLMParseError(ValueError):
    """Reply was not usable JSON. Callers answer this with one repair turn."""


class LLMTruncatedError(LLMParseError):
    """Reply hit max_output_tokens; JSON is cut off."""


class LLMBlockedError(RuntimeError):
    """Prompt or reply blocked by safety/recitation filters. Not retried."""


class LLMRateLimitError(RuntimeError):
    """Quota exhausted and the server's retry delay exceeds our latency budget."""


# ---------------------------------------------------------------------------
# Pydantic -> Gemini-safe JSON schema
# ---------------------------------------------------------------------------
# Kept keys are the ones Gemini documents as supported; everything else is dropped
# because unsupported keywords are either ignored or cause 400s on some versions.
_KEEP_KEYS = {"type", "properties", "required", "items", "enum", "description",
              "minItems", "maxItems", "minimum", "maximum", "format"}


def to_gemini_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to a flat JSON schema Gemini accepts on every SDK version.

    - inlines $ref/$defs (no references left)
    - collapses Optional[X] (anyOf [X, null]) to X; the draft models default to "" anyway
    - drops defaults/titles/additionalProperties
    - marks every property required, so Gemini always emits every key
    """
    raw = model_cls.model_json_schema()
    defs = raw.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return resolve({**target, **extra})
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            base = resolve(options[0]) if options else {"type": "string"}
            if "description" in node:
                base["description"] = node["description"]
            return base
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k not in _KEEP_KEYS:
                continue
            if k == "properties":
                out[k] = {name: resolve(sub) for name, sub in v.items()}
            elif k == "items":
                out[k] = resolve(v)
            else:
                out[k] = v
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"].keys())
        return out

    return resolve(raw)


# ---------------------------------------------------------------------------
# Client-side rate limiter (free tier has low requests-per-minute quotas)
# ---------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, rpm: Optional[float]):
        self._interval = 60.0 / rpm if rpm else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        if slot > now:
            time.sleep(slot - now)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_BLOCK_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE",
                  "IMAGE_SAFETY", "OTHER"}
_RETRY_DELAY_RE = re.compile(r"retry(?:Delay)?[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s", re.I)


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value) or "").split(".")[-1].upper()


class GeminiClient:
    """Gemini via `google-genai`, using client.models.generate_content.

    generate_content is labelled "legacy" by Google but documented as fully supported;
    it is used here because its token-usage fields are stable and well documented.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, *,
                 timeout_s: float = 30.0, max_retries: int = 2, max_backoff_s: float = 3.0,
                 thinking_level: Optional[str] = None, temperature: Optional[float] = None,
                 rpm: Optional[float] = None, client: Any = None, schema_mode: Optional[str] = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.thinking_level = thinking_level or os.getenv("GEMINI_THINKING_LEVEL", "low")
        t = temperature if temperature is not None else os.getenv("GEMINI_TEMPERATURE")
        self.temperature = float(t) if t not in (None, "") else None
        self.max_retries = max_retries
        self.max_backoff_s = max_backoff_s
        rpm_env = os.getenv("GEMINI_RPM")
        self._limiter = _RateLimiter(rpm if rpm is not None else (float(rpm_env) if rpm_env else None))

        if client is not None:                      # injected (tests, or a pre-built client)
            self._client = client
            self.schema_mode = schema_mode or "response_format"
            return

        from google import genai                    # lazy import: tests run without the SDK
        from google.genai import types

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (free key: Google AI Studio).")
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        # Newer SDKs take response_format (current docs); older ones response_json_schema.
        fields = getattr(types.GenerateContentConfig, "model_fields", {}) or {}
        self.schema_mode = schema_mode or (
            "response_format" if "response_format" in fields
            else "response_json_schema" if "response_json_schema" in fields
            else "response_schema"
        )

    # -- request building ---------------------------------------------------
    def build_config(self, system: str, max_tokens: int,
                     response_schema: Optional[Type[BaseModel]]) -> Dict[str, Any]:
        thinking = self.model.startswith("gemini-3") and self.thinking_level
        cfg: Dict[str, Any] = {
            "system_instruction": system,
            # Thinking tokens share the output budget; give headroom so JSON isn't cut off.
            "max_output_tokens": max_tokens + (2048 if thinking else 0),
        }
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if thinking:
            cfg["thinking_config"] = {"thinking_level": self.thinking_level}

        schema = to_gemini_schema(response_schema) if response_schema else None
        if self.schema_mode == "response_format":
            text_fmt: Dict[str, Any] = {"mime_type": "application/json"}
            if schema:
                text_fmt["schema"] = schema
            cfg["response_format"] = {"text": text_fmt}
        elif self.schema_mode == "response_json_schema":
            cfg["response_mime_type"] = "application/json"
            if schema:
                cfg["response_json_schema"] = schema
        else:                                       # very old SDKs: OpenAPI subset
            cfg["response_mime_type"] = "application/json"
            if schema:
                cfg["response_schema"] = schema
        return cfg

    @staticmethod
    def build_contents(user: str, history: Optional[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        role_map = {"assistant": "model", "model": "model", "user": "user"}
        turns = [*(history or []), {"role": "user", "content": user}]
        return [{"role": role_map.get(t["role"], "user"), "parts": [{"text": t["content"]}]} for t in turns]

    # -- response handling --------------------------------------------------
    @staticmethod
    def parse_response(resp: Any, latency_ms: float) -> LLMReply:
        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            fb = getattr(resp, "prompt_feedback", None)
            reason = _enum_name(getattr(fb, "block_reason", None)) or "NO_CANDIDATES"
            raise LLMBlockedError(f"prompt blocked: {reason}")

        cand = candidates[0]
        finish = _enum_name(getattr(cand, "finish_reason", ""))
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        text = "".join(getattr(p, "text", "") or "" for p in parts if not getattr(p, "thought", False))

        if finish in _BLOCK_REASONS and not text:
            raise LLMBlockedError(f"reply blocked: {finish}")
        if finish == "MAX_TOKENS":
            raise LLMTruncatedError("reply cut off at max_output_tokens; return shorter JSON")

        um = getattr(resp, "usage_metadata", None)
        prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
        output_tokens = (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        return LLMReply(text=text, prompt_tokens=prompt_tokens, completion_tokens=output_tokens,
                        latency_ms=latency_ms, finish_reason=finish)

    # -- retries ------------------------------------------------------------
    def _retry_delay(self, err: Exception, attempt: int) -> Optional[float]:
        """Seconds to wait before retrying, or None if not retryable / too slow."""
        code = getattr(err, "code", None) or getattr(err, "status_code", None)
        name = type(err).__name__.lower()
        if code not in _RETRYABLE_CODES and "timeout" not in name and "connect" not in name:
            return None
        m = _RETRY_DELAY_RE.search(str(err))
        delay = float(m.group(1)) if m else 0.5 * (2 ** attempt)
        if delay > self.max_backoff_s:
            if code == 429:
                raise LLMRateLimitError(f"quota exhausted, server asks to wait {delay:.0f}s") from err
            return None
        return delay

    def complete(self, system: str, user: str, *, max_tokens: int = 1500,
                 history: Optional[List[Dict[str, str]]] = None,
                 response_schema: Optional[Type[BaseModel]] = None) -> LLMReply:
        config = self.build_config(system, max_tokens, response_schema)
        contents = self.build_contents(user, history)
        attempt = 0
        expanded = False
        while True:
            self._limiter.wait()
            t0 = time.perf_counter()
            try:
                resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
            except Exception as err:                # SDK raises google.genai.errors.APIError subclasses
                delay = self._retry_delay(err, attempt) if attempt < self.max_retries else None
                if delay is None:
                    raise
                time.sleep(delay)
                attempt += 1
                continue
            try:
                return self.parse_response(resp, (time.perf_counter() - t0) * 1000)
            except LLMTruncatedError:
                if expanded:
                    raise
                # Usually thinking ate the budget: one retry with double the output room.
                expanded = True
                config = {**config, "max_output_tokens": min(config["max_output_tokens"] * 2, 16384)}


# ---------------------------------------------------------------------------
# Fake client for tests / offline demo
# ---------------------------------------------------------------------------
class FakeLLM:
    """Offline stand-in. `responses` is a list consumed in order, or a callable
    (system, user) -> str|dict|Exception so one fake can serve concurrent calls."""

    def __init__(self, responses: Union[List[Any], Callable[[str, str], Any]],
                 model: str = "gemini-3.5-flash-lite", latency_ms: float = 0.0):
        self.model = model
        self._responses = responses
        self._latency = latency_ms
        self._lock = threading.Lock()
        self.calls: List[Tuple[str, str]] = []
        self.schemas: List[Optional[Type[BaseModel]]] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 1500,
                 history: Optional[List[Dict[str, str]]] = None,
                 response_schema: Optional[Type[BaseModel]] = None) -> LLMReply:
        with self._lock:
            self.calls.append((system, user))
            self.schemas.append(response_schema)
            if callable(self._responses):
                out = self._responses(system, user)
            else:
                if not self._responses:
                    raise RuntimeError("FakeLLM ran out of scripted responses")
                out = self._responses.pop(0)
        if isinstance(out, Exception):
            raise out
        if self._latency:
            time.sleep(self._latency / 1000)
        text = out if isinstance(out, str) else json.dumps(out)
        return LLMReply(text=text, prompt_tokens=len(system + user) // 4,
                        completion_tokens=len(text) // 4, latency_ms=self._latency, finish_reason="STOP")


# ---------------------------------------------------------------------------
# Robust JSON extraction (still needed: repair turns and older models can add chatter)
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


def parse_llm_json(text: str) -> dict:
    """Parse a JSON object even if wrapped in ```json fences or surrounded by chatter."""
    if not text or not text.strip():
        raise LLMParseError("empty reply")
    t = _FENCE_RE.sub("", text.strip().lstrip("﻿"))
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise LLMParseError(f"no JSON object found in reply: {text[:120]!r}")
