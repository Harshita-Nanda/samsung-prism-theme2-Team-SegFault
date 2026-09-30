"""Phase 1 orchestrator: Query Enrichment + Structure Extraction.

Enrichment and extraction are independent, so they run in parallel threads.
Cold-path latency is therefore max(enrichment, extraction), not the sum,
which matters for the P95 <= 8 s cold budget.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .enrichment import EnrichmentResult, enrich_query
from .extraction import ExtractionResult, ScreenHint, extract_structure
from .llm_client import LLMClient, Usage
from .sanitizers import assert_no_urls, scrub_urls
from .schema import APIResponse

_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="phase1")


@dataclass
class Phase1Result:
    query: str
    enrichment: EnrichmentResult
    extraction: ExtractionResult
    latency_ms: float
    usage: Usage

    @property
    def fallback(self) -> Optional[str]:
        return self.extraction.fallback

    @property
    def screen_hints(self) -> List[ScreenHint]:
        return self.extraction.screen_hints

    @property
    def warnings(self) -> List[str]:
        return self.enrichment.warnings + self.extraction.warnings


def run_phase1(query: str, siis_response: Optional[str], llm: LLMClient, parallel: bool = True) -> Phase1Result:
    t0 = time.perf_counter()
    if parallel:
        f_enrich = _POOL.submit(enrich_query, query, llm)
        f_extract = _POOL.submit(extract_structure, query, siis_response, llm)
        enrichment, extraction = f_enrich.result(), f_extract.result()
    else:
        enrichment = enrich_query(query, llm)
        extraction = extract_structure(query, siis_response, llm, canonical_query=enrichment.canonical_query)

    usage = Usage(model=getattr(llm, "model", ""))
    usage.merge(enrichment.usage)
    usage.merge(extraction.usage)
    return Phase1Result(query=query, enrichment=enrichment, extraction=extraction,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 1), usage=usage)


def to_api_payload(result: Phase1Result, cache_hit: bool = False) -> Dict[str, Any]:
    """Preview of the final API body (Phase 4 will add deeplinks + caching around this).

    Validated through schema.APIResponse and the URL gate before it is returned.
    """
    meta: Dict[str, Any] = {
        "latency_ms": int(round(result.latency_ms)),
        "cache_hit": cache_hit,
        "model": result.usage.model,
        "cost_usd": result.usage.cost_usd,
    }
    if result.fallback:
        meta["fallback"] = result.fallback
    payload = APIResponse(
        query=scrub_urls(result.query),
        query_variations=result.enrichment.query_variations,
        response=result.extraction.response,
        meta=meta,
    ).model_dump(mode="json")
    assert_no_urls(payload)
    return payload
