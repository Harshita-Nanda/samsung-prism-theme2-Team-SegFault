"""pipeline.py - real pipeline: Phase 1 engine (Person 1) -> Phase 2 mapper (Person 3).
 
Returns the final API body: {query, query_variations, response, meta}.
"""
from __future__ import annotations
 
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional
 
from engine import GeminiClient, run_phase1, to_api_payload  # Person 1
from catalog import DeeplinkCatalog                           # Person 3
from mapper import map_and_order                              # Person 3
from schema import Goal                                       # Person 3 (root schema.py)
 
logger = logging.getLogger(__name__)
 
DEEPLINKS_PATH = Path(__file__).resolve().parent / "deeplinks.json"
 
_lock = threading.Lock()
_llm: Optional[GeminiClient] = None
_catalog: Optional[DeeplinkCatalog] = None
 
 
def get_llm() -> GeminiClient:
    """One Gemini client for the whole process (needs GEMINI_API_KEY)."""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                _llm = GeminiClient()
    return _llm
 
 
def get_catalog() -> DeeplinkCatalog:
    """Load deeplinks.json + build indexes once. Call at startup to fail early.
 
    Default: dense model required (Person 3's rule). If it cannot load (e.g. no
    internet on first run), set ALLOW_TFIDF_FALLBACK=1 to accept degraded mode.
    """
    global _catalog
    if _catalog is None:
        with _lock:
            if _catalog is None:
                allow = os.getenv("ALLOW_TFIDF_FALLBACK", "0") == "1"
                _catalog = DeeplinkCatalog(DEEPLINKS_PATH, allow_tfidf_fallback=allow)
                logger.info("Deeplink catalog ready, retrieval_mode=%s", _catalog.retrieval_mode)
    return _catalog
 
 
def _siis_to_text(siis_response: Any) -> Optional[str]:
    """run_phase1 wants the SIIS article as a plain string (or None)."""
    if siis_response is None or isinstance(siis_response, str):
        return siis_response
    return json.dumps(siis_response, ensure_ascii=False)
 
 
def run_pipeline(query: str, siis_response: Any = None) -> Dict[str, Any]:
    """Phase 1 (LLM) -> Phase 2 (deeplink mapping + ordering) -> final JSON dict."""
    result = run_phase1(query, _siis_to_text(siis_response), get_llm())
    payload = to_api_payload(result)  # query, query_variations, response, meta
 
    # Fallback answers (no_match / no_siis_context / llm_error) have empty
    # contexts - nothing to map. main.py must NOT cache them (meta["fallback"]).
    if payload["meta"].get("fallback"):
        return payload
 
    catalog = get_catalog()
    mapped = []
    for ctx in payload["response"]["contexts"]:
        goal = Goal.model_validate(ctx)
        goal = map_and_order(goal, catalog)
        mapped.append(goal.model_dump(mode="json"))
    payload["response"]["contexts"] = mapped
    return payload
 