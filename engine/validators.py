"""Contract validator: checks a response against EVERY hackathon rule.

Use it in tests, in CI over results.jsonl, and as a last gate before caching.
Returns a list of human-readable violations (empty list == compliant).

    python -m engine.validators results.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List

from pydantic import ValidationError

from .sanitizers import GOAL_RE, find_urls, sentence_case, title_case, word_count
from .schema import APIResponse, ContextDeeplinkResponse

_FENCE = re.compile(r"```")


def _walk_strings(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def validate_response(resp: Dict[str, Any]) -> List[str]:
    """Validate a ContextDeeplinkResponse dict ({"contexts": [...]})."""
    errors: List[str] = []
    try:
        model = ContextDeeplinkResponse.model_validate(resp)
    except ValidationError as e:
        return [f"schema: {e.errors()[:3]}"]

    for path, s in _walk_strings(resp):
        if path.endswith(".deeplink") and s.startswith("bixby://"):
            continue
        if find_urls(s):
            errors.append(f"url_leak at {path}: {s[:80]!r}")
        if _FENCE.search(s):
            errors.append(f"markdown_fence at {path}")

    for gi, g in enumerate(model.contexts):
        p = f"contexts[{gi}]"
        if not GOAL_RE.match(g.goal):
            errors.append(f"{p}.goal syntax: {g.goal!r}")
        if not 2 <= word_count(g.title) <= 3:
            errors.append(f"{p}.title word count {word_count(g.title)}: {g.title!r}")
        if g.title != sentence_case(g.title):
            errors.append(f"{p}.title not sentence case: {g.title!r}")
        if not 0.0 <= g.score <= 1.0:
            errors.append(f"{p}.score out of range: {g.score}")
        if not g.actions:
            errors.append(f"{p} has no actions")
        seen_critical = False
        names = set()
        for ai, a in enumerate(g.actions):
            ap = f"{p}.actions[{ai}]"
            cat = a.category.value if a.category else "manual"
            if a.actionName != title_case(a.actionName):
                errors.append(f"{ap}.actionName not Title Case: {a.actionName!r}")
            if a.actionName.lower() in names:
                errors.append(f"{ap}.actionName duplicate screen: {a.actionName!r}")
            names.add(a.actionName.lower())
            n = word_count(a.description)
            if not (5 <= n <= 7 and a.description.startswith("It will")):
                errors.append(f"{ap}.description ({n} words): {a.description!r}")
            if cat == "critical":
                seen_critical = True
            elif seen_critical:
                errors.append(f"{ap} non-critical action after a critical one")
            if not a.stepGroups or not any(sg.steps for sg in a.stepGroups):
                errors.append(f"{ap} has no steps")
            for si, sg in enumerate(a.stepGroups):
                if cat == "manual" and sg.actionableDeeplink is not None:
                    errors.append(f"{ap}.stepGroups[{si}] manual action carries a deeplink")
                if sg.actionableDeeplink and not sg.actionableDeeplink.deeplink.startswith("bixby://"):
                    errors.append(f"{ap}.stepGroups[{si}] deeplink not bixby://")
                for st in sg.steps:
                    if not st.strip():
                        errors.append(f"{ap}.stepGroups[{si}] empty step")
    return errors


def validate_api_payload(payload: Dict[str, Any]) -> List[str]:
    """Validate the full API body (query / query_variations / response / meta)."""
    errors: List[str] = []
    try:
        APIResponse.model_validate(payload)
    except ValidationError as e:
        errors.append(f"api_schema: {e.errors()[:3]}")
    for k in ("latency_ms", "cache_hit", "model", "cost_usd"):
        if k not in payload.get("meta", {}):
            errors.append(f"meta missing {k}")
    variations = payload.get("query_variations", [])
    if len({v.strip().lower() for v in variations}) != len(variations):
        errors.append("query_variations contain duplicates")
    for path, s in _walk_strings({"q": payload.get("query", ""), "v": variations}):
        if find_urls(s):
            errors.append(f"url_leak at {path}: {s[:80]!r}")
    errors += validate_response(payload.get("response", {}))
    if payload.get("meta", {}).get("fallback") == "no_match" and payload["response"].get("contexts"):
        errors.append("fallback=no_match but contexts is not empty")
    return errors


def _main(path: str) -> int:
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if line.lstrip().startswith("```"):
                print(f"line {ln}: markdown fence around JSON")
                bad += 1
                continue
            errs = validate_api_payload(json.loads(line))
            if errs:
                bad += 1
                print(f"line {ln}: " + "; ".join(errs))
    print("OK" if not bad else f"{bad} non-compliant line(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1]))
