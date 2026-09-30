"""
Phase 2 - Deeplink Mapping & Action Ordering.

Input:  a Goal (schema.Goal) produced by Phase 1, where every StepGroup has
        `steps` filled in and `actionableDeeplink` / `validationDeeplink` are
        both None. Action.category is Person 1's initial classification
        (auto / manual / critical).

Output: the same Goal, with:
  - StepGroup.actionableDeeplink resolved from the catalog (or left None)
  - StepGroup.validationDeeplink copied only from the matched entry's own
    `validation` object (never fabricated)
  - Action.category preserved from Person 1 unless it structurally conflicts
    with a hard rule (see _validate_category)
  - Goal.actions reordered: auto -> manual -> critical

Hard rules enforced here (per corrected contract):
  1. URIs are copied verbatim from deeplinks.json. Never rewritten, never
     using a bixby:// scheme.
  2. actionableDeeplink.deeplink == matched entry's top-level `deeplink`
     (the action URI). validationDeeplink comes only from that entry's
     nested `validation` object, when present.
  3. Category is Person 1's to set; Phase 2 validates and orders, it does
     not reclassify based on retrieval confidence.
  4. manual actions always get actionableDeeplink = None, regardless of any
     match found.
  5. critical actions may carry a valid actionableDeeplink, but are always
     ordered after both auto and manual actions.
  6. voiceassist://dummy_positive is used only when a step explicitly names
     a concrete Settings screen ("Tap <Name> settings" / "Settings > <Name>")
     AND no evidence-gated catalog entry matches it. Its description and
     message are 5-7 words naming only that screen -- never the raw query.
     "Turn off phone" or "toggle option" cannot trigger it.
  7. Never fabricate a URI, a validation rule, or a `classes` value.

Evidence gating: rank agreement between BM25 and the vector signal is NOT
enough (an all-zero, out-of-vocabulary query ties every score and hands rank
0 to DL-0001 on both signals). A candidate is accepted only if it is in the
top-N of both signals AND its raw BM25 score and raw cosine similarity both
clear per-retrieval-mode floors (EvidenceThresholds).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from catalog import DeeplinkCatalog, CatalogEntry, RankedMatch
from schema import (
    Action,
    Deeplink,
    Goal,
    StepGroup,
    ValidationDeepLink,
    actionCategory as ActionCategory,  # schema.py's actual enum name
)

CONFIDENT_TOP_N = 3


# --- Evidence gating --------------------------------------------------------
@dataclass(frozen=True)
class EvidenceThresholds:
    """Minimum RAW scores a candidate must reach (both must clear)."""

    min_bm25_score: float
    min_vector_score: float


# Calibration basis (honest scope): the BM25 floor and the TF-IDF cosine floor
# were chosen by scoring this repo's deeplinks.json with the rank_bm25
# Okapi formula (k1=1.5, b=0.75) and TF-IDF over description + message +
# qna_description + originalType. Observed: out-of-vocabulary/gibberish ->
# BM25 0.0-4.5; unrelated in-vocabulary text -> 9-13; genuine step queries ->
# 24-50. 20.0 sits in that gap. BM25 grows with query length, so it is a
# coarse guard; the cosine floor is the length-independent one.
# The "sentence-transformers" cosine floor (0.35) is NOT calibrated against
# the real all-MiniLM-L6-v2 model (it could not be run where this was
# written). It must be re-checked by QA on real traffic; override per call via
# map_and_order(..., thresholds=EvidenceThresholds(...)).
EVIDENCE_THRESHOLDS_BY_MODE = {
    "sentence-transformers": EvidenceThresholds(min_bm25_score=20.0, min_vector_score=0.35),
    "tfidf_fallback": EvidenceThresholds(min_bm25_score=20.0, min_vector_score=0.30),
}
DEFAULT_EVIDENCE_THRESHOLDS = EvidenceThresholds(min_bm25_score=20.0, min_vector_score=0.30)


def evidence_thresholds_for(
    retrieval_mode: str, override: Optional[EvidenceThresholds] = None
) -> EvidenceThresholds:
    if override is not None:
        return override
    return EVIDENCE_THRESHOLDS_BY_MODE.get(retrieval_mode, DEFAULT_EVIDENCE_THRESHOLDS)


def has_sufficient_evidence(
    match: RankedMatch, override: Optional[EvidenceThresholds] = None
) -> bool:
    """True only if raw BM25 AND raw cosine are both > 0 and clear the floors."""
    t = evidence_thresholds_for(match.retrieval_mode, override)
    return match.has_score_evidence(t.min_bm25_score, t.min_vector_score)


# --- Polarity ---------------------------------------------------------------
_OFF_PATTERN = re.compile(r"\b(disable[sd]?|turn(?:ed)? off|switch(?:ed)? off|deactivate[sd]?)\b")
_ON_PATTERN = re.compile(r"\b(enable[sd]?|turn(?:ed)? on|switch(?:ed)? on|activate[sd]?)\b")


def _infer_polarity(text: str) -> Optional[str]:
    """'on' / 'off' only from explicit words in the step text; else None."""
    low = text.lower()
    has_off = bool(_OFF_PATTERN.search(low))
    has_on = bool(_ON_PATTERN.search(low))
    if has_off and not has_on:
        return "off"
    if has_on and not has_off:
        return "on"
    return None


def _polarity_of(entry: CatalogEntry) -> Optional[str]:
    if entry.originalType in ("onURL", "onClickURL"):
        return "on"
    if entry.originalType == "offURL":
        return "off"
    return None


def _polarity_penalty(entry: CatalogEntry, preferred: str) -> int:
    """1 only when the entry's catalog polarity is the opposite of preferred."""
    p = _polarity_of(entry)
    return 1 if (p is not None and p != preferred) else 0


# --- dummy_positive: explicit Settings-screen identification ----------------
_SCREEN_STOP_WORDS = {
    "the", "a", "an", "your", "this", "that", "its", "it", "and", "or", "then",
    "to", "on", "in", "of", "for", "with", "device", "phone", "tablet",
    "relevant", "quick", "settings", "setting", "option", "options", "app",
    "apps", "tap", "open", "select", "choose", "toggle", "turn", "go",
    "navigate",
}
# "Tap Display calibration settings" / "Open the Lock screen settings"
_NAV_SETTINGS_RE = re.compile(
    r"\b(?:open|tap(?:\s+on)?|go\s+to|select|choose|navigate\s+to)\s+(?:the\s+)?"
    r"((?:[A-Za-z0-9&/\-]+\s+){1,3}?)settings?\b",
    re.IGNORECASE,
)
# "Settings > Display > Screen timeout"
_ARROW_RE = re.compile(r"\bSettings\s*(?:>|\u203a|\u2192|->)\s*([^.;\n]+)", re.IGNORECASE)
_ARROW_SPLIT_RE = re.compile(r"\s*(?:>|\u203a|\u2192|->)\s*")


def _clean_screen_name(raw: str) -> Optional[str]:
    """1-3 content words naming a screen, else None (generic / not a screen)."""
    words = [w.strip(".,;:()\"'") for w in raw.split()]
    words = [w for w in words if w]
    while words and words[-1].lower() in {"settings", "setting"}:
        words.pop()
    if not 1 <= len(words) <= 3:
        return None
    if any(w.lower() in _SCREEN_STOP_WORDS for w in words):
        return None
    if not all(re.search(r"[A-Za-z]", w) for w in words):
        return None
    return " ".join(words)


def _extract_settings_screen_name(steps: List[str]) -> Optional[str]:
    """
    Returns the concrete Settings screen a step group explicitly names (the
    last one mentioned), or None. Only two shapes count: "<nav verb> <Name>
    settings" and "Settings > ... > <Name>". Generic text ("toggle option",
    "turn off phone", "open Settings", "Quick settings panel") returns None.
    """
    found: Optional[str] = None
    for step in steps:
        for m in _ARROW_RE.finditer(step):
            segments = [x for x in _ARROW_SPLIT_RE.split(m.group(1)) if x.strip()]
            if segments:
                name = _clean_screen_name(segments[-1])
                if name:
                    found = name
        for m in _NAV_SETTINGS_RE.finditer(step):
            name = _clean_screen_name(m.group(1))
            if name:
                found = name
    return found


def _word_count(text: str) -> int:
    return len(text.split())


# --- Deeplink builders ------------------------------------------------------
def _to_actionable_deeplink(entry: CatalogEntry) -> Deeplink:
    """Build schema.Deeplink from a catalog entry's top-level fields only."""
    return Deeplink(
        deeplink=entry.deeplink,          # verbatim action URI
        description=entry.description,
        message=entry.message or "",
        originalType=entry.originalType,
        classes=None,                     # never fabricated -- no source field
    )


def _to_validation_deeplink(entry: CatalogEntry) -> Optional[ValidationDeepLink]:
    """Build schema.ValidationDeepLink strictly from entry.validation, if any."""
    v = entry.validation
    if not v:
        return None
    return ValidationDeepLink(
        deeplink=v["deeplink"],
        key=v["key"],
        resultType=v.get("resultType"),
        condition=v.get("condition"),
        value=v.get("value"),
    )


def _dummy_deeplink(catalog: DeeplinkCatalog, screen_name: str) -> Optional[Deeplink]:
    """
    dummy_positive Deeplink for an explicitly named, catalog-absent screen.
    description and message are each 5-7 words naming only `screen_name`
    (1-3 words + a fixed 4-word template); the raw query is never used.
    Returns None if the 5-7 word constraint cannot be met.
    """
    dummy = catalog.dummy_entry
    assert dummy is not None
    description = f"Opens the {screen_name} settings screen"
    message = f"Open the {screen_name} settings screen"
    if not (5 <= _word_count(description) <= 7 and 5 <= _word_count(message) <= 7):
        return None
    return Deeplink(
        deeplink=dummy.deeplink,           # voiceassist://dummy_positive, verbatim
        description=description,
        message=message,
        originalType=dummy.originalType,
        classes=None,
    )


# --- Resolution -------------------------------------------------------------
def _select_confident_match(
    matches: List[RankedMatch],
    steps_text: str,
    thresholds: Optional[EvidenceThresholds],
) -> Optional[CatalogEntry]:
    """
    First candidate that (a) is top-N on BOTH signals and (b) clears the raw
    score floors, after a stable polarity re-rank. With no explicit on/off
    wording in the step text, disabling (offURL) entries are deprioritised:
    a disabling deeplink is preferred only when the step says to disable /
    turn off. This uses only the catalog's real `originalType` field.
    """
    preferred = _infer_polarity(steps_text) or "on"
    ordered = sorted(matches, key=lambda m: _polarity_penalty(m.entry, preferred))
    for m in ordered:
        if not m.agrees_in_top(CONFIDENT_TOP_N):
            continue
        if not has_sufficient_evidence(m, thresholds):
            continue
        return m.entry
    return None


def _resolve_step_group_deeplink(
    step_group: StepGroup,
    action_category: str,
    catalog: DeeplinkCatalog,
    action_name: str = "",
    action_description: str = "",
    thresholds: Optional[EvidenceThresholds] = None,
) -> StepGroup:
    """Resolve actionableDeeplink/validationDeeplink for one StepGroup."""

    # Rule 4: manual actions never carry an actionable deeplink.
    if action_category == ActionCategory.manual:
        step_group.actionableDeeplink = None
        step_group.validationDeeplink = None
        return step_group

    steps_text = " ".join(step_group.steps)
    # Action name/description concentrate the query on-topic; raw step text
    # alone is diluted by boilerplate ("Navigate to and open Settings.").
    query_text = f"{action_name}. {action_description}. {steps_text}"
    matches: List[RankedMatch] = catalog.search(query_text, top_k=5)

    confident = _select_confident_match(matches, steps_text, thresholds)
    if confident is not None:
        step_group.actionableDeeplink = _to_actionable_deeplink(confident)
        step_group.validationDeeplink = _to_validation_deeplink(confident)
        return step_group

    # No evidence-gated catalog match. dummy_positive only when a concrete
    # Settings screen is explicitly named in the steps (and, by reaching
    # here, is absent from the catalog).
    screen_name = _extract_settings_screen_name(step_group.steps)
    if screen_name is not None:
        dummy = _dummy_deeplink(catalog, screen_name)
        if dummy is not None:
            step_group.actionableDeeplink = dummy
            step_group.validationDeeplink = None  # dummy has no validation rule
            return step_group

    step_group.actionableDeeplink = None
    step_group.validationDeeplink = None
    return step_group


def _validate_category(action: Action) -> str:
    """Category is Person 1's to set; Phase 2 preserves it (must be present)."""
    if action.category is None:
        raise ValueError(f"Action '{action.actionName}' has no category set by Phase 1")
    return action.category


_ORDER_RANK = {
    ActionCategory.auto: 0,
    ActionCategory.manual: 1,
    ActionCategory.critical: 2,
}


def map_and_order(
    goal: Goal,
    catalog: DeeplinkCatalog,
    thresholds: Optional[EvidenceThresholds] = None,
) -> Goal:
    """
    Phase 2 entry point. Mutates and returns `goal`:
      - resolves deeplinks per StepGroup (evidence-gated)
      - preserves Action.category from Phase 1
      - reorders Goal.actions: auto -> manual -> critical
    `thresholds` overrides the per-retrieval-mode evidence floors.
    """
    for action in goal.actions:
        category = _validate_category(action)
        for step_group in action.stepGroups:
            _resolve_step_group_deeplink(
                step_group,
                category,
                catalog,
                action_name=action.actionName,
                action_description=action.description,
                thresholds=thresholds,
            )

        # Rule 4 belt-and-suspenders.
        if category == ActionCategory.manual:
            for sg in action.stepGroups:
                sg.actionableDeeplink = None
                sg.validationDeeplink = None

    goal.actions.sort(key=lambda a: _ORDER_RANK.get(a.category, 99))
    return goal
