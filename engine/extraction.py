"""Phase [1] Structure Extraction.

Input : complaint + raw SIIS reference text
Output: ContextDeeplinkResponse (schema-valid, URL-free, rule-compliant) plus
        per-action screen hints for Phase 2 deeplink mapping.

Pipeline
  scrub SIIS input -> LLM (intermediate JSON) -> parse/repair -> per field:
  scrub + enforce -> ground every step against the reference (anti-hallucination)
  -> resolve categories -> split embedded critical steps -> merge same-screen
  actions -> order (critical last) -> score -> validate with schema.py ->
  final URL gate.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .llm_client import LLMClient, LLMParseError, Usage, parse_llm_json
from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE, REPAIR_TEMPLATE
from .sanitizers import (CRITICAL_RULES, CriticalRule, assert_no_urls, build_goal, critical_rule_for,
                         critical_rule_for_step, enforce_action_name, enforce_description, enforce_title,
                         is_physical_action, navigates_settings, normalize_goal_type, normalize_steps,
                         scrub_text_block, scrub_urls)
from .schema import Action, ContextDeeplinkResponse, Goal, StepGroup, actionCategory

log = logging.getLogger(__name__)

MAX_REFERENCE_CHARS = 12_000      # bounds prompt cost + latency on huge SIIS articles
MAX_GOALS = 3
GROUNDING_THRESHOLD = 0.5         # share of a step's content words that must appear in the reference

FALLBACK_NO_SIIS = "no_siis_context"
FALLBACK_NO_MATCH = "no_match"
FALLBACK_LLM_ERROR = "llm_error"


# ---------------------------------------------------------------------------
# Lenient intermediate models for the LLM output
# ---------------------------------------------------------------------------
class _DraftStepGroup(BaseModel):
    model_config = ConfigDict(extra="ignore")
    steps: List[str] = []

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce(cls, v):
        if isinstance(v, str):
            return [s for s in v.splitlines() if s.strip()]
        return [str(s) for s in (v or []) if isinstance(s, (str, int, float))]


class _DraftAction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    actionName: str = ""
    description: str = ""
    category: str = Field("manual", json_schema_extra={"enum": ["auto", "manual", "critical"]})
    screen_path: Optional[str] = ""
    stepGroups: List[_DraftStepGroup] = []

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_steps(cls, data):
        # Models sometimes emit {"steps": [...]} directly on the action.
        if isinstance(data, dict) and not data.get("stepGroups") and data.get("steps"):
            data = {**data, "stepGroups": [{"steps": data["steps"]}]}
        return data


class _DraftGoal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    topic: str = ""
    goal_type: str = Field("Troubleshooting", json_schema_extra={"enum": ["Troubleshooting", "Configuration"]})
    title: str = ""
    confidence: float = Field(0.5, json_schema_extra={"minimum": 0, "maximum": 1})
    actions: List[_DraftAction] = []

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.5


class _DraftExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    match: bool = True
    reason: Optional[str] = None
    goals: List[_DraftGoal] = []


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ScreenHint:
    """Everything Phase 2 needs to match an action to a catalog deeplink.

    Phase 2 must match these against catalog metadata (description / message /
    qna_description), never against the masked URI string.
    """
    goal_index: int
    action_index: int
    action_name: str
    description: str
    category: str
    screen_path: str
    steps: List[str]


@dataclass
class ExtractionResult:
    response: ContextDeeplinkResponse
    screen_hints: List[ScreenHint]
    usage: Usage
    fallback: Optional[str] = None           # None | no_match | no_siis_context | llm_error
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grounding (anti-hallucination)
# ---------------------------------------------------------------------------
_GENERIC = {
    # stopwords
    "a", "an", "the", "and", "or", "to", "of", "on", "in", "at", "for", "with", "by", "from", "your",
    "you", "it", "its", "this", "that", "is", "are", "be", "if", "then", "as", "into", "any", "all",
    "between", "under", "until", "again", "there", "here", "up", "down", "off", "out",
    # generic UI verbs/nouns every guide contains
    "tap", "open", "select", "navigate", "go", "toggle", "turn", "press", "swipe", "choose", "enable",
    "disable", "launch", "find", "scroll", "optionally", "next", "back", "settings", "setting", "app",
    "apps", "phone", "device", "screen", "option", "options", "menu", "button", "icon", "confirm",
    "ok", "done", "check", "make", "sure", "want", "preferred", "desired", "display", "switch",
}


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t


def _content_tokens(text: str) -> List[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _GENERIC and len(t) > 1]


def grounding_coverage(step: str, reference_tokens: set) -> float:
    toks = _content_tokens(step)
    if not toks:
        return 1.0                 # "Tap Display." style steps carry no claim to verify
    return sum(1 for t in toks if t in reference_tokens) / len(toks)


# ---------------------------------------------------------------------------
# Category resolution
# ---------------------------------------------------------------------------
def _norm_category(value: str) -> actionCategory:
    v = (value or "").strip().lower()
    if v in ("auto", "automatic", "settings"):
        return actionCategory.auto
    if v in ("critical", "destructive", "disruptive"):
        return actionCategory.critical
    return actionCategory.manual


def resolve_category(llm_category: str, name: str, description: str, steps: List[str]) -> actionCategory:
    """LLM proposes, rules decide. Critical keywords always win (they must go last)."""
    if critical_rule_for(f"{name} {description}"):
        return actionCategory.critical
    cat = _norm_category(llm_category)
    all_text = " ".join([name, description, *steps])
    if cat == actionCategory.auto and is_physical_action(all_text) and not navigates_settings(steps):
        return actionCategory.manual              # physical task mislabelled as a screen
    if cat == actionCategory.manual and navigates_settings(steps) and not is_physical_action(all_text):
        return actionCategory.auto                # a Settings screen mislabelled as manual
    return cat


def _critical_rank(action: Action) -> int:
    rule = critical_rule_for(f"{action.actionName} {action.description}") or critical_rule_for(
        " ".join(s for g in action.stepGroups for s in g.steps))
    return rule.rank if rule else 35               # unknown critical: middle of the pack


# ---------------------------------------------------------------------------
# Goal post-processing
# ---------------------------------------------------------------------------
@dataclass
class _WorkingAction:
    name: str
    description: str
    category: actionCategory
    screen_path: str
    groups: List[List[str]]
    coverage: List[float]


def _screen_key(a: _WorkingAction) -> str:
    base = a.screen_path or a.name
    return re.sub(r"[^a-z0-9]", "", base.lower())


def _process_goal(draft: _DraftGoal, reference_tokens: set, warnings: List[str]) -> Tuple[Optional[Goal], List[_WorkingAction]]:
    working: List[_WorkingAction] = []
    split_out: List[_WorkingAction] = []

    for d in draft.actions:
        screen_path = scrub_urls(d.screen_path or "")
        fallback_name = screen_path.split(">")[-1].strip() if screen_path else ""
        name = enforce_action_name(d.actionName, fallback=fallback_name)

        groups: List[List[str]] = []
        coverage: List[float] = []
        for g in d.stepGroups:
            kept = []
            for step in normalize_steps(g.steps):
                cov = grounding_coverage(step, reference_tokens)
                if cov < GROUNDING_THRESHOLD:
                    warnings.append(f"dropped_ungrounded_step: {step[:80]!r} (coverage {cov:.2f})")
                    continue
                kept.append(step)
                coverage.append(cov)
            if kept:
                groups.append(kept)
        if not groups:
            warnings.append(f"dropped_empty_action: {name!r}")
            continue

        flat = [s for g in groups for s in g]
        category = resolve_category(d.category, name, d.description, flat)

        # Pull embedded disruptive steps ("...then restart your phone") into their own
        # critical action so they are sequenced last.
        if category != actionCategory.critical:
            buckets: dict = {}
            new_groups = []
            for g in groups:
                keep = []
                for s in g:
                    rule = critical_rule_for_step(s)
                    if rule:
                        buckets.setdefault(rule.kind, (rule, []))[1].append(s)
                    else:
                        keep.append(s)
                if keep:
                    new_groups.append(keep)
            if buckets and not new_groups:
                category = actionCategory.critical   # every step was disruptive: whole action is critical
            elif buckets:
                groups = new_groups
                for rule, steps in buckets.values():
                    split_out.append(_WorkingAction(rule.action_name, rule.description, actionCategory.critical,
                                                    "", [steps], [1.0] * len(steps)))
                    warnings.append(f"split_critical_step_into_action: {rule.action_name}")

        known = critical_rule_for(name) if category == actionCategory.critical else None
        working.append(_WorkingAction(
            name=name,
            description=enforce_description(d.description, subject=name,
                                            fallback=known.description if known else None),
            category=category,
            screen_path=screen_path if category != actionCategory.manual else "",
            groups=groups,
            coverage=coverage,
        ))

    working.extend(split_out)

    # One Action = One Screen: merge actions that target the same screen.
    merged: List[_WorkingAction] = []
    index = {}
    for a in working:
        key = _screen_key(a)
        if key in index:
            tgt = merged[index[key]]
            existing = [set(_content_tokens(s)) for g in tgt.groups for s in g]
            existing_text = {s.lower() for g in tgt.groups for s in g}
            for g in a.groups:
                for st in g:
                    toks = set(_content_tokens(st))
                    redundant = st.lower() in existing_text or (toks and any(toks <= e for e in existing))
                    if not redundant:
                        tgt.groups[-1].append(st)          # same screen -> same step group
                        existing.append(toks)
                        existing_text.add(st.lower())
            tgt.coverage += a.coverage
            if a.category == actionCategory.critical:
                tgt.category = actionCategory.critical
            warnings.append(f"merged_same_screen_actions: {a.name!r} -> {tgt.name!r}")
        else:
            index[key] = len(merged)
            merged.append(a)

    if not merged:
        return None, []

    actions = [
        Action(
            actionName=a.name,
            description=a.description,
            category=a.category,
            # Phase 1 never attaches deeplinks; manual actions must never carry one.
            stepGroups=[StepGroup(steps=g, actionableDeeplink=None, validationDeeplink=None) for g in a.groups],
        )
        for a in merged
    ]
    # Stable sort: non-critical keep the model's (reference) order; critical go last by severity.
    order = sorted(range(len(actions)), key=lambda i: (
        actions[i].category == actionCategory.critical,
        _critical_rank(actions[i]) if actions[i].category == actionCategory.critical else 0,
        i,
    ))
    actions = [actions[i] for i in order]
    merged = [merged[i] for i in order]

    goal_type = normalize_goal_type(draft.goal_type)
    topic = draft.topic or draft.title or merged[0].name
    all_cov = [c for a in merged for c in a.coverage] or [1.0]
    confidence = min(max(draft.confidence, 0.0), 1.0)
    score = round(min(max(0.6 * confidence + 0.4 * (sum(all_cov) / len(all_cov)), 0.0), 1.0), 2)

    goal = Goal(
        goal=build_goal(topic, goal_type, fallback_topic=merged[0].name),
        title=enforce_title(draft.title, topic=topic, goal_type=goal_type),
        actions=actions,
        score=score,
    )
    return goal, merged


# ---------------------------------------------------------------------------
# LLM call with one repair round
# ---------------------------------------------------------------------------
def _call_extraction(llm: LLMClient, usage: Usage, user: str) -> _DraftExtraction:
    history = None
    last_err = None
    for _attempt in range(2):
        prompt = user if history is None else REPAIR_TEMPLATE.format(error=last_err)
        reply = llm.complete(EXTRACTION_SYSTEM_PROMPT, prompt, max_tokens=1800, history=history,
                             response_schema=_DraftExtraction)
        usage.add(reply)
        try:
            return _DraftExtraction.model_validate(parse_llm_json(reply.text))
        except (LLMParseError, ValidationError) as e:
            last_err = str(e)[:300]
            history = [{"role": "user", "content": user}, {"role": "assistant", "content": reply.text}]
    raise LLMParseError(f"extraction JSON invalid after repair: {last_err}")


def _empty(usage: Usage, fallback: str, warnings: List[str]) -> ExtractionResult:
    return ExtractionResult(ContextDeeplinkResponse(contexts=[]), [], usage, fallback, warnings)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_structure(query: str, siis_response: Optional[str], llm: LLMClient,
                      canonical_query: Optional[str] = None) -> ExtractionResult:
    usage = Usage(model=getattr(llm, "model", ""))
    warnings: List[str] = []

    reference = scrub_text_block(siis_response)[:MAX_REFERENCE_CHARS]
    if not reference.strip():
        return _empty(usage, FALLBACK_NO_SIIS, warnings)

    canonical_line = f"Canonical form: {canonical_query}" if canonical_query else ""
    user = EXTRACTION_USER_TEMPLATE.format(query=scrub_urls(query), canonical_line=canonical_line,
                                           reference=reference)
    try:
        draft = _call_extraction(llm, usage, user)
    except Exception as e:
        log.warning("extraction failed: %s", e)
        warnings.append(f"extraction_llm_failed: {type(e).__name__}: {str(e)[:120]}")
        return _empty(usage, FALLBACK_LLM_ERROR, warnings)

    if not draft.match or not draft.goals:
        if draft.reason:
            warnings.append(f"model_no_match: {scrub_urls(draft.reason)[:120]}")
        return _empty(usage, FALLBACK_NO_MATCH, warnings)

    reference_tokens = set(_content_tokens(reference))
    built: List[Tuple[Goal, List[_WorkingAction]]] = []
    for dg in draft.goals[:MAX_GOALS]:
        goal, working = _process_goal(dg, reference_tokens, warnings)
        if goal is not None:
            built.append((goal, working))

    if not built:
        return _empty(usage, FALLBACK_NO_MATCH, warnings)

    built.sort(key=lambda gw: gw[0].score, reverse=True)
    response = ContextDeeplinkResponse(contexts=[g for g, _ in built])

    # Final hard gates: schema round-trip + zero URL leak.
    response = ContextDeeplinkResponse.model_validate(response.model_dump(mode="json"))
    assert_no_urls(response.model_dump(mode="json"))

    hints = [
        ScreenHint(gi, ai, act.actionName, act.description, act.category.value,
                   w.screen_path, [s for grp in act.stepGroups for s in grp.steps])
        for gi, (goal, working) in enumerate(built)
        for ai, (act, w) in enumerate(zip(goal.actions, working))
    ]
    return ExtractionResult(response, hints, usage, None, warnings)
