"""Deterministic rule enforcement for Phase 1.

Rule of thumb for this whole module: never trust the LLM to obey a formatting rule.
The prompt *asks*; this module *enforces*. Everything here is pure Python, has no
network calls, and runs in well under a millisecond per field.

Contents
  1. URL scrubbing (zero-leak)        scrub_urls, scrub_text_block, find_urls, assert_no_urls
  2. Casing helpers                    title_case, sentence_case
  3. Field enforcers                   enforce_description, enforce_title, build_goal,
                                       enforce_action_name, normalize_steps
  4. Category rules                    critical_rule_for, is_physical_action, ...
  5. Query normalisation / cache key   normalize_query, semantic_key
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. URL SCRUBBING
# ---------------------------------------------------------------------------

# TLDs we treat as "this is a web address" when seen without a scheme
# (e.g. "Visit samsung.com/support."). A whitelist avoids false positives on
# things like "e.g.", "One UI 6.1" or "Settings.Display".
_TLDS = (
    "com|net|org|edu|gov|mil|int|info|biz|io|co|ai|app|dev|me|tv|ly|gl|link|site|"
    "online|xyz|page|help|support|samsung|store|shop|tech|cloud|in|us|uk|kr|de|jp|"
    "cn|fr|es|it|nl|au|ca|br|ru|eu|asia"
)

_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)]*)\)")          # [text](url) / ![alt](url)
_ANGLE_URL_RE = re.compile(r"<\s*(?:https?|ftps?)://[^>]*>", re.I)  # <https://...>
_URL_BODY = r"[^\s)\]>\"'<]*"   # stop at closing brackets/quotes so "(https://x.co)." tidies cleanly
_SCHEME_URL_RE = re.compile(r"\b(?:https?|ftps?|hxxps?)\s*:\s*//" + _URL_BODY, re.I)
_WWW_RE = re.compile(r"\bwww\d{0,3}\.[^\s)\]>\"'<]+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_BARE_DOMAIN_RE = re.compile(
    r"(?<![@\w/])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:" + _TLDS + r")\b"
    r"(?::\d{2,5})?(?:/[^\s)\]]*)?",
    re.I,
)
_URL_PATTERNS = (_ANGLE_URL_RE, _SCHEME_URL_RE, _WWW_RE, _EMAIL_RE, _BARE_DOMAIN_RE)

# A sentence that contained a URL AND matches this is a "go look it up" referral.
# The whole sentence is dropped, not just the URL, otherwise we emit garbage like
# "Visit  for more details."
_REFERRAL_RE = re.compile(
    r"\b(visit|see|check out|refer to|click|browse|learn more|read more|more info(?:rmation)?|"
    r"for (?:more |further )?details|website|web ?page|web ?site|online|link|url|"
    r"support page|community|forum|download from|available at|found at)\b",
    re.I,
)
_DANGLING_TAIL_RE = re.compile(r"[\s,:;-]*\b(?:at|on|via|from|here|to|visit|see)\s*([.!?]?)\s*$", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def find_urls(text: str) -> List[str]:
    """Return every web-address-like substring in text (used by guards and tests)."""
    if not text:
        return []
    found: List[str] = []
    for m in _MD_LINK_RE.finditer(text):
        found.append(m.group(0))
    for pat in _URL_PATTERNS:
        found.extend(m.group(0) for m in pat.finditer(text))
    return found


def _strip_urls_raw(text: str) -> Tuple[str, int]:
    """Remove URLs from a string. Markdown links keep their visible text."""
    n = 0

    def _md(m: re.Match) -> str:
        nonlocal n
        n += 1
        return m.group(1)

    out = _MD_LINK_RE.sub(_md, text)
    for pat in _URL_PATTERNS:
        out, k = pat.subn("", out)
        n += k
    return out, n


def _tidy(text: str) -> str:
    text = re.sub(r"\(\s*\)|\[\s*\]|<\s*>|\{\s*\}", "", text)   # empty brackets left behind
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)                  # space before punctuation
    text = re.sub(r"([,;:])\1+", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;:-")


def scrub_urls(text: Optional[str]) -> str:
    """Remove web URLs from one short field (step, name, description...).

    Sentence-aware: a sentence that only existed to point at a URL
    ("For more help, visit samsung.com/support.") is removed entirely.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    kept: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        cleaned, n = _strip_urls_raw(sentence)
        if n:
            if _REFERRAL_RE.search(sentence):
                continue                                   # drop the referral sentence
            cleaned = _DANGLING_TAIL_RE.sub(r"\1", cleaned)
            if len(re.findall(r"[A-Za-z]{2,}", cleaned)) < 2:
                continue                                   # nothing meaningful left
        cleaned = _tidy(cleaned)
        if cleaned:
            kept.append(cleaned)
    return " ".join(kept).strip()


def scrub_text_block(text: Optional[str]) -> str:
    """Scrub a long multi-line block (e.g. the raw SIIS text) line by line.

    Used on the *input* so the LLM never even sees a URL to copy.
    """
    if not text:
        return ""
    lines = [scrub_urls(line) for line in str(text).splitlines()]
    joined = "\n".join(line for line in lines if line.strip())
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def assert_no_urls(payload: Any, path: str = "$") -> None:
    """Final hard gate. Walks any dict/list/str and raises if a web URL survived.

    `deeplink` fields are allowed to hold bixby:// URIs, but a web URL there is
    still a leak and fails the gate.
    """
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k == "deeplink" and isinstance(v, str) and v.startswith("bixby://"):
                continue
            assert_no_urls(v, f"{path}.{k}")
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            assert_no_urls(v, f"{path}[{i}]")
    elif isinstance(payload, str):
        hits = find_urls(payload)
        if hits:
            raise ValueError(f"URL leak at {path}: {hits[:3]}")


# ---------------------------------------------------------------------------
# 2. CASING HELPERS
# ---------------------------------------------------------------------------

# Canonical spellings that must survive any case conversion.
KNOWN_TERMS = {
    "wifi": "Wi-Fi", "wi-fi": "Wi-Fi", "bluetooth": "Bluetooth", "nfc": "NFC",
    "gps": "GPS", "usb": "USB", "usb-c": "USB-C", "sim": "SIM", "esim": "eSIM",
    "sd": "SD", "ui": "UI", "lte": "LTE", "5g": "5G", "4g": "4G", "hdr": "HDR",
    "amoled": "AMOLED", "aod": "AOD", "ram": "RAM", "os": "OS", "apn": "APN",
    "vpn": "VPN", "dns": "DNS", "hz": "Hz", "fps": "FPS",
    "samsung": "Samsung", "galaxy": "Galaxy", "bixby": "Bixby", "android": "Android",
    "google": "Google", "play": "Play",
}
_SMALL_WORDS = {
    "a", "an", "the", "and", "or", "nor", "but", "of", "on", "in", "to", "for",
    "at", "by", "via", "vs", "with", "from", "per", "as",
}
_WORD_RE = re.compile(r"\S+")


def _split_punct(token: str) -> Tuple[str, str, str]:
    m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", token)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")


def _is_special(core: str) -> bool:
    """Acronyms / model names / mixed case (NFC, S24, iPhone) keep their casing."""
    if not core:
        return False
    if any(ch.isdigit() for ch in core):
        return True
    letters = [c for c in core if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return True
    return any(c.isupper() for c in core[1:]) and not core[1:].isupper()


def _cap(core: str) -> str:
    return "-".join(p[:1].upper() + p[1:].lower() if p else p for p in core.split("-"))


def title_case(text: str) -> str:
    """'configure navigation bar settings' -> 'Configure Navigation Bar Settings'."""
    tokens = _WORD_RE.findall(text or "")
    out = []
    for i, tok in enumerate(tokens):
        pre, core, post = _split_punct(tok)
        low = core.lower()
        if low in KNOWN_TERMS:
            core = KNOWN_TERMS[low]
        elif _is_special(core):
            pass
        elif low in _SMALL_WORDS and 0 < i < len(tokens) - 1:
            core = low
        else:
            core = _cap(core)
        out.append(pre + core + post)
    return " ".join(out)


def sentence_case(text: str) -> str:
    """'Swipe Navigation Settings' -> 'Swipe navigation settings' (acronyms kept)."""
    tokens = _WORD_RE.findall(text or "")
    out = []
    for i, tok in enumerate(tokens):
        pre, core, post = _split_punct(tok)
        low = core.lower()
        if low in KNOWN_TERMS:
            core = KNOWN_TERMS[low]
        elif _is_special(core):
            pass
        elif i == 0:
            core = low[:1].upper() + low[1:]
        else:
            core = low
        out.append(pre + core + post)
    return " ".join(out)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


# ---------------------------------------------------------------------------
# 3. FIELD ENFORCERS
# ---------------------------------------------------------------------------

DESC_MIN_WORDS, DESC_MAX_WORDS = 5, 7
TITLE_MIN_WORDS, TITLE_MAX_WORDS = 2, 3

# Words a description/title must never end on ("It will let you choose the").
_DANGLING = {
    "a", "an", "the", "and", "or", "to", "for", "of", "on", "in", "at", "by", "with",
    "your", "you", "its", "their", "from", "that", "which", "so", "as", "is", "are",
    "be", "this", "these", "any", "all", "via", "into", "when", "if", "but", "more",
    "less", "than", "very", "can", "will", "it",
}
_THIRD_PERSON_FIX = {
    "lets": "let", "helps": "help", "allows": "allow", "enables": "enable",
    "opens": "open", "shows": "show", "fixes": "fix", "turns": "turn",
    "resets": "reset", "restores": "restore", "checks": "check", "clears": "clear",
    "stops": "stop", "reduces": "reduce", "improves": "improve", "saves": "save",
    "extends": "extend", "frees": "free", "removes": "remove", "disables": "disable",
    "adjusts": "adjust", "limits": "limit", "prevents": "prevent", "boosts": "boost",
    "refreshes": "refresh", "does": "do", "has": "have", "is": "be",
}


def _base_verb(word: str) -> str:
    low = word.lower()
    if low in _THIRD_PERSON_FIX:
        return _THIRD_PERSON_FIX[low]
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if re.search(r"(sh|ch|x|z|ss)es$", low):
        return low[:-2]
    if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
        return low[:-1]
    return low


def _lower_unless_special(word: str) -> str:
    pre, core, post = _split_punct(word)
    low = core.lower()
    if low in KNOWN_TERMS:
        return pre + KNOWN_TERMS[low] + post
    return word if _is_special(core) else pre + low + post


def _maybe_base_verb(word: str) -> str:
    """De-3rd-person a leading verb only when it clearly is one ('Erases' yes, 'Focus' no)."""
    low = word.lower()
    if low in _THIRD_PERSON_FIX:
        return _THIRD_PERSON_FIX[low]
    if len(low) > 4 and low.endswith("s") and not re.search(r"(ss|us|is|ous)$", low):
        return _base_verb(low)
    return word


# Words where a long description can be cut without breaking the phrase.
_CLAUSE_BOUNDARY = {
    "that", "which", "so", "for", "to", "when", "by", "if", "until", "while", "because",
    "since", "after", "before", "on", "in", "at", "from", "with", "without", "over", "during",
}
_ARTICLES = {"the", "a", "an"}


def _shorten(words: List[str], max_words: int, min_words: int) -> List[str]:
    """Shrink a word list to <= max_words while keeping it a complete phrase.

    1. cut at the last clause boundary that leaves min..max words
    2. otherwise drop articles
    3. otherwise hard-cut, then drop a half-finished 'and X' tail
    """
    if len(words) <= max_words:
        return words
    for p in range(max_words, min_words - 1, -1):
        if p < len(words) and words[p].lower() in _CLAUSE_BOUNDARY and words[p - 1].lower() not in _DANGLING:
            return words[:p]
    no_art = words[:2] + [w for w in words[2:] if w.lower() not in _ARTICLES]
    if len(no_art) <= max_words:
        return no_art
    for p in range(max_words, min_words - 1, -1):
        if p < len(no_art) and no_art[p].lower() in _CLAUSE_BOUNDARY and no_art[p - 1].lower() not in _DANGLING:
            return no_art[:p]
    cut = no_art[:max_words]
    if len(cut) >= 4 and cut[-2].lower() in {"and", "or"}:
        cut = cut[:-2]
    return cut


def enforce_description(text: Optional[str], subject: str = "", fallback: Optional[str] = None) -> str:
    """Force: starts with 'It will', 5-7 words, no trailing punctuation, no dangling end.

    `fallback` (a known-good description) is used when the model gave nothing usable;
    otherwise `subject` (usually the action name) builds a truthful generic sentence.
    """
    t = scrub_urls(text or "")
    t = re.sub(r"\s+", " ", t).strip(" .!?;:,\"'")
    t = t.split(". ")[0]                                   # one sentence only

    m = re.match(r"(?i)^it\s+will\b\s*(.*)$", t)
    m2 = re.match(r"(?i)^(?:(?:this|that|it|which)\s+)?(?:will|would|should|can)\s+(.*)$", t) \
        or re.match(r"(?i)^it'll\s+(.*)$", t)
    m3 = re.match(r"(?i)^(?:this|it|which)\s+(\S+)\s*(.*)$", t)
    converted = False
    if m:
        body = m.group(1)
    elif m2:
        body = m2.group(1)
    elif m3:
        # "It lets you choose..." -> "let you choose...". If the 2nd word isn't a
        # 3rd-person verb ("This setting lets...") we can't repair safely -> fallback.
        verb = m3.group(1)
        body = f"{_base_verb(verb)} {m3.group(2)}".strip() if verb.lower().endswith("s") else ""
        converted = bool(body)
    else:
        first, _, rest = t.partition(" ")            # "Erases everything..." / "Choose..."
        base = _maybe_base_verb(first) if first else ""
        converted = base.lower() != first.lower()
        body = f"{base} {rest}".strip()

    body_words = [w for w in body.replace(",", "").split() if w]
    if converted:                                      # keep parallel verbs in base form
        for i in range(1, len(body_words)):
            if body_words[i - 1].lower() in {"and", "or"}:
                body_words[i] = _maybe_base_verb(body_words[i])
    words = ["It", "will"] + [_lower_unless_special(w) for w in body_words]

    words = _shorten(words, DESC_MAX_WORDS, DESC_MIN_WORDS)
    while len(words) > DESC_MIN_WORDS and words[-1].lower() in _DANGLING:
        words.pop()

    if len(words) < DESC_MIN_WORDS or words[-1].lower() in _DANGLING:
        while len(words) > 2 and words[-1].lower() in _DANGLING:
            words.pop()
        if len(words) <= 3:                                 # only a bare verb (or nothing) left
            if fallback:
                return enforce_description(fallback)
            words = words[:2]
            subj = [_lower_unless_special(w) for w in (subject or "this setting").split()]
            words = _shorten(words + ["help", "with"] + subj, DESC_MAX_WORDS, DESC_MIN_WORDS)
            while len(words) > DESC_MIN_WORDS and words[-1].lower() in _DANGLING:
                words.pop()
        if len(words) < DESC_MIN_WORDS:
            words += ["on", "your", "device"]              # grammatical, neutral padding
        words = words[:DESC_MAX_WORDS]
    return " ".join(words)


_TITLE_FILLER = {
    "fix", "fixing", "fixes", "how", "to", "resolve", "resolving", "solve", "solving", "troubleshoot",
    "troubleshooting", "guide", "help", "the", "a", "an", "your", "my", "problem", "problems",
    "issue", "issues", "steps", "for", "of",
}


def enforce_title(title: Optional[str], topic: str = "", goal_type: str = "Troubleshooting") -> str:
    """Force: 2-3 words, sentence case, no punctuation. Drops filler before cutting."""
    t = scrub_urls(title or "")
    t = re.sub(r"[^\w\s\-/&']", " ", t)
    words = [w for w in t.split() if w]
    if len(words) > TITLE_MAX_WORDS:
        trimmed = [w for w in words if w.lower() not in _TITLE_FILLER]
        words = trimmed if len(trimmed) >= TITLE_MIN_WORDS else words
    while words and words[0].lower() in _DANGLING:
        words.pop(0)
    if len(words) > TITLE_MAX_WORDS:
        words = words[:TITLE_MAX_WORDS]
    while len(words) > TITLE_MIN_WORDS and words[-1].lower() in _DANGLING:
        words.pop()
    if len(words) < TITLE_MIN_WORDS or words[-1].lower() in _DANGLING:
        topic_words = [w for w in re.sub(r"[^\w\s\-]", " ", topic or "").split()
                       if w and w.lower() not in _TITLE_FILLER]
        if len(topic_words) >= TITLE_MIN_WORDS:
            words = topic_words[:TITLE_MAX_WORDS]
        else:
            base = [w for w in words if w.lower() not in _DANGLING] or topic_words or ["Device"]
            words = base[:1] + (["settings"] if goal_type == "Configuration" else ["troubleshooting"])
    return sentence_case(" ".join(words))


GOAL_RE = re.compile(
    r"^Follow these steps to perform this (?P<topic>[A-Z0-9][\w\-/&' ]*?) (?P<kind>Troubleshooting|Configuration)$"
)
_TOPIC_NOISE_RE = re.compile(r"(?i)\b(troubleshooting|troubleshoot|configuration|configure|guide|steps?)\b")


def normalize_goal_type(goal_type: Optional[str]) -> str:
    g = (goal_type or "").lower()
    if any(k in g for k in ("config", "setting", "setup", "set up", "customi")):
        return "Configuration"
    return "Troubleshooting"


def _topic_words(topic: str) -> List[str]:
    t = _TOPIC_NOISE_RE.sub(" ", scrub_urls(topic or ""))
    t = re.sub(r"[^\w\s\-/&']", " ", t)
    words = [w.strip("-/&'_") for w in t.split()]
    words = [w for w in words if re.search(r"[A-Za-z0-9]", w)]
    while words and words[0].lower() in {"the", "a", "an", "this", "your", "my", "and", "or", "of",
                                         "to", "on", "in", "for", "with", "it", "will"}:
        words.pop(0)
    return words[:4]


def build_goal(topic: Optional[str], goal_type: Optional[str], fallback_topic: str = "Device") -> str:
    """Build the goal string deterministically instead of trusting the model's sentence."""
    kind = normalize_goal_type(goal_type)
    for candidate in (topic, fallback_topic, "Device"):
        words = _topic_words(candidate or "")
        if words:
            goal = f"Follow these steps to perform this {title_case(' '.join(words))} {kind}"
            if GOAL_RE.match(goal):
                return goal
    return f"Follow these steps to perform this Device {kind}"


def enforce_action_name(name: Optional[str], fallback: str = "") -> str:
    n = scrub_urls(name or "")
    n = re.sub(r"(?i)^\s*(?:step|action)\s*\d+\s*[:.)-]?\s*", "", n)
    n = re.sub(r"^\s*\d+[.)]\s*", "", n)
    n = re.sub(r"[*_`#\"]", "", n)
    n = re.sub(r"\s*(?:>|→|->|›|»)\s*", " ", n)             # "Display > Navigation" -> one name
    n = re.sub(r"[.!?:;,]+$", "", n).strip()
    if not n:
        n = fallback or "Device Settings"
    return title_case(n)


# ---- Steps -----------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦·]+|\(?\d{1,2}[.)]|step\s*\d+\s*[:.)-]?|[a-z][.)](?=\s))\s*", re.I)
_NAV_SEP_RE = re.compile(r"\s*(?:>|→|->|›|»)\s*")
_IMPERATIVE = (
    r"tap|select|toggle|turn|choose|enable|disable|press|swipe|open|go|navigate|scroll|"
    r"clear|remove|restart|reboot|check|set|adjust|slide|drag|hold|touch|long-press|"
    r"uninstall|install|update|confirm|close|find|launch|allow|deny|disconnect|connect|"
    r"plug|unplug|clean|insert|wait|reset|power|enter|boot|use|try|make|keep|back"
)
# Verbs that start a *new* interaction when chained. Deliberately excludes
# hold/touch/power so "Press and hold" / "Volume down and Power keys" stay intact.
_SPLIT_VERBS = (
    r"tap|select|toggle|turn|choose|enable|disable|open|go|navigate|scroll|clear|remove|"
    r"restart|check|set|adjust|slide|drag|uninstall|install|confirm|close|find|launch|"
    r"allow|disconnect|connect|unplug|clean|insert|wait|reset|enter|press|swipe"
)
# "..., then Tap X" / "...; tap X" -> split (case-insensitive, 'then' is unambiguous).
_SPLIT_THEN_RE = re.compile(
    r"(?:,\s*(?:and\s+)?then\s+|\s+and\s+then\s+|\s+then\s+|;\s*)(?=(?:" + _SPLIT_VERBS + r"|touch|hold|long-press)\b)", re.I
)
# "... and tap X" -> split ONLY on a lowercase verb. Capitalised words after "and"
# are UI labels ("Buttons and Swipe gestures", "Clear cache and Clear data").
_SPLIT_AND_RE = re.compile(r"(?<!\bto)(?:,\s*and\s+|\s+and\s+)(?=(?:" + _SPLIT_VERBS + r")\b)")  # keeps "Navigate to and open Settings"
_NAV_VERB_RE = re.compile(r"(?i)^(go to|navigate to|open|launch|head to|access)\s+")


def _finish_step(s: str) -> str:
    s = re.sub(r"[*_`]+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-")
    if not s:
        return ""
    pre, core, post = _split_punct(s.split(" ")[0])
    first = s.split(" ")[0]
    if core and not _is_special(core) and core.lower() not in KNOWN_TERMS:
        first = pre + core[:1].upper() + core[1:] + post
    s = " ".join([first] + s.split(" ")[1:])
    if s[-1] not in ".!?":
        s += "."
    return s


def _expand_nav_chain(step: str) -> List[str]:
    """'Go to Settings > Display > Navigation bar' -> 3 single-interaction steps."""
    if not _NAV_SEP_RE.search(step):
        return [step]
    verb_m = _NAV_VERB_RE.match(step)
    body = step[verb_m.end():] if verb_m else step
    parts = [p.strip(" .") for p in _NAV_SEP_RE.split(body) if p.strip(" .")]
    if len(parts) < 2:
        return [step]
    out = []
    first = parts[0]
    if first.lower().startswith("settings"):
        out.append("Navigate to and open Settings" + first[len("settings"):])
    else:
        out.append(f"{verb_m.group(1).capitalize() if verb_m else 'Open'} {first}")
    for p in parts[1:]:
        # Capitalised segments are on-screen labels ("Reset", "Clear data"): always tap them.
        out.append(p if re.match(r"^(?:" + _IMPERATIVE + r")\b", p) else f"Tap {p}")
    return out


def normalize_steps(raw_steps: Iterable[str]) -> List[str]:
    """Scrub, de-bullet, split multi-action steps, expand 'A > B > C', dedupe."""
    out: List[str] = []
    for raw in raw_steps or []:
        if not isinstance(raw, str):
            continue
        s = scrub_urls(raw)
        s = _BULLET_RE.sub("", s).strip()
        if not s:
            continue
        for chunk in _expand_nav_chain(s):
            pieces = [p for part in _SPLIT_THEN_RE.split(chunk) for p in _SPLIT_AND_RE.split(part)]
            for piece in pieces:
                piece = _finish_step(piece)
                if piece and len(re.findall(r"[A-Za-z]{2,}", piece)) >= 1:
                    if not out or out[-1].lower() != piece.lower():
                        out.append(piece)
    return out


# ---------------------------------------------------------------------------
# 4. CATEGORY RULES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CriticalRule:
    kind: str
    rank: int                   # lower = less disruptive = sequenced earlier among criticals
    pattern: re.Pattern
    action_name: str
    description: str


# Ordered most-specific first. Descriptions are pre-validated to 5-7 words.
CRITICAL_RULES: Tuple[CriticalRule, ...] = (
    CriticalRule("factory_reset", 70,
                 re.compile(r"(?i)\bfactory\s*(?:data\s*)?reset\b|\berase\s+all\s+(?:data|content)\b|\bmaster\s+reset\b"),
                 "Factory Data Reset", "It will restore original factory device settings"),
    CriticalRule("reset_network", 50,
                 re.compile(r"(?i)\breset\s+(?:mobile\s+)?network\s+settings\b"),
                 "Reset Network Settings", "It will reset all network connection settings"),
    CriticalRule("reset_all", 60,
                 re.compile(r"(?i)\breset\s+all\s+settings\b"),
                 "Reset All Settings", "It will reset settings to their defaults"),
    CriticalRule("recovery", 40,
                 re.compile(r"(?i)\bwipe\s+cache\s+partition\b|\brecovery\s+mode\b"),
                 "Recovery Mode", "It will clear the system cache partition"),
    CriticalRule("software_update", 30,
                 re.compile(r"(?i)\b(?:software|firmware|system|os)\s+updates?\b"),
                 "Software Update", "It will install the latest software update"),
    CriticalRule("safe_mode", 20,
                 re.compile(r"(?i)\bsafe\s+mode\b"),
                 "Safe Mode", "It will check for problematic downloaded apps"),
    CriticalRule("restart", 10,
                 re.compile(r"(?i)\b(?:restart|reboot)\s+(?:your\s+|the\s+)?(?:phone|device|galaxy|mobile|smartphone)\b"
                            r"|\breboot\b|\bforce\s+restart\b|\bpower\s+off\s+(?:your\s+|the\s+)?(?:phone|device)\b"
                            r"|\bturn\s+(?:your\s+|the\s+)?(?:phone|device)\s+off\b|^\s*restart\s*$"
                            r"|\b(?:tap|select|press)\s+restart\s*[.!]?\s*$"),
                 "Restart Device", "It will refresh the system by restarting"),
)
_STEP_STARTS_IMPERATIVE = re.compile(
    r"(?i)^\W*(?:tap|select|go|open|perform|restart|reboot|reset|boot|enter|turn|power|press|force|touch|hold|"
    r"install|download|update|start|erase|wipe|choose|run|do|back up)\b"
)

_PHYSICAL_RE = re.compile(
    r"(?i)\b(clean|wipe\s+(?:the\s+)?(?:port|lens|screen|camera|sensor)|dust|lint|debris|"
    r"compressed\s+air|toothpick|brush|cloth|remove\s+(?:the\s+)?(?:case|cover|screen\s+protector|sim|sd\s+card|battery)|"
    r"take\s+off|original\s+charger|charging\s+cable|cable|wall\s+(?:outlet|socket)|adapter|"
    r"service\s+cent(?:er|re)|repair|replace|moisture|let\s+it\s+dry|cool\s+down|unplug|plug\s+in)\b"
)
_SETTINGS_NAV_RE = re.compile(r"(?i)\b(?:open|launch|go\s+to|navigate\s+to)\s+(?:and\s+open\s+)?settings\b|^\s*settings\b")


def critical_rule_for(text: str) -> Optional[CriticalRule]:
    for rule in CRITICAL_RULES:
        if rule.pattern.search(text or ""):
            return rule
    return None


def critical_rule_for_step(step: str) -> Optional[CriticalRule]:
    """Stricter step-level check: the step must *instruct* the disruptive action."""
    if not _STEP_STARTS_IMPERATIVE.search(step or ""):
        return None
    return critical_rule_for(step)


def is_physical_action(text: str) -> bool:
    return bool(_PHYSICAL_RE.search(text or ""))


def navigates_settings(steps: Iterable[str]) -> bool:
    return any(_SETTINGS_NAV_RE.search(s or "") for s in steps)


# ---------------------------------------------------------------------------
# 5. QUERY NORMALISATION / DETERMINISTIC SEMANTIC KEY
# ---------------------------------------------------------------------------
# This is the *local*, zero-latency cache key (tier 1). Phase 3 adds dense
# embeddings (tier 2) for paraphrases this map does not cover.

_PHRASE_CANON: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p, re.I), rep) for p, rep in (
        (r"\b(?:dies|dying|drains?|draining|drained|runs?\s+out|losing\s+charge|goes?\s+down)\s+(?:so\s+|really\s+|too\s+|very\s+)?(?:fast|quick(?:ly)?)\b", " battery drain "),
        (r"\bbattery\s+(?:is\s+)?(?:draining|drain(?:s|ed)?|dies|dying)\b", " battery drain "),
        (r"\b(?:laggy|lagging|lags?|sluggish|slow(?:ed)?(?:\s+down)?|hangs?|hanging|stutter\w*|freez\w*|frozen)\b", " slow "),
        (r"\b(?:flicker\w*|flashing|blinking|blinks?)\b", " flicker "),
        (r"\b(?:gestures?|swip\w*)\b", " gesture "),
        (r"\b(?:overheat\w*|heating\s+up|gets?\s+(?:hot|warm)|too\s+hot)\b", " overheat "),
        (r"\b(?:blurr?y|blurr?ed|out\s+of\s+focus|not\s+focus\w*)\b", " blur "),
        (r"\b(?:application|apps)\b", " app "),
        (r"\b(?:downloading|installing|downloaded|installed|download|install)\b", " install "),
        (r"\b(?:upgrade|os\s+update|software\s+update|system\s+update|updated)\b", " update "),
        (r"\b(?:mobile|smartphone|handset|cellphone|device)\b", " phone "),
        (r"\b(?:vertical(?:ly)?|up\s+(?:and|or|&)\s+down|up/down)\b", " vertical "),
        (r"\b(?:horizontal(?:ly)?|sideways|left\s+(?:and|or|&)\s+right|left/right)\b", " horizontal "),
    )
)
_STOPWORDS = {
    "a", "an", "the", "my", "i", "me", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "and", "or", "but", "so", "to", "of", "on", "in", "at", "for", "with",
    "after", "since", "when", "why", "what", "how", "do", "does", "did", "can", "could",
    "should", "would", "will", "just", "really", "very", "too", "now", "got", "get", "gets",
    "keep", "keeps", "still", "any", "some", "new", "recent", "recently", "yesterday", "ever",
    "has", "have", "had", "instead", "of", "please", "help", "samsung", "galaxy", "phone",
    "issue", "problem", "annoying", "anymore", "everything", "all", "up", "down", "way",
}


def normalize_query(query: str) -> str:
    """Lowercase, strip URLs/punctuation, squash char repeats, map colloquialisms."""
    q = unicodedata.normalize("NFKC", query or "").lower()
    q, _ = _strip_urls_raw(q)
    q = re.sub(r"(.)\1{2,}", r"\1\1", q)                  # "sooooo" -> "soo"
    q = q.replace("’", "'")
    q = re.sub(r"[^a-z0-9/'&\s-]", " ", q)
    for pat, rep in _PHRASE_CANON:
        q = pat.sub(rep, q)
    return re.sub(r"\s+", " ", q).strip()


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def semantic_key(query: str) -> str:
    """Order-insensitive content-token key: paraphrases with the same intent collide."""
    toks = {_stem(t) for t in re.findall(r"[a-z0-9]+", normalize_query(query)) if t not in _STOPWORDS and len(t) > 1}
    return "|".join(sorted(toks))
