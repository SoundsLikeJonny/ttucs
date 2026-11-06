"""Text utilities shared by data preparation, models and the inference pipeline."""

from __future__ import annotations

import re

from ttucs.slate.normalize import normalize_transcript

STOPWORDS = frozenset("""
a an the and or of to in on at for with by from into onto over under is are was were be been
it its this that these those there here some any very really just then than as up down out off
we i you he she they my our your his her their me us them so um uh like okay ok also plus
recording recorded record take sound sounds noise being going gonna get getting got
""".split())

_KEYWORD_TAIL = re.compile(r"(?:^|(?<=[.!?]))\s*((?:[A-Z0-9][\w'&/-]*(?:\s[\w'&/-]+){0,3},\s*){2,}"
                           r"[A-Z0-9][\w'&/-]*(?:\s[\w'&/-]+){0,3}\.?)\s*$")


def split_description(description: str) -> tuple[str, list[str]]:
    """Split a library description into its prose part and trailing keyword list.

    Professional libraries (e.g. BOOM ONE) write ``"<sentences>. Kw1, Kw2, Kw3."``.

    >>> split_description("Wooden door creaks open. Door, Wood, Creak, Open.")
    ('Wooden door creaks open.', ['Door', 'Wood', 'Creak', 'Open'])
    """
    description = (description or "").strip()
    m = _KEYWORD_TAIL.search(description)
    if not m:
        return description, []
    prose = description[: m.start(1)].strip()
    keywords = [k.strip(" .") for k in m.group(1).split(",") if k.strip(" .")]
    if not prose:  # description is *only* keywords
        return description, keywords
    return prose, keywords


def preprocess(text: str) -> str:
    """Canonical model input: STT-normalised, lower-case, single-spaced."""
    text = normalize_transcript(text)
    text = re.sub(r"[^\w\s'&-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_take_numbers(fx_name: str) -> str:
    """'Augusta A109 Fly Away 01' -> 'Augusta A109 Fly Away'."""
    return re.sub(r"(?:\s+\d{1,3})+$", "", fx_name.strip())


def content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9'&-]*", text.lower())
            if t not in STOPWORDS and len(t) > 1]


def make_fx_name(description: str, max_chars: int = 25) -> str:
    """Build a short, title-cased UCS ``FXName`` from a description.

    UCS recommends ~25 characters; we greedily keep content words in order until the
    budget is reached (always at least one word).
    """
    words = content_tokens(description)
    out: list[str] = []
    for w in words:
        if w in out:
            continue
        cand = " ".join([*out, w])
        if out and len(cand) > max_chars:
            break
        out.append(w)
    return " ".join(w.upper() if len(w) <= 3 and any(c.isdigit() for c in w) else w.capitalize()
                    for w in out) or "Untitled"


def make_keywords(description: str, extra: list[str] | None = None, limit: int = 15) -> list[str]:
    """Unique, title-cased keywords from the description plus optional extra terms."""
    seen: dict[str, None] = {}
    for w in content_tokens(description) + [e.lower() for e in (extra or [])]:
        seen.setdefault(w, None)
    return [w.title() for w in list(seen)[:limit]]


def initials(name: str | None) -> str:
    """'Jon Evans' -> 'JE'."""
    if not name:
        return ""
    return "".join(p[0].upper() for p in re.findall(r"[A-Za-z0-9]+", name))[:6]


def abbreviate(name: str | None, max_len: int = 16) -> str:
    """Short SourceID from a project name: 'Thick As Thieves' -> 'ThickAsThieves'."""
    if not name:
        return ""
    s = "".join(p[:1].upper() + p[1:] for p in re.findall(r"[A-Za-z0-9]+", name))
    if len(s) <= max_len:
        return s
    return initials(name)
