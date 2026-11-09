"""Rule-based parser for spoken field-recording slates.

A slate such as::

    "um okay project thick as thieves, location squamish forest, recording a wooden door
     creak slow, mkh four sixteen into a mix pre six, ninety six k twenty four bit, take three"

is split into structured fields (project, location, gear, sample rate, take, ...) and a
*description* - the only part that is fed to the UCS classifier. Parsed metadata is
returned alongside predictions but never used for training.

The parser works in three passes over the normalised transcript:

1. **hard spans** - regexes for unambiguous technical tokens (gear lexicon, ``96 k``,
   ``24 bit``, ``take 3``, dates, perspectives);
2. **cue spans** - phrases such as ``project``/``location``/``recording`` whose value runs
   until the next span, cue or clause boundary;
3. **description** - the explicit description cue if present, otherwise the content-bearing
   leftover clauses joined in spoken order (so multi-part ambience slates classify in full).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ttucs.slate.gear import ALL_GEAR
from ttucs.slate.normalize import MONTHS, normalize_transcript

# ----------------------------------------------------------------------------- fields


@dataclass
class SlateFields:
    """Structured result of parsing one slate transcript."""

    raw_text: str
    normalized_text: str = ""
    description: str = ""
    project: str | None = None
    location: str | None = None
    microphones: list[str] = field(default_factory=list)
    mic_types: list[str] = field(default_factory=list)
    mic_setup: list[str] = field(default_factory=list)
    recorder: str | None = None
    sample_rate: int | None = None
    bit_depth: str | None = None
    take: str | None = None
    scene: str | None = None
    slate: str | None = None
    perspective: list[str] = field(default_factory=list)
    date: str | None = None
    creator: str | None = None
    notes: str | None = None
    unparsed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def user_data(self) -> str | None:
        """Compact gear string suitable for the UCS ``UserData`` filename block."""
        parts = [re.sub(r"^(Sennheiser|Schoeps|Neumann|Sound Devices|Audio-Technica) ", "", m)
                 for m in self.microphones]
        parts += self.mic_setup
        return "-".join(p.replace(" ", "") for p in parts) or None


@dataclass
class _Span:
    start: int
    end: int
    field: str
    value: str
    priority: int


# ---------------------------------------------------------------------------- patterns
_CLAUSE_END = re.compile(r"[,;!?]|\.(?!\d)| - ")
_LOCATION_NOUNS = re.compile(
    r"\b(?:beach|forest|woods|park|studio(?: [a-z])?|stage|kitchen|garage|cabin|canyon|warehouse|yard|"
    r"village|church|shop|cove|street|road|room|house|field|lake|river|creek|harbou?r|port|marina|"
    r"station|airport|garden|backyard|basement|office|factory|farm|mountain|valley|desert|city|"
    r"downtown|tunnel|bridge|parking lot|seawall|alley|barn|school|mall|market|highway)\b")
_GLUE = {"a", "an", "the", "and", "with", "on", "into", "in", "at", "using", "through",
         "via", "of", "to", "is", "it's", "its", "we're", "were", "from", "for", "by", "my",
         "our", "some", "then", "also", "plus", "so", "that", "this", "running", "recorded",
         "recording", "going", "go", "set", "up", "here", "there", "now", "okay", "ok", "uh",
         "pair", "rig", "mic", "mics", "recorder", "as", "well", "just", "be", "will",
         "gonna", "rolling", "speed", "slate", "marker", "sound", "sounds", "speeding",
         "i", "i'm", "i've", "we", "we've", "got", "setup"}

# Perspective words that are metadata on their own (already captured in ``perspective``);
# they must not be joined into the description or reported as unparsed.
_PERSPECTIVE_ONLY = re.compile(
    r"(?:exterior|interior|outside|inside|outdoors?|indoors?|ext|int|close[- ]?up|cu|med|dist)\b")

# Brand/capsule remnants of mic mentions the lexicon could not fully parse
# ("i'm using an sennheiser") - kept out of the description, left in ``unparsed``.
_BRAND_WORDS = {"sennheiser", "schoeps", "rode", "zoom", "tascam", "nagra", "sony", "sanken",
                "neumann", "shure", "dpa", "lom", "technica", "devices", "mixpre",
                "mkh", "mk", "cmc", "cmit", "ccm", "ntg", "kmr", "mke"}

_PERSPECTIVES = [
    (re.compile(r"\bclose up\b|\bclose[- ]miked\b|\bclose perspective\b|\bperspective close\b|"
                r"\bcu\b|\bclose mics?\b"), "CU"),
    (re.compile(r"\bmedium (?:perspective|distance|shot)\b|\bperspective medium\b|\bmed\b"), "MED"),
    (re.compile(r"\bdistant (?:perspective|miking|mic)\b|\bperspective (?:distant|far)\b|"
                r"\bfar perspective\b|\bdist\b"), "DIST"),
    (re.compile(r"\bonboard\b|\bon[- ]board\b"), "OB"),
    (re.compile(r"\binterior\b|\binside\b|\bint\b"), "INT"),
    (re.compile(r"\bexterior\b|\boutside\b|\bext\b|\boutdoors?\b"), "EXT"),
]
# perspective phrases that are *only* metadata (removed from the description)
_PERSPECTIVE_META = re.compile(
    r"\b(?:perspective (?:close|medium|distant|far)|(?:close|medium|distant|far) perspective|"
    r"close[- ]miked|close up|cu|med|dist)\b")

_HARD: list[tuple[str, re.Pattern[str], int]] = [
    ("sample_rate", re.compile(
        r"\b(?:sample rate|sampling rate|at)?\s?(44\.1|48|88\.2|96|176\.4|192|384)\s?(?:k\b|000\b)"), 9),
    ("sample_rate", re.compile(r"\b(?:sample rate|sampling at)\s(44\.1|48|88\.2|96|192)\b"), 9),
    ("bit_depth", re.compile(r"\b(16|24|32)\s?bit(?:\s?(float|floating point))?\b"), 9),
    ("bit_depth", re.compile(r"\b(32)\s?(float)\b"), 9),
    ("take", re.compile(r"\btake (?:number |no\.? |#)?(\d+\w?)\b"), 9),
    ("scene", re.compile(r"\bscene (?:number |no\.? |#)?(\d+\w?)\b"), 9),
    ("slate", re.compile(r"\bslate (?:number |no\.? |#)?(\d+)\b"), 9),
    ("date", re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b"), 8),
    ("date", re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"), 8),
    ("date", re.compile(
        r"\b(?:date(?: is)?\s|today is\s|it's\s)?((?:" + "|".join(MONTHS) + r")\s\d{1,2}(?:st|nd|rd|th)?"
        r"(?:,?\s(?:\d{4}|\d{2}\s?\d{2}))?)\b"), 8),
    ("date", re.compile(
        r"\b(\d{1,2}(?:st|nd|rd|th)? of (?:" + "|".join(MONTHS) + r")(?:,?\s\d{4})?)\b"), 8),
]

# Cues introduce free-text values; the value runs until the next span/cue/clause end.
_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("project", re.compile(r"\b(?:project(?: name)?(?: is| called)?|production(?: is)?|"
                           r"for the (?:project|show|game|film)|show(?: is| called)?|"
                           r"for (?:project|show)|this is for|library(?: is| called)?)\s")),
    ("location", re.compile(r"\b(?:location(?: is)?|loc|on location (?:at|in)|recorded (?:at|in)|"
                            r"recording (?:at|in)(?! (?:stereo|mono|ms|xy|ortf|ambisonic))|"
                            r"we(?:'re| are) (?:at|in)|i(?:'m| am) (?:at|in)|here (?:at|in)|"
                            r"located (?:at|in)|place(?: is)?|venue(?: is)?|site(?: is)?)\s")),
    ("creator", re.compile(r"\b(?:recordist(?: is)?|recorded by|recording by|my name is|"
                           r"slated by|sound recordist(?: is)?|engineer(?: is)?|creator(?: is)?)\s")),
    ("notes", re.compile(r"\b(?:notes?(?: are| is)?|comment(?: is)?|fyi)(?::\s?|\s)")),
    ("description", re.compile(r"\b(?:recording(?: of| some| a| an| the)?|this is(?: a| an| the| some)?|"
                               r"sound of|sounds of|description(?: is)?|subject(?: is)?|"
                               r"capturing|getting(?: some)?|we(?:'ve| have) got|we have|"
                               r"fx(?: is)?|effect(?: is)?|source(?: is)?)\s")),
]
_CUE_PRIORITY = {"project": 6, "location": 6, "creator": 6, "notes": 5, "description": 4}


def _title(text: str) -> str:
    small = {"a", "an", "the", "of", "and", "in", "on", "at", "to", "for", "by"}
    words = text.split()
    return " ".join(w if (i and w in small) else (w[:1].upper() + w[1:]) for i, w in enumerate(words))


def _strip_glue(text: str) -> str:
    words = [w for w in re.split(r"\s+", text.strip(" ,.;:-")) if w]
    while words and words[0] in _GLUE:
        words.pop(0)
    while words and words[-1] in _GLUE:
        words.pop()
    return " ".join(words)


def _content_words(text: str) -> int:
    return sum(1 for w in text.split() if w not in _GLUE and not w.isdigit())


def _perspective_only(text: str) -> bool:
    return bool(_PERSPECTIVE_ONLY.fullmatch(text))


def _gear_fragment(text: str) -> bool:
    words = [w for w in re.split(r"\s+", text) if w not in _GLUE and not w.isdigit()]
    return bool(words) and all(w in _BRAND_WORDS for w in words)


# ------------------------------------------------------------------------------ parser
class SlateParser:
    """Parse STT slate transcripts into :class:`SlateFields`.

    Parameters
    ----------
    min_description_words:
        Leftover clauses with fewer content words are not used as description.
    """

    def __init__(self, min_description_words: int = 1) -> None:
        self.min_description_words = min_description_words

    def __call__(self, text: str) -> SlateFields:
        return self.parse(text)

    def parse(self, text: str) -> SlateFields:
        norm = normalize_transcript(text or "")
        out = SlateFields(raw_text=text or "", normalized_text=norm)
        if not norm:
            return out

        spans = self._hard_spans(norm, out)
        spans += self._cue_spans(norm, spans, out)
        out.perspective = self._perspectives(norm)

        desc_spans = [s for s in spans if s.field == "description" and s.value]
        if desc_spans:
            out.description = " ".join(s.value for s in desc_spans)
        leftovers = [c for c in self._leftovers(norm, spans) if not _perspective_only(c)]
        if not out.description:
            # No explicit cue: every content-bearing leftover is part of what was recorded
            # ("ambience nearby traffic, crossing sign, children playing, wind"), so join them
            # for the classifier instead of keeping only the longest clause.
            parts = [c for c in leftovers
                     if _content_words(c) > 0 and not _gear_fragment(c)]
            if parts and _content_words(" ".join(parts)) >= self.min_description_words:
                out.description = " ".join(parts)
                leftovers = [c for c in leftovers if c not in parts]
        out.description = _PERSPECTIVE_META.sub(" ", out.description)
        out.description = re.sub(r"\s+", " ", _strip_glue(out.description)).strip()
        if not out.description:  # nothing recognisable: classify the whole utterance
            out.description = _strip_glue(norm)
        out.unparsed = [x for x in leftovers if _content_words(x) > 0]
        return out

    # -------------------------------------------------------------------- internals
    def _hard_spans(self, norm: str, out: SlateFields) -> list[_Span]:
        spans: list[_Span] = []
        for gp in ALL_GEAR:
            for m in gp.regex.finditer(norm):
                spans.append(_Span(m.start(), m.end(), gp.kind, gp.fmt(m).strip(), 10))
        for name, rx, prio in _HARD:
            for m in rx.finditer(norm):
                val = " ".join(g for g in m.groups() if g)
                spans.append(_Span(m.start(), m.end(), name, val, prio))
        for m in _PERSPECTIVE_META.finditer(norm):
            spans.append(_Span(m.start(), m.end(), "perspective", m.group(0), 7))
        spans = _resolve_overlaps(spans)
        for s in spans:
            if s.field == "mic" and s.value not in out.microphones:
                out.microphones.append(s.value)
            elif s.field == "mic_type" and s.value not in out.mic_types:
                out.mic_types.append(s.value)
            elif s.field == "setup" and s.value not in out.mic_setup:
                out.mic_setup.append(s.value)
            elif s.field == "recorder" and out.recorder is None:
                out.recorder = s.value
            elif s.field == "sample_rate" and out.sample_rate is None:
                out.sample_rate = round(float(s.value) * 1000)
            elif s.field == "bit_depth" and out.bit_depth is None:
                out.bit_depth = "32f" if "float" in s.value else s.value.split()[0]
            elif s.field in {"take", "scene", "slate"} and getattr(out, s.field) is None:
                setattr(out, s.field, s.value)
            elif s.field == "date" and out.date is None:
                out.date = _normalize_date(s.value)
        return spans

    def _cue_spans(self, norm: str, hard: list[_Span], out: SlateFields) -> list[_Span]:
        cues: list[tuple[int, int, str]] = []
        for name, rx in _CUES:
            for m in rx.finditer(norm):
                if any(h.start <= m.start() < h.end for h in hard):
                    continue
                cues.append((m.start(), m.end(), name))
        cues.sort()
        # drop cues nested in an earlier, longer cue match ("recording at" vs "recording")
        filtered: list[tuple[int, int, str]] = []
        for c in cues:
            if filtered and c[0] < filtered[-1][1]:
                if _CUE_PRIORITY[c[2]] > _CUE_PRIORITY[filtered[-1][2]] and c[0] == filtered[-1][0]:
                    filtered[-1] = c
                continue
            filtered.append(c)
        boundaries = sorted([h.start for h in hard] + [c[0] for c in filtered]
                            + [m.start() for m in _CLAUSE_END.finditer(norm)] + [len(norm)])
        spans: list[_Span] = []
        for start, vstart, name in filtered:
            if any(h.start <= vstart < h.end for h in hard):
                vend = vstart  # value would start inside a technical token
            else:
                vend = next((b for b in boundaries if b >= vstart), len(norm))
            vend = self._trim_value(norm, name, vstart, vend)
            value = _strip_glue(norm[vstart:vend])
            if name != "description":
                value = value[:80]
            spans.append(_Span(start, vend, name, value, _CUE_PRIORITY[name]))
            if not value:
                continue
            if name == "project" and out.project is None:
                out.project = _title(value)
            elif name == "location" and out.location is None:
                out.location = _title(value)
            elif name == "creator" and out.creator is None:
                out.creator = _title(" ".join(value.split()[:4]))
            elif name == "notes" and out.notes is None:
                out.notes = value
        return spans

    @staticmethod
    def _trim_value(norm: str, name: str, vstart: int, vend: int) -> int:
        """Shorten run-on values of unpunctuated transcripts.

        STT output often has no commas, so ``location tofino beach waves crashing`` would
        swallow the description. Locations end after their head noun (beach, forest, ...);
        projects are capped at four words when the slate has no clause punctuation.
        """
        seg = norm[vstart:vend]
        if name == "location":
            m = _LOCATION_NOUNS.search(seg)
            if m and len(seg[m.end():].split()) >= 2:
                return vstart + m.end()
        elif name in {"project", "creator"} and not _CLAUSE_END.search(norm):
            limit = 4 if name == "project" else 2
            words = list(re.finditer(r"\S+", seg))
            if len(words) > limit + 1:
                return vstart + words[limit - 1].end()
        return vend

    @staticmethod
    def _perspectives(norm: str) -> list[str]:
        return [tag for rx, tag in _PERSPECTIVES if rx.search(norm)]

    @staticmethod
    def _leftovers(norm: str, spans: list[_Span]) -> list[str]:
        chars = list(norm)
        for s in spans:
            for i in range(s.start, min(s.end, len(chars))):
                chars[i] = "|"
        text = "".join(chars)
        pieces = re.split(r"\|+|[,;!?]|\.(?!\d)| - ", text)
        return [p for p in (_strip_glue(x) for x in pieces) if p]


def _resolve_overlaps(spans: list[_Span]) -> list[_Span]:
    """Keep the highest-priority, then longest, span among overlapping ones."""
    spans = sorted(spans, key=lambda s: (-s.priority, -(s.end - s.start), s.start))
    kept: list[_Span] = []
    for s in spans:
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def _normalize_date(value: str) -> str:
    """Best-effort ISO formatting; returns the spoken form when the year is missing."""
    v = value.strip()
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})$", v)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"(" + "|".join(MONTHS)
                 + r")\s(\d{1,2})(?:st|nd|rd|th)?(?:,?\s(\d{2})\s?(\d{2})|,?\s(\d{4}))?$", v)
    if not m:
        m2 = re.match(r"(\d{1,2})(?:st|nd|rd|th)? of (" + "|".join(MONTHS) + r")(?:,?\s(\d{4}))?$", v)
        if m2:
            day, month, year = int(m2.group(1)), MONTHS.index(m2.group(2)) + 1, m2.group(3)
            return f"{year}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"
        return v
    month, day = MONTHS.index(m.group(1)) + 1, int(m.group(2))
    year = m.group(5) or (m.group(3) + m.group(4) if m.group(3) else None)
    return f"{year}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"


_DEFAULT = SlateParser()


def parse_slate(text: str) -> SlateFields:
    """Parse a slate transcript with the default :class:`SlateParser`."""
    return _DEFAULT.parse(text)
