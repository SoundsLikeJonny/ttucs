"""Normalisation of speech-to-text (STT) output.

STT transcripts of field-recording slates are lower-case, lightly punctuated and full of
artefacts: filler words ("um", "uh"), spoken numbers ("ninety six k", "four sixteen"),
spelled-out acronyms ("m k h") and homophones ("mike" for *mic*, "road" for *Rode*).
:func:`normalize_transcript` turns that into a canonical string the slate parser can
pattern-match reliably.
"""

from __future__ import annotations

import re

UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
}
TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
NUMBER_WORDS = set(UNITS) | set(TEENS) | set(TENS) | set(SCALES)

FILLERS = re.compile(
    r"\b(?:um+|uh+|uhm+|erm+|er|ah+|hmm+|mm+|you know|i mean|kind of like|sort of like)\b[,]?",
    re.IGNORECASE,
)
LEADING_FILLERS = re.compile(r"^(?:\s*(?:okay|ok|alright|all right|so|right|and|yeah)\b[,.]?\s*)+",
                             re.IGNORECASE)

# Homophones / spelling variants commonly produced by STT engines for audio gear terms.
REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bforty[- ]four (?:point )?one\b", "44.1"), (r"\beighty[- ]eight (?:point )?two\b", "88.2"),
    (r"\bmikes\b", "mics"), (r"\bmike\b", "mic"), (r"\bmic'?d\b", "miked"),
    (r"\bmicrophones\b", "mics"), (r"\bmicrophone\b", "mic"),
    (r"\bsen+ h?eis+er\b|\bsen+heis+er\b|\bsen+heiser\b|\bsenhizer\b|\bsenheiser\b", "sennheiser"),
    (r"\bsho?e?ps\b(?=\s*(?:cmc|mk|cmit|ccm|mic|\d))|\bschoeps?\b|\bshoeps\b", "schoeps"),
    (r"\broad\b(?=\s*(?:ntg|nt|video))", "rode"),
    (r"\bd\s?p\s?a\b", "dpa"), (r"\bm\s?k\s?h\b", "mkh"), (r"\bn\s?t\s?g\b", "ntg"),
    (r"\bc\s?m\s?i\s?t\b", "cmit"), (r"\bc\s?m\s?c\b", "cmc"), (r"\bk\s?m\s?r\b", "kmr"),
    (r"\bo\s?r\s?t\s?f\b", "ortf"), (r"\bmid[- ]side\b", "ms"), (r"\bm\s?/\s?s\b", "ms"),
    (r"\bx\s?/\s?y\b|\bx y\b", "xy"), (r"\ba\s?/\s?b\b", "ab"),
    (r"\ba b\b(?= (?:pair|stereo|setup|rig|technique))", "ab"),
    (r"\b(?:dual|double)\s+a\s?/?\s?b\b", "dual ab"),
    (r"\bmix\s?pre\b|\bmixed pre\b|\bmix pray\b", "mixpre"),
    (r"\btask\s?cam\b", "tascam"), (r"\bsound device\b", "sound devices"),
    (r"\bkilo ?hertz\b|\bk ?hz\b|\bkay\b", "k"), (r"\bbits\b", "bit"),
    (r"\bthirty two bit float\b", "32 bit float"),
    (r"\bclose ?up\b", "close up"), (r"\bon ?board\b", "onboard"),
    (r"\buh-huh\b", ""), (r"\bh ?(?:2|two) ?a\b", "h2a"),
]

_SINGLE_LETTERS = re.compile(r"\b((?:[b-hj-z]\s){1,}[b-hj-z])\b")


def words_to_numbers(text: str) -> str:
    """Convert spoken numbers to digits, keeping the "digit group" reading used on slates.

    >>> words_to_numbers("ninety six k twenty four bit take three")
    '96 k 24 bit take 3'
    >>> words_to_numbers("mkh four sixteen and an eighty forty")
    'mkh 416 and an 8040'
    >>> words_to_numbers("two thousand twenty six")
    '2026'
    """
    tokens = re.findall(r"[\w'.-]+|[^\w\s]", text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i].lower()
        if tok in ORDINALS and _is_date_context(tokens, i):
            if out and out[-1] in {"20", "30"} and ORDINALS[tok] < 10:
                out[-1] = str(int(out[-1]) + ORDINALS[tok])  # "twenty sixth" -> 26
            else:
                out.append(str(ORDINALS[tok]))
            i += 1
            continue
        if tok not in NUMBER_WORDS or (tok == "oh" and not _is_numeric_context(tokens, i)) or \
                (tok == "one" and not _is_numeric_context(tokens, i)):
            out.append(tokens[i])
            i += 1
            continue
        j = i
        chunks: list[str] = []
        total = 0          # value accumulated across a 'thousand'
        current = None     # value of the chunk being built
        last = None        # kind of the last token: unit|teen|tens|scale
        while j < len(tokens):
            t = tokens[j].lower()
            if t == "and" and last == "scale" and j + 1 < len(tokens) and \
                    tokens[j + 1].lower() in NUMBER_WORDS:
                j += 1
                continue
            if t not in NUMBER_WORDS:
                break
            if t in SCALES:
                base = current if current is not None else 1
                if t == "thousand":
                    total, current = (total + base) * 1000, 0
                else:
                    current = base * 100
                last = "scale"
            elif t in TENS:
                if current is None or (last != "scale"):
                    if current is not None:
                        chunks.append(str(total + current))
                        total = 0
                    current = TENS[t]
                else:
                    current += TENS[t]
                last = "tens"
            else:
                val = UNITS.get(t, TEENS.get(t))
                kind = "unit" if t in UNITS else "teen"
                if current is not None and (
                    (last == "tens" and kind == "unit" and current % 10 == 0)
                    or last == "scale"
                ):
                    current += val
                else:
                    if current is not None:
                        chunks.append(str(total + current))
                        total = 0
                    current = val
                last = kind
            j += 1
        if current is not None:
            chunks.append(str(total + current))
        prev = tokens[i - 1].lower() if i > 0 else ""
        nxt = tokens[j].lower() if j < len(tokens) else ""
        groups = [chunks]
        # "take three ninety six k" / "mix pre six ninety six k": split id from value
        if len(chunks) > 1 and prev in _ID_CUES:
            groups = [chunks[:1], chunks[1:]]
        if len(groups[-1]) > 1 and nxt in {"k", "bit", "khz"}:
            groups = [*groups[:-1], groups[-1][:-1], groups[-1][-1:]]
        out.append(" ".join("".join(g) for g in groups))
        i = j
    text = " ".join(out)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # "44 1 k" / "44 point 1 k" -> 44.1 k
    text = re.sub(r"\b(\d{2,3})\s(?:point\s|\.\s?)?(\d)\s?k\b", r"\1.\2 k", text)
    return text


_ID_CUES = {"take", "scene", "slate", "track", "number", "shot", "roll", "clip", "mixpre",
            "f", "h", "dr", "pre"}

MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december")


def _is_date_context(tokens: list[str], i: int) -> bool:
    """Ordinals ('sixth') are only converted inside dates ('november sixth', 'sixth of may')."""
    prev = tokens[i - 1].lower() if i > 0 else ""
    nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
    return prev in MONTHS or nxt == "of" or (prev in TENS and i > 1
                                              and tokens[i - 2].lower() in MONTHS)


def _is_numeric_context(tokens: list[str], i: int) -> bool:
    """'one'/'oh' are only numbers next to other numbers or after count-like cues."""
    prev = tokens[i - 1].lower() if i > 0 else ""
    nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
    if prev in NUMBER_WORDS or nxt in NUMBER_WORDS or prev.isdigit() or nxt.isdigit():
        return True
    return prev in {"take", "scene", "slate", "track", "number", "no", "day", "shot",
                    "roll", "clip", "channel", "version", "mixpre", "f", "h"}


def join_spelled_letters(text: str) -> str:
    """'m k h' -> 'mkh'; 'c u' -> 'cu' (ignores 'a' and 'i' which are words)."""
    return _SINGLE_LETTERS.sub(lambda m: m.group(1).replace(" ", ""), text)


def normalize_transcript(text: str) -> str:
    """Canonicalise an STT transcript for parsing (lower-case, digits, no fillers)."""
    text = text.replace("\u2019", "'").replace("\u2014", ", ").replace("\u2013", "-")
    text = text.lower()
    text = FILLERS.sub(" ", text)
    text = LEADING_FILLERS.sub("", text)
    text = join_spelled_letters(text)
    for pat, rep in REPLACEMENTS:
        text = re.sub(pat, rep, text)
    text = words_to_numbers(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])\1+", r"\1", text)
    return text.strip(" ,.;")
