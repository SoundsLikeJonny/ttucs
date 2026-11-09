"""Lexicon of field-recording gear (microphones, recorders, stereo techniques).

Patterns run on *normalised* transcripts (see :mod:`ttucs.slate.normalize`): lower-case,
spoken numbers converted to digits, spelled letters joined. Each pattern maps to a
formatter producing a canonical display name. Extend these lists to teach the parser
new gear; no retraining is needed because slate fields are not model inputs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GearPattern:
    kind: str  # "mic" | "recorder" | "setup" | "mic_type"
    regex: re.Pattern[str]
    fmt: Callable[[re.Match[str]], str]


def _p(kind: str, pattern: str, fmt: Callable[[re.Match[str]], str] | str) -> GearPattern:
    f = fmt if callable(fmt) else (lambda m, s=fmt: s)
    return GearPattern(kind, re.compile(pattern), f)


def _up(s: str | None) -> str:
    return re.sub(r"\s+", "", s or "").upper()


MICS: list[GearPattern] = [
    _p("mic", r"\b(?:sennheiser\s+)?mkh\s?(\d{2,4})\b", lambda m: f"Sennheiser MKH {m.group(1)}"),
    # STT often drops the "h" of MKH ("m k eighty forty" -> "mk 8040"). 3-4 digit capsule
    # numbers are unambiguous: Schoeps capsules are 1-2 digits (MK 4/8/21/22/41).
    _p("mic", r"\b(?:sennheiser\s+)?mk\s?(\d{3,4})\b", lambda m: f"Sennheiser MKH {m.group(1)}"),
    _p("mic", r"\bsennheiser\s+mk\s?(30|50)\b", lambda m: f"Sennheiser MKH {m.group(1)}"),
    _p("mic", r"\bsennheiser\s+(me|mke)\s?(\d{1,3})\b",
       lambda m: f"Sennheiser {m.group(1).upper()}{m.group(2)}"),
    _p("mic", r"\b(?:sennheiser\s+)?ambeo(?:\s+vr)?\b", "Sennheiser Ambeo VR"),
    _p("mic", r"\bschoeps\s+(cmc\s?\d+\w*|mk\s?\d+\w*|cmit\s?\d*\w*|ccm\s?\d+\w*)",
       lambda m: f"Schoeps {_up(m.group(1))}"),
    _p("mic", r"\b(cmit\s?5u?|cmc\s?6|mk\s?41|mk\s?4|mk\s?8|mk\s?21|mk\s?22)\b",
       lambda m: f"Schoeps {_up(m.group(1))}"),
    _p("mic", r"\bschoeps\b", "Schoeps"),
    _p("mic", r"\b(?:rode\s+)?(ntg\s?\d|nt\s?sf\s?1|ntsf1|nt\s?4|nt\s?5|nt\s?1a?)\b",
       lambda m: f"Rode {_up(m.group(1))}"),
    _p("mic", r"\bdpa\s?(\d{4}\w?)\b", lambda m: f"DPA {m.group(1)}"),
    _p("mic", r"\b(?:neumann\s+)?(kmr\s?8[12]i?|u\s?87\w*|km\s?18[34]|rsm\s?191|tlm\s?\d{3}|km\s?1\d\d)\b",
       lambda m: f"Neumann {_up(m.group(1))}"),
    _p("mic", r"\b(?:shure\s+)?(sm\s?5[78]|sm\s?7b?|vp\s?88|beta\s?\d{2})\b",
       lambda m: f"Shure {_up(m.group(1))}"),
    _p("mic", r"\b(?:audio[- ]technica\s+)(?:at\s?)?(\w*\d{3,4}\w*)\b",
       lambda m: f"Audio-Technica {_up(m.group(1))}"),
    _p("mic", r"\b(bp\s?40\d{2}|at\s?40\d{2}|at\s?875r?|at\s?8022)\b",
       lambda m: f"Audio-Technica {_up(m.group(1))}"),
    _p("mic", r"\bsanken\s+(cs\s?3e?|co\s?100k|cub\s?01|cos\s?11d?)\b",
       lambda m: f"Sanken {_up(m.group(1))}"),
    _p("mic", r"\b(?:lom\s+)?(usi\s?pro|usi|ushi|geofon|elektrosluch|mikroUsi)\b",
       lambda m: f"LOM {m.group(1).title()}"),
    _p("mic", r"\b(?:aquarian\s+)?h2a\b", "Aquarian H2a"),
    _p("mic", r"\b(?:jez riley french|jrf)(?:\s+(c\s?series|d\s?series|hydrophone|contact))?\b",
       lambda m: "JrF " + (m.group(1) or "").title()),
    _p("mic", r"\b(?:sony\s+)?ecm\s?(\w+)\b", lambda m: f"Sony ECM-{_up(m.group(1))}"),
    _p("mic", r"\b(?:zoom\s+)?(?:ssh\s?6|msh\s?6|xyh\s?6)\b", lambda m: f"Zoom {_up(m.group(0))}"),
]

MIC_TYPES: list[GearPattern] = [
    _p("mic_type", r"\bhydrophones?\b", "hydrophone"),
    _p("mic_type", r"\bcontact mics?\b|\bpiezos?\b", "contact mic"),
    _p("mic_type", r"\bgeophones?\b", "geophone"),
    _p("mic_type", r"\blav(?:alier)?s?\b", "lavalier"),
    _p("mic_type", r"\bshotguns?(?: mics?)?\b", "shotgun"),
    _p("mic_type", r"\bbinaural(?: mics?| head)?\b", "binaural"),
    _p("mic_type", r"\bcoil pick ?ups?\b|\btelephone pick ?up\b", "coil pickup"),
]

RECORDERS: list[GearPattern] = [
    _p("recorder", r"\b(?:sound devices\s+)?mixpre\s?(3|6|10)(?:\s?(ii|2)\b)?",
       lambda m: f"Sound Devices MixPre-{m.group(1)}" + (" II" if m.group(2) else "")),
    _p("recorder", r"\bsound devices\s+(\d{3}(?:\s?t\b)?|scorpio|a20\w*)",
       lambda m: f"Sound Devices {_up(m.group(1))}"),
    _p("recorder", r"\b(?:sound devices\s+)?mixpre\b", "Sound Devices MixPre"),
    _p("recorder", r"\bsound devices\b", "Sound Devices"),
    _p("recorder", r"\b(7[0-9]{2}t|788t|744t|722|702t?|633|664|833|888)\b(?=.*\brecorder\b)|"
                   r"\b(788t|744t|702t)\b",
       lambda m: f"Sound Devices {(m.group(1) or m.group(2)).upper()}"),
    _p("recorder", r"\bscorpio\b", "Sound Devices Scorpio"),
    _p("recorder", r"\bzoom\s+(f\s?\d{1,2}|h\s?\d(?:n|\s?pro|\s?essential)?|m\s?\d|uac\w*)\b",
       lambda m: f"Zoom {_up(m.group(1))}"),
    _p("recorder", r"\b(f[368]|h[456]n?)\b", lambda m: f"Zoom {m.group(1).upper()}"),
    _p("recorder", r"\btascam\s+(dr\s?-?\s?\d+\w*(?:\s?x\b)?|hd\s?p2|portacapture\s?x?\d*|x\d+|frontier)\b",
       lambda m: f"Tascam {_up(m.group(1))}"),
    _p("recorder", r"\btascam\b", "Tascam"),
    _p("recorder", r"\bnagra\s?(\w+)?\b", lambda m: "Nagra " + (m.group(1) or "").upper()),
    _p("recorder", r"\bsony\s+pcm\s?-?\s?(\w+)\b", lambda m: f"Sony PCM-{_up(m.group(1))}"),
    _p("recorder", r"\b(?:roland\s+)?r\s?0?([57]|26|88)\b(?=.*\b(?:recorder|roland)\b)",
       lambda m: f"Roland R-{m.group(1)}"),
    _p("recorder", r"\bzaxcom\s+(\w+)\b", lambda m: f"Zaxcom {m.group(1).title()}"),
    _p("recorder", r"\bcantar\s?(\w+)?\b", lambda m: "Aaton Cantar " + (m.group(1) or "").upper()),
    _p("recorder", r"\b(iphone|ipad|phone)\b", lambda m: m.group(1)),
]

SETUPS: list[GearPattern] = [
    _p("setup", r"\bdouble ms\b", "Double MS"),
    _p("setup", r"\bms\b(?: (?:pair|rig|stereo|setup))?", "MS"),
    _p("setup", r"\bxy\b(?: (?:pair|rig|setup))?", "XY"),
    _p("setup", r"\bortf\b", "ORTF"),
    _p("setup", r"\bnos\b", "NOS"),
    _p("setup", r"\bblumlein\b", "Blumlein"),
    _p("setup", r"\bdecca tree\b", "Decca Tree"),
    _p("setup", r"\bspaced (?:pair|omnis?)\b|\b(?:dual|double)\s?ab\b|\bab pair\b|"
                r"\bab\b(?= (?:pair|stereo|omnis?))",
       "Spaced Pair (AB)"),
    _p("setup", r"\bquad\b|\bquadraphonic\b", "Quad"),
    _p("setup", r"\bambisonics?\b|\bfirst order ambi\w*\b|\bfoa\b", "Ambisonic"),
    _p("setup", r"\bdual mono\b", "Dual Mono"),
    _p("setup", r"\b5\.?1\b(?= (?:surround|array))|\bsurround\b", "Surround"),
    _p("setup", r"\bin stereo\b|\bstereo\b(?! (?:system|speaker))", "Stereo"),
    _p("setup", r"\bin mono\b|\bmono\b", "Mono"),
]

ALL_GEAR: list[GearPattern] = MICS + MIC_TYPES + RECORDERS + SETUPS
