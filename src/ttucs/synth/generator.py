"""LLM-free synthetic data generation for UCS classification.

Two generators:

* :class:`DescriptionGenerator` - short sound descriptions for any CatID, built from
  hand-written vocabulary banks (``resources/synth_banks.yaml``), the UCS Explanation and
  the UCS Synonyms of the CatID (weighted by how *distinctive* each synonym is across the
  whole list, so "Carbon"/"CO2", shared by every AIR subcategory, rarely appears).
* :class:`SlateGenerator` - wraps a description into a realistic speech-to-text slate
  (project, location, gear, sample rate, take, fillers, spoken numbers, no punctuation)
  and returns the ground-truth fields, used to evaluate the slate parser and to make the
  classifier robust to parser leftovers.

An optional ``llm_hook`` callable can paraphrase generated descriptions (e.g. with a local
LLM); it is not used by default so generation is fully deterministic and offline.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources

import yaml

from ttucs.taxonomy import UCSEntry, UCSTaxonomy

GENERIC_SUBCATS = {"BREAK", "CRASH & DEBRIS", "FRICTION", "HANDLE", "IMPACT", "MOVEMENT",
                   "TONAL", "CREAK", "RIP", "FLAP", "DUST", "SUCTION", "BUBBLES"}


def load_banks() -> dict:
    with resources.files("ttucs.resources").joinpath("synth_banks.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Example:
    """One labelled training/evaluation example."""

    text: str
    label: str
    source: str
    group: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "label": self.label, "source": self.source,
                "group": self.group, "meta": self.meta}


class DescriptionGenerator:
    """Template + vocabulary-bank generator of sound descriptions for a CatID."""

    def __init__(self, taxonomy: UCSTaxonomy, banks: dict | None = None, seed: int = 0,
                 max_synonym_df: int = 25,
                 llm_hook: Callable[[str, UCSEntry], str] | None = None) -> None:
        self.tax = taxonomy
        self.banks = banks or load_banks()
        self.rng = random.Random(seed)
        self.llm_hook = llm_hook
        self.mods = [m for group in self.banks["modifiers"].values() for m in group]
        self.contexts = self.banks["contexts"]
        df: Counter[str] = Counter()
        for e in taxonomy.entries:
            df.update({s.lower() for s in e.synonyms})
        self._syn_weights: dict[str, tuple[list[str], list[float]]] = {}
        for e in taxonomy.entries:
            syns = sorted({s.lower() for s in e.synonyms if df[s.lower()] <= max_synonym_df})
            weights = [1.0 / math.sqrt(df[s]) for s in syns]
            self._syn_weights[e.cat_id] = (syns, weights)

    # ------------------------------------------------------------------ helpers
    def _syn(self, e: UCSEntry) -> str | None:
        syns, w = self._syn_weights[e.cat_id]
        return self.rng.choices(syns, weights=w, k=1)[0] if syns else None

    def _sub_words(self, e: UCSEntry) -> str | None:
        if e.subcategory in {"MISC", "GENERAL"}:
            return None
        return e.subcategory.lower().replace(" & ", " and ")

    def _expl_fragment(self, e: UCSEntry) -> str | None:
        parts = [p.strip() for p in re.split(r"[.;,]|\bor\b|\bfrom\b", e.explanation) if p.strip()]
        parts = [p for p in parts if 2 <= len(p.split()) <= 9 and "category" not in p.lower()]
        if not parts:
            return None
        frag = self.rng.choice(parts).lower()
        frag = re.sub(r"^(sounds? of|the sound of|anything|things|all|any|includes?|such as|e\.g\.?|like|"
                      r"for example|also|see)\s+", "", frag)
        return frag.strip(" '\"()")

    def _pick(self, seq: list[str]) -> str:
        return self.rng.choice(seq)

    # ------------------------------------------------------------------ public
    def generate_one(self, e: UCSEntry) -> str:
        cat = self.banks["categories"].get(e.category, {"objects": [e.category.lower()], "actions": []})
        objects, actions = cat.get("objects") or [e.category.lower()], cat.get("actions") or [""]
        sub_actions = self.banks["subcategory_actions"].get(e.subcategory)
        generic = e.subcategory in GENERIC_SUBCATS
        r = self.rng.random

        syn, sub = self._syn(e), self._sub_words(e)
        mod = self._pick(self.mods) if r() < 0.55 else ""
        ctx = self._pick(self.contexts) if r() < 0.15 else ""

        if generic or e.subcategory in {"MISC", "GENERAL"}:
            head = self._pick(objects)
            act = self._pick(sub_actions) if (sub_actions and r() < 0.8) else self._pick(actions)
            templates = [
                f"{mod} {head} {act}", f"{head} {act} {mod}", f"{mod} {head} {act} {ctx}",
                f"{head} {syn or act}", f"{syn or head} {head} {act}", f"{act} {head} {mod}",
            ]
        else:
            noun_choices = [x for x in (sub, syn, syn, self._syn(e)) if x]
            head = self._pick(noun_choices) if noun_choices else self._pick(objects)
            act = self._pick(actions) if r() < 0.7 else (self._syn(e) or "")
            other = self._syn(e) or ""
            templates = [
                f"{mod} {head} {act}", f"{head} {act}", f"{head} {other} {act}",
                f"{mod} {head} {other}", f"{sub or other} {head} {act} {ctx}",
                f"{head} {act} {mod} {ctx}", f"{other} and {head} {act}",
            ]
            if r() < 0.25:
                templates.append(f"{self._pick(objects)} {head} {act}")
        frag = self._expl_fragment(e)
        if frag and r() < 0.12:
            text = f"{mod} {frag}" if r() < 0.5 else frag
        else:
            text = self._pick(templates)
        text = re.sub(r"\s+", " ", text).strip()
        # light lexical noise: occasional word dropout to avoid template memorisation
        words = text.split()
        if len(words) > 3 and r() < 0.15:
            words.pop(self.rng.randrange(len(words)))
        text = " ".join(words)
        if self.llm_hook is not None:
            text = self.llm_hook(text, e)
        return text

    def generate(self, entry: UCSEntry, n: int, max_tries_factor: int = 6) -> list[str]:
        """Return up to ``n`` *unique* descriptions for ``entry``."""
        out: dict[str, None] = {}
        tries = 0
        while len(out) < n and tries < n * max_tries_factor:
            out.setdefault(self.generate_one(entry), None)
            tries += 1
        return list(out)


# ============================================================================ slates
_MICS = [
    ("mkh four sixteen", "Sennheiser MKH 416"), ("sennheiser mkh four sixteen", "Sennheiser MKH 416"),
    ("mkh eighty forty", "Sennheiser MKH 8040"), ("mkh eighty fifty", "Sennheiser MKH 8050"),
    ("mkh eighty sixty", "Sennheiser MKH 8060"), ("mkh thirty", "Sennheiser MKH 30"),
    ("schoeps cmc six", "Schoeps CMC6"), ("schoeps mk four", "Schoeps MK4"),
    ("cmit five u", "Schoeps CMIT5U"), ("rode ntg three", "Rode NTG3"), ("rode nt four", "Rode NT4"),
    ("dpa forty sixty", "DPA 4060"), ("dpa four zero six zero", "DPA 4060"),
    ("sanken cs three e", "Sanken CS3E"), ("neumann kmr eighty one", "Neumann KMR81"),
    ("shure sm fifty seven", "Shure SM57"), ("audio technica bp4029", "Audio-Technica BP4029"),
    ("lom usi pro", "LOM Usi Pro"), ("sennheiser ambeo", "Sennheiser Ambeo VR"),
]
_MIC_TYPES = [("a hydrophone", "hydrophone"), ("contact mics", "contact mic"),
              ("a lav", "lavalier"), ("a shotgun", "shotgun"), ("binaural mics", "binaural")]
_RECORDERS = [
    ("mix pre six", "Sound Devices MixPre-6"), ("mixpre three", "Sound Devices MixPre-3"),
    ("sound devices seven forty four", "Sound Devices 744"),
    ("sound devices eight thirty three", "Sound Devices 833"), ("zoom f six", "Zoom F6"),
    ("zoom f three", "Zoom F3"), ("zoom h five", "Zoom H5"), ("zoom h six", "Zoom H6"),
    ("tascam dr forty x", "Tascam DR40X"), ("sound devices mix pre ten", "Sound Devices MixPre-10"),
]
_SETUPS = [("ms", "MS"), ("mid side", "MS"), ("x y", "XY"), ("ortf", "ORTF"),
           ("spaced pair", "Spaced Pair (AB)"), ("in mono", "Mono"), ("in stereo", "Stereo"),
           ("ambisonic", "Ambisonic")]
_RATES = [("ninety six k", 96000), ("96k", 96000), ("forty eight k", 48000), ("48 khz", 48000),
          ("one ninety two k", 192000), ("192k", 192000), ("forty four one", 44100),
          ("ninety six kilohertz", 96000)]
_BITS = [("twenty four bit", "24"), ("24 bit", "24"), ("thirty two bit float", "32f"),
         ("sixteen bit", "16"), ("32 bit float", "32f")]
_PROJ_A = ["thick as thieves", "nightfall", "dragon game", "iron harbor", "lost signal", "red tide",
           "glass city", "hollow pines", "salt and smoke", "kestrel", "deep station", "the long road",
           "paper moons", "afterglow", "black river", "quiet storm", "neon drift", "stone giants"]
_LOC = ["squamish forest", "the vancouver seawall", "tofino beach", "a parking garage downtown",
        "my kitchen", "studio b", "the foley stage", "grandma's cabin", "lynn canyon",
        "an abandoned warehouse", "the backyard", "stanley park", "a farm near abbotsford",
        "the train yard", "whistler village", "a church in gastown", "the shop", "deep cove"]
_NAMES = ["jon evans", "sam lee", "maria garcia", "alex chen", "priya patel", "tom baker",
          "nina kowalski", "dave smith"]
_FILLERS = ["um", "uh", "okay", "so", "alright", "uh so", "um okay"]
_DATES = [("november sixth twenty twenty six", "2026-11-06"), ("march third", "03-03"),
          ("2026-06-14", "2026-06-14"), ("july twenty first twenty twenty five", "2025-07-21")]
_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve"]


class SlateGenerator:
    """Wrap descriptions into noisy, spoken-style slate transcripts with ground truth."""

    def __init__(self, seed: int = 0, p_field: float = 0.55, p_fillers: float = 0.35,
                 p_no_punct: float = 0.5) -> None:
        self.rng = random.Random(seed)
        self.p_field, self.p_fillers, self.p_no_punct = p_field, p_fillers, p_no_punct

    def wrap(self, description: str) -> tuple[str, dict]:
        r, pick = self.rng.random, self.rng.choice
        truth: dict = {"description": description}
        segs: list[str] = []
        if r() < self.p_field:
            p = pick(_PROJ_A)
            segs.append(pick([f"project {p}", f"for the project {p}", f"project name {p}"]))
            truth["project"] = p
        if r() < self.p_field:
            loc = pick(_LOC)
            segs.append(pick([f"location {loc}", f"we're at {loc}", f"recorded at {loc}"]))
            truth["location"] = loc
        if r() < 0.2:
            n = pick(_NAMES)
            segs.append(pick([f"recordist {n}", f"recorded by {n}", f"my name is {n}"]))
            truth["creator"] = n
        if r() < 0.15:
            spoken, iso = pick(_DATES)
            segs.append(spoken)
            truth["date"] = iso
        gear: list[str] = []
        if r() < self.p_field:
            spoken, canon = pick(_MICS)
            truth["microphones"] = [canon]
            if r() < 0.3:
                s2, c2 = pick(_SETUPS)
                gear.append(f"{s2} {pick(['with', 'on'])} {spoken}" if r() < 0.5 else f"{spoken} {s2}")
                truth["mic_setup"] = [c2]
            else:
                gear.append(spoken)
        elif r() < 0.15:
            spoken, canon = pick(_MIC_TYPES)
            gear.append(spoken)
            truth["mic_types"] = [canon]
        if r() < self.p_field:
            spoken, canon = pick(_RECORDERS)
            gear.append(("into " if gear and r() < 0.5 else "") + spoken)
            truth["recorder"] = canon
        if gear:
            segs.append(" ".join(gear))
        if r() < 0.45:
            spoken, sr = pick(_RATES)
            tech = [spoken]
            truth["sample_rate"] = sr
            if r() < 0.7:
                b, bv = pick(_BITS)
                tech.append(b)
                truth["bit_depth"] = bv
            segs.append(" ".join(tech))
        if r() < 0.5:
            n = self.rng.randint(1, 12)
            segs.append(f"take {_NUM[n] if r() < 0.7 else n}")
            truth["take"] = str(n)
        desc_seg = pick([description, f"recording {description}", f"this is {description}",
                         f"recording a {description}", f"sound of {description}"]) \
            if r() < 0.6 else description
        pos = self.rng.randint(0, len(segs))
        segs.insert(pos, desc_seg)
        if pos > 0 and r() < 0.5:  # metadata-first order is the most common slate habit
            segs.insert(0, segs.pop(pos))
        if r() < self.p_fillers:
            segs.insert(0, pick(_FILLERS))
        sep = " " if r() < self.p_no_punct else ", "
        text = sep.join(segs)
        if r() < 0.25:
            text = text.replace(" ", " uh ", 1) if r() < 0.5 else text
        return text, truth
