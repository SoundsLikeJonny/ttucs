"""Leakage checks for the hand-written evaluation sets (AGENTS.md section 6).

Fails (exit 1) when:
1. a bank phrase of 3+ words (``resources/catid_phrases.yaml`` or any multi-word entry in
   ``resources/synth_banks.yaml``) appears verbatim in the dev descriptions or the realistic
   slates;
2. a dev description shares a 4-gram, or a whole substring, with a realistic slate.

Run it after editing the banks, the dev set or the realistic slates:

    python scripts/check_leakage.py            # exit 0 clean, 1 when leakage is found
    python scripts/check_leakage.py --dev FILE --test FILE
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ttucs.synth.generator import load_banks

ROOT = Path(__file__).resolve().parents[1]
NGRAM = 4
MIN_PHRASE_WORDS = 3


@dataclass(frozen=True)
class Violation:
    check: str      # "phrase" | "cross_4gram" | "substring"
    where: str      # file or split the offending text belongs to
    detail: str     # human-readable description


def norm_text(s: str) -> str:
    """Lowercase, keep [a-z0-9] runs, single-spaced."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def word_ngrams(s: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = norm_text(s).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def bank_phrases(banks: dict) -> list[tuple[str, str]]:
    """``(bank_path, normalised phrase)`` for every bank string of MIN_PHRASE_WORDS+ words,
    from both the per-CatID phrase bank and the generic vocabulary banks."""
    out: list[tuple[str, str]] = []

    def walk(node: object, where: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{where}.{k}" if where else str(k))
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, where)
        elif isinstance(node, str) and len(norm_text(node).split()) >= MIN_PHRASE_WORDS:
            out.append((where, norm_text(node)))

    walk(banks.get("catid_phrases") or {}, "catid_phrases")
    walk({k: v for k, v in banks.items() if k != "catid_phrases"}, "synth_banks")
    return out


def read_texts(path: Path) -> list[tuple[str, str]]:
    """``(group, text)`` pairs from a JSONL file with text/label/group fields."""
    out: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append((str(d.get("group") or d.get("label") or "?"), d["text"]))
    return out


def check_phrases(banks: dict, dev: Sequence[tuple[str, str]],
                  realistic: Sequence[tuple[str, str]]) -> list[Violation]:
    """Bank phrases of 3+ words appearing verbatim in either hand-written set."""
    out = []
    for where, phrase in bank_phrases(banks):
        for label, texts in (("dev_descriptions", dev), ("realistic_slates", realistic)):
            for group, text in texts:
                if phrase in norm_text(text):
                    out.append(Violation("phrase", f"{label}:{group}",
                                         f'bank phrase "{phrase}" ({where}) appears verbatim'))
    return out


def check_cross(dev: Sequence[tuple[str, str]],
                realistic: Sequence[tuple[str, str]]) -> list[Violation]:
    """4-gram or substring overlap between dev descriptions and realistic slates."""
    out = []
    for dgroup, dtext in dev:
        dn = norm_text(dtext)
        dgrams = word_ngrams(dn)
        for rgroup, rtext in realistic:
            rn = norm_text(rtext)
            shared = dgrams & word_ngrams(rn)
            if shared:
                out.append(Violation("cross_4gram", f"dev:{dgroup} realistic:{rgroup}",
                                     f'shares 4-gram "{" ".join(sorted(shared)[0])}"'))
            elif dn in rn or rn in dn:
                out.append(Violation("substring", f"dev:{dgroup} realistic:{rgroup}",
                                     f'one text contains the other ("{dn[:60]}")'))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, default=ROOT / "examples" / "dev_descriptions.jsonl")
    ap.add_argument("--test", type=Path, default=ROOT / "examples" / "realistic_slates.jsonl")
    args = ap.parse_args(argv)

    banks = load_banks()
    dev, realistic = read_texts(args.dev), read_texts(args.test)
    violations = check_phrases(banks, dev, realistic) + check_cross(dev, realistic)
    if violations:
        print(f"LEAKAGE CHECK FAILED: {len(violations)} violation(s)")
        for v in violations:
            print(f"  [{v.check}] {v.where}: {v.detail}")
        return 1
    print(f"leakage check clean ({len(dev)} dev texts, {len(realistic)} realistic slates, "
          f"{len(bank_phrases(banks))} bank phrases of {MIN_PHRASE_WORDS}+ words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
