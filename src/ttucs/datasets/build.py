"""Assemble balanced train/val/test splits from synthetic data and UCS label documents."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from ttucs.slate.parser import parse_slate
from ttucs.synth.generator import DescriptionGenerator, Example, SlateGenerator
from ttucs.taxonomy import UCSTaxonomy

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------ io
def write_jsonl(examples: list[Example], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(Example(text=d["text"], label=d["label"], source=d.get("source", ""),
                                   group=d.get("group", ""), meta=d.get("meta", {})))
    return out


# --------------------------------------------------------------------------- splits
def split_bucket(group: str, seed: int) -> float:
    """Deterministic uniform number in [0, 1) for a group key (stable across runs/OSes)."""
    h = hashlib.sha1(f"{seed}:{group}".encode(), usedforsecurity=False).hexdigest()
    return int(h[:12], 16) / float(16**12)


def group_split(examples: list[Example], val: float, test: float, seed: int
                ) -> tuple[list[Example], list[Example], list[Example]]:
    """Split by ``group`` so near-duplicates (take 01/02/03) never straddle splits."""
    tr, va, te = [], [], []
    for ex in examples:
        b = split_bucket(ex.group or ex.text, seed)
        (te if b < test else va if b < test + val else tr).append(ex)
    return tr, va, te


# ------------------------------------------------------------------------ synthetic
def generate_synthetic(taxonomy: UCSTaxonomy, counts: dict[str, int], seed: int,
                       max_synonym_df: int = 25, phrase_prob: float = 0.5,
                       multi_clause_prob: float = 0.0) -> list[Example]:
    gen = DescriptionGenerator(taxonomy, seed=seed, max_synonym_df=max_synonym_df,
                               phrase_prob=phrase_prob, multi_clause_prob=multi_clause_prob)
    out: list[Example] = []
    for cat_id, n in counts.items():
        entry = taxonomy[cat_id]
        for i, text in enumerate(gen.generate(entry, n)):
            # groups of 4 consecutive samples share a split bucket (limits template leakage)
            out.append(Example(text=text, label=cat_id, source="synthetic",
                               group=f"synth:{cat_id}:{i // 4}"))
    return out


def label_documents(taxonomy: UCSTaxonomy, cat_ids: list[str]) -> list[Example]:
    """UCS explanation / synonym documents as extra training text (cold start for rare classes)."""
    out = []
    for cid in cat_ids:
        e = taxonomy[cid]
        docs = [e.explanation, f"{e.category.lower()} {e.subcategory.lower()}"]
        syn = list(e.synonyms)
        for i in range(0, min(len(syn), 24), 8):
            docs.append(" ".join(s.lower() for s in syn[i:i + 8]))
        out += [Example(text=d, label=cid, source="ucs_doc", group=f"doc:{cid}") for d in docs if d]
    return out


def slate_augment(examples: list[Example], fraction: float, seed: int) -> list[Example]:
    """Replace a fraction of texts by parser output on a synthetic slate around them."""
    rng = random.Random(seed)
    sg = SlateGenerator(seed=seed)
    out = []
    for ex in examples:
        if ex.source != "ucs_doc" and rng.random() < fraction:
            slate, _ = sg.wrap(ex.text)
            parsed = parse_slate(slate).description
            out.append(Example(text=parsed, label=ex.label, source=ex.source, group=ex.group,
                               meta={**ex.meta, "slate": slate, "augmented": True}))
        else:
            out.append(ex)
    return out


def make_slate_testset(examples: list[Example], seed: int, n: int = 2000) -> list[Example]:
    rng = random.Random(seed)
    sg = SlateGenerator(seed=seed + 1)
    picked = rng.sample(examples, min(n, len(examples)))
    out = []
    for ex in picked:
        slate, truth = sg.wrap(ex.text)
        out.append(Example(text=slate, label=ex.label, source=f"slate:{ex.source}", group=ex.group,
                           meta={"truth": truth}))
    return out


# ---------------------------------------------------------------------------- build
def build_datasets(cfg: dict, taxonomy: UCSTaxonomy) -> dict[str, list[Example]]:
    """Build all splits described by ``cfg['data']``. Returns a dict of split -> examples."""
    seed = int(cfg["seed"])
    dcfg = cfg["data"]
    exclude = set(dcfg.get("exclude_categories") or [])
    labels = [e.cat_id for e in taxonomy.trainable(exclude)]
    vf, tf = dcfg["split"]["val"], dcfg["split"]["test"]
    splits: dict[str, list[Example]] = defaultdict(list)

    # ---- synthetic descriptions (balanced: the same budget for every CatID)
    scfg = dcfg["synthetic"]
    if scfg.get("enabled", True):
        counts = {cid: max(scfg["min_per_class"], scfg["target_per_class"]) for cid in labels}
        synth = generate_synthetic(taxonomy, counts, seed, scfg.get("max_synonym_df", 25),
                                   scfg.get("phrase_prob", 0.5),
                                   scfg.get("multi_clause_prob", 0.0))
        if scfg.get("cache"):
            write_jsonl(synth, scfg["cache"])
        str_, sva, ste = group_split(synth, vf, tf, seed)
        splits["train"] += str_
        splits["val"] += sva
        splits["test_synth"] = ste
        splits["test_slates_synth"] = make_slate_testset(ste, seed)

    if dcfg.get("label_documents", True):
        splits["train"] += label_documents(taxonomy, labels)
    if dcfg.get("slate_augment_fraction", 0) > 0:
        splits["train"] = slate_augment(splits["train"], dcfg["slate_augment_fraction"], seed)

    rs = dcfg.get("realistic_slates")
    if rs and Path(rs).is_file():
        splits["test_realistic"] = read_jsonl(rs)
    random.Random(seed).shuffle(splits["train"])
    for k, v in splits.items():
        log.info("split %-18s %7d examples, %4d labels", k, len(v), len({e.label for e in v}))
    return dict(splits)
