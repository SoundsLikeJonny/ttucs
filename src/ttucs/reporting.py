"""Class-distribution report: real (BOOM ONE) coverage of UCS vs. the balanced training set."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ttucs.datasets.boom import load_boom_one
from ttucs.synth.generator import Example
from ttucs.taxonomy import UCSTaxonomy


def _raw_boom_counts(path: str | Path, tax: UCSTaxonomy) -> Counter[str]:
    """Rows per CatID in the BOOM ONE sheet (one row = one sound file)."""
    counts: Counter[str] = Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            e = tax.resolve(row.get("CatID") or "")
            if e is not None:
                counts[e.cat_id] += 1
    return counts


def distribution_report(train: list[Example], tax: UCSTaxonomy, out_dir: str | Path,
                        boom_path: str | Path | None = None,
                        exclude: set[str] | frozenset[str] = frozenset({"ARCHIVED"})) -> Path:
    """Write ``distribution.md``, ``class_counts.csv`` and PNG plots into ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [e.cat_id for e in tax.trainable(exclude)]
    by_src: dict[str, Counter[str]] = {}
    for ex in train:
        by_src.setdefault(ex.source, Counter())[ex.label] += 1
    total = Counter({lab: sum(c[lab] for c in by_src.values()) for lab in labels})
    boom_rows: Counter[str] = Counter()
    boom_examples: Counter[str] = Counter()
    if boom_path and Path(boom_path).is_file():
        boom_rows = _raw_boom_counts(boom_path, tax)
        boom_examples = Counter(e.label for e in load_boom_one(boom_path, tax, ("description",), exclude))

    # ---- CSV
    with (out / "class_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        srcs = sorted(by_src)
        w.writerow(["cat_id", "category", "subcategory", "boom_rows", *[f"train_{s}" for s in srcs],
                    "train_total"])
        for lab in labels:
            e = tax[lab]
            w.writerow([lab, e.category, e.subcategory, boom_rows[lab],
                        *[by_src[s][lab] for s in srcs], total[lab]])

    # ---- plot 1: sorted per-CatID counts, BOOM raw vs. balanced training set
    fig, ax = plt.subplots(figsize=(12, 5))
    if boom_rows:
        b = np.sort([boom_rows[lab] for lab in labels])[::-1]
        ax.plot(np.arange(len(b)), np.maximum(b, 0.8), label="BOOM ONE rows (real, unbalanced)")
    t = np.sort([total[lab] for lab in labels])[::-1]
    ax.plot(np.arange(len(t)), t, label="ttucs training set (BOOM capped + synthetic top-up)")
    ax.set_yscale("log")
    ax.set_xlabel(f"UCS CatID rank (n={len(labels)})")
    ax.set_ylabel("examples per CatID (log)")
    ax.set_title("Per-CatID class distribution before and after balancing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=110)
    plt.close(fig)

    # ---- plot 2: stacked sources per UCS Category
    cats = sorted({tax[lab].category for lab in labels})
    fig, ax = plt.subplots(figsize=(16, 6))
    bottom = np.zeros(len(cats))
    for s in sorted(by_src):
        vals = np.array([sum(by_src[s][lab] for lab in labels if tax[lab].category == c) for c in cats])
        ax.bar(cats, vals, bottom=bottom, label=s)
        bottom += vals
    ax.set_xticks(range(len(cats)), cats, rotation=90, fontsize=7)
    ax.set_ylabel("training examples")
    ax.set_title("Training examples per UCS Category by source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "category_sources.png", dpi=110)
    plt.close(fig)

    # ---- markdown
    missing = [lab for lab in labels if boom_rows[lab] == 0] if boom_rows else []
    miss_by_cat = Counter(tax[lab].category for lab in missing)
    vals = np.array([total[lab] for lab in labels])
    lines = ["# Class distribution report", "",
             f"UCS v{tax.version}: {len(tax)} CatIDs, {len(labels)} trainable "
             f"(excluded categories: {', '.join(sorted(exclude)) or 'none'}).", ""]
    if boom_rows:
        bv = np.array([boom_rows[lab] for lab in labels])
        lines += ["## BOOM ONE (real metadata)", "",
                  f"- rows with a valid UCS CatID: {sum(boom_rows.values())}",
                  f"- trainable CatIDs covered: {int((bv > 0).sum())} / {len(labels)}; "
                  f"missing: {len(missing)}",
                  f"- rows per covered CatID: median {np.median(bv[bv > 0]):.0f}, "
                  f"mean {bv[bv > 0].mean():.1f}, max {bv.max()} ({labels[int(bv.argmax())]})",
                  f"- unique prose descriptions (after keyword-tail removal): {sum(boom_examples.values())}",
                  "", "Missing CatIDs per category (top 15): " + ", ".join(
                      f"{c} {n}" for c, n in miss_by_cat.most_common(15)), "",
                  "Top-10 dominant CatIDs: " + ", ".join(
                      f"{c} {n}" for c, n in boom_rows.most_common(10)), ""]
    lines += ["## Balanced training set", "",
              "| source | examples | CatIDs |", "|---|---:|---:|"]
    for s in sorted(by_src):
        lines.append(f"| {s} | {sum(by_src[s].values())} | {len(by_src[s])} |")
    lines += ["", f"- examples per CatID: min {vals.min()}, median {np.median(vals):.0f}, "
              f"max {vals.max()}; imbalance ratio max/min {vals.max() / max(vals.min(), 1):.1f}", "",
              "![class distribution](class_distribution.png)", "",
              "![category sources](category_sources.png)", ""]
    path = out / "distribution.md"
    path.write_text("\n".join(lines))
    return path
