"""Class-distribution report for the generated training set (per CatID and per UCS Category)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ttucs.synth.generator import Example
from ttucs.taxonomy import UCSTaxonomy


def distribution_report(train: list[Example], tax: UCSTaxonomy, out_dir: str | Path,
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

    # ---- CSV
    with (out / "class_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        srcs = sorted(by_src)
        w.writerow(["cat_id", "category", "subcategory", *[f"train_{s}" for s in srcs],
                    "train_total"])
        for lab in labels:
            e = tax[lab]
            w.writerow([lab, e.category, e.subcategory,
                        *[by_src[s][lab] for s in srcs], total[lab]])

    # ---- plot 1: sorted per-CatID counts of the training set
    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.sort([total[lab] for lab in labels])[::-1]
    ax.plot(np.arange(len(t)), np.maximum(t, 0.8), label="ttucs training set")
    ax.set_yscale("log")
    ax.set_xlabel(f"UCS CatID rank (n={len(labels)})")
    ax.set_ylabel("examples per CatID (log)")
    ax.set_title("Per-CatID class distribution of the training set")
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
    vals = np.array([total[lab] for lab in labels])
    lines = ["# Class distribution report", "",
             f"UCS v{tax.version}: {len(tax)} CatIDs, {len(labels)} trainable "
             f"(excluded categories: {', '.join(sorted(exclude)) or 'none'}).", ""]
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
