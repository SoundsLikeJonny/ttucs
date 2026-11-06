"""Evaluation: classification metrics, calibration and slate-parser field accuracy."""

from __future__ import annotations

import csv
import itertools
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ttucs.pipeline import TTUCS
from ttucs.slate.parser import SlateParser
from ttucs.synth.generator import Example

# splits whose text is a full slate transcript (go through the parser first)
SLATE_SPLITS = ("test_slates_synth", "test_realistic")


def classification_metrics(probs: np.ndarray, y: np.ndarray, label_to_cat: np.ndarray,
                           threshold: float | None = None, n_bins: int = 15) -> dict[str, float]:
    """top-1/top-5/macro-F1/category accuracy/ECE (and coverage at ``threshold``)."""
    if len(y) == 0:
        return {}
    from ttucs._extras import require_train_extras

    require_train_extras("evaluation")
    from sklearn.metrics import f1_score

    pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    conf = probs.max(1)
    correct = pred == y
    present = np.unique(y)
    m = {
        "n": float(len(y)),
        "top1": float(correct.mean()),
        "top5": float((top5 == y[:, None]).any(1).mean()),
        # macro-F1 over the classes that occur in this split (sklearn's ``labels=``)
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "category_acc": float((label_to_cat[pred] == label_to_cat[y]).mean()),
        "ece": expected_calibration_error(conf, correct, n_bins),
        "mean_confidence": float(conf.mean()),
    }
    if threshold is not None:
        keep = conf >= threshold
        m["coverage"] = float(keep.mean())
        m["selective_top1"] = float(correct[keep].mean()) if keep.any() else 0.0
    return m


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(bins):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate_split(pipe: TTUCS, examples: Sequence[Example], parse: bool) -> tuple[dict, dict]:
    """Return (metrics, details) for one split. ``parse`` runs the slate parser first."""
    labels = [e.label for e in examples if e.label in pipe.model.label_index]
    examples = [e for e in examples if e.label in pipe.model.label_index]
    if not examples:
        return {}, {}
    if parse:
        descs = [pipe.parser.parse(e.text).description for e in examples]
    else:
        descs = [e.text for e in examples]
    probs = pipe.probabilities(descs)
    y = pipe.model.encode_labels(labels)
    metrics = classification_metrics(probs, y, pipe.model.label_to_cat, pipe.unknown_threshold)
    return metrics, {"descriptions": descs, "probs": probs, "y": y}


def evaluate_splits(pipe: TTUCS, splits: dict[str, Sequence[Example]]) -> tuple[dict, dict]:
    metrics, details = {}, {}
    for name, exs in splits.items():
        if not name.startswith("test") or not exs:
            continue
        m, d = evaluate_split(pipe, exs, parse=name in SLATE_SPLITS)
        if m:
            metrics[name], details[name] = m, d
    return metrics, details


# ------------------------------------------------------------------------- parser
_PARSER_FIELDS = ("description", "project", "location", "creator", "microphones", "mic_setup",
                  "recorder", "sample_rate", "bit_depth", "take", "date")


def _norm(v: Any) -> str:
    if isinstance(v, list):
        return "|".join(sorted(_norm(x) for x in v))
    s = re.sub(r"[^a-z0-9.| ]", " ", str(v).lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def parser_field_accuracy(examples: Sequence[Example], parser: SlateParser | None = None
                          ) -> dict[str, float]:
    """Exact-match accuracy per field (and token F1 for the description) on synthetic slates
    whose ground truth is stored in ``example.meta['truth']``."""
    parser = parser or SlateParser()
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    desc_f1 = []
    for ex in examples:
        truth = ex.meta.get("truth")
        if not truth:
            continue
        got = parser.parse(ex.text).to_dict()
        for f in _PARSER_FIELDS:
            if f not in truth:
                continue
            if f == "description":
                desc_f1.append(token_f1(got[f], truth[f]))
                continue
            totals[f] += 1
            hits[f] += int(_norm(got.get(f) or "") == _norm(truth[f]))
    out = {f"parser_{f}_acc": hits[f] / totals[f] for f in totals}
    if desc_f1:
        out["parser_description_token_f1"] = float(np.mean(desc_f1))
        out["parser_description_exact"] = float(np.mean([x == 1.0 for x in desc_f1]))
    return out


# ------------------------------------------------------------------------ reports
def per_class_report(pipe: TTUCS, details: dict, path: str | Path) -> Path:
    """CSV with support / precision / recall / F1 per CatID."""
    from sklearn.metrics import precision_recall_fscore_support

    y, pred = details["y"], details["probs"].argmax(1)
    present = np.unique(np.concatenate([y, pred]))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cat_id", "category", "subcategory", "support", "precision", "recall", "f1"])
        for i, lab in enumerate(present):
            e = pipe.taxonomy[pipe.labels[lab]]
            w.writerow([e.cat_id, e.category, e.subcategory, int(s[i]), f"{p[i]:.4f}", f"{r[i]:.4f}",
                        f"{f[i]:.4f}"])
    return path


def top_confusions(pipe: TTUCS, details: dict, path: str | Path, k: int = 100) -> Path:
    y, pred = details["y"], details["probs"].argmax(1)
    pairs = Counter((int(a), int(b)) for a, b in zip(y, pred, strict=True) if a != b)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true_cat_id", "predicted_cat_id", "count"])
        for (a, b), c in pairs.most_common(k):
            w.writerow([pipe.labels[a], pipe.labels[b], c])
    return path


def category_confusion_png(pipe: TTUCS, details: dict, path: str | Path, title: str = "") -> Path:
    """Row-normalised confusion matrix at UCS *Category* level (82x82 is readable; 738x738 is not)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    l2c = pipe.model.label_to_cat
    yt, yp = l2c[details["y"]], l2c[details["probs"].argmax(1)]
    n = len(pipe.model.categories)
    cm = np.zeros((n, n))
    np.add.at(cm, (yt, yp), 1)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(16, 15))
    ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), pipe.model.categories, rotation=90, fontsize=6)
    ax.set_yticks(range(n), pipe.model.categories, fontsize=6)
    ax.set_xlabel("predicted category")
    ax.set_ylabel("true category")
    ax.set_title(title or "UCS category confusion (row-normalised)")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def predictions_csv(pipe: TTUCS, examples: Sequence[Example], path: str | Path) -> Path:
    """Human-readable predictions (used for the hand-written realistic slates)."""
    preds = pipe.predict_batch([e.text for e in examples], top_k=5)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "description", "true", "predicted", "confidence", "decision", "top5_hit",
                    "top5"])
        for e, p in zip(examples, preds, strict=True):
            top5 = [a.cat_id for a in p.alternatives]
            w.writerow([e.text, p.description, e.label, p.alternatives[0].cat_id,
                        f"{p.alternatives[0].confidence:.3f}", p.decision, int(e.label in top5),
                        " ".join(top5)])
    return path


def format_table(metrics: dict[str, dict[str, float]]) -> str:
    cols = ["n", "top1", "top5", "macro_f1", "category_acc", "ece", "coverage", "selective_top1"]
    lines = [f"{'split':<20}" + "".join(f"{c:>15}" for c in cols)]
    for split, m in metrics.items():
        row = f"{split:<20}"
        for c in cols:
            v = m.get(c)
            row += f"{'':>15}" if v is None else (f"{int(v):>15d}" if c == "n" else f"{v:>15.4f}")
        lines.append(row)
    return "\n".join(lines)
