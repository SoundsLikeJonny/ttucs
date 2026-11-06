"""Training entry point with MLflow tracking.

``ttucs train --config configs/bow.yaml`` runs:

1. dataset build (or reuse of ``ttucs generate-data`` output with a matching config hash);
2. model fit with per-epoch metrics;
3. temperature calibration on the validation split;
4. evaluation of every ``test_*`` split (synthetic, BOOM ONE, synthetic slates, hand-written
   realistic slates) plus slate-parser field accuracy;
5. MLflow logging: params, metrics, per-class reports, confusion artifacts, the model as a
   ``pyfunc`` with signature, and (optionally) registration in the model registry.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ttucs._version import __version__
from ttucs.config import flatten
from ttucs.datasets.build import build_datasets, read_jsonl, write_jsonl
from ttucs.evaluate import (
    category_confusion_png,
    evaluate_splits,
    format_table,
    parser_field_accuracy,
    per_class_report,
    predictions_csv,
    top_confusions,
)
from ttucs.models import MODEL_REGISTRY, fit_temperature
from ttucs.pipeline import TTUCS
from ttucs.synth.generator import Example
from ttucs.taxonomy import UCSTaxonomy

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def data_fingerprint(cfg: dict) -> str:
    """Hash of everything that influences the generated splits."""
    payload = {"seed": cfg["seed"], "ucs": cfg["ucs_version"], "data": cfg["data"],
               "version": __version__}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:12]


def generate_data(cfg: dict, out_dir: str | Path | None = None) -> dict[str, list[Example]]:
    """Build all splits and write them as JSONL + ``manifest.json``."""
    seed_everything(int(cfg["seed"]))
    tax = UCSTaxonomy.load(cfg["ucs_version"])
    splits = build_datasets(cfg, tax)
    out = Path(out_dir or cfg["data"].get("processed_dir", "data/processed"))
    out.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        write_jsonl(exs, out / f"{name}.jsonl")
    manifest = {"fingerprint": data_fingerprint(cfg), "ttucs_version": __version__,
                "ucs_version": tax.version,
                "splits": {k: {"examples": len(v), "labels": len({e.label for e in v})}
                           for k, v in splits.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d splits to %s", len(splits), out)
    return splits


def load_or_build(cfg: dict, rebuild: bool = False) -> dict[str, list[Example]]:
    out = Path(cfg["data"].get("processed_dir", "data/processed"))
    man = out / "manifest.json"
    if not rebuild and man.is_file():
        manifest = json.loads(man.read_text())
        if manifest.get("fingerprint") == data_fingerprint(cfg):
            log.info("reusing generated data in %s (fingerprint %s)", out, manifest["fingerprint"])
            return {k: read_jsonl(out / f"{k}.jsonl") for k in manifest["splits"]}
        log.info("data config changed - regenerating")
    return generate_data(cfg, out)


@contextlib.contextmanager
def _mlflow_run(cfg: dict):
    mcfg = cfg.get("mlflow", {})
    if not mcfg.get("enabled", True):
        yield None
        return
    import mlflow

    mlflow.set_tracking_uri(mcfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(mcfg.get("experiment", "ttucs"))
    with mlflow.start_run(run_name=mcfg.get("run_name")) as run:
        yield run


def train(cfg: dict, rebuild_data: bool = False) -> dict[str, Any]:
    """Train, calibrate, evaluate and log a model. Returns a summary dict."""
    seed = int(cfg["seed"])
    seed_everything(seed)
    tax = UCSTaxonomy.load(cfg["ucs_version"])
    splits = load_or_build(cfg, rebuild_data)
    exclude = set(cfg["data"].get("exclude_categories") or [])
    entries = tax.trainable(exclude)
    labels, cats = [e.cat_id for e in entries], [e.category for e in entries]
    mcfg = {**cfg["model"], "ucs_version": tax.version}
    model_cls = MODEL_REGISTRY[mcfg["name"]]
    model = model_cls(labels, cats, mcfg)
    out_dir = Path(cfg["output_dir"])
    train_pairs = [(e.text, e.label) for e in splits.get("train", [])]
    val_pairs = [(e.text, e.label) for e in splits.get("val", [])]

    with _mlflow_run(cfg) as run:
        mlf = None
        if run is not None:
            import mlflow as mlf

            mlf.log_params({k: str(v)[:500] for k, v in flatten(cfg).items()})
            mlf.set_tags({"ttucs_version": __version__, "ucs_version": tax.version,
                          "model": mcfg["name"], "python": platform.python_version(),
                          "torch": torch.__version__})
            mlf.log_params({"n_labels": len(labels), "n_train": len(train_pairs),
                            "n_val": len(val_pairs)})

        def on_epoch(epoch: int, metrics: dict) -> None:
            if mlf is not None:
                mlf.log_metrics({k: float(v) for k, v in metrics.items()}, step=epoch)

        t0 = time.time()
        history = model.fit(train_pairs, val_pairs, {**cfg["train"], "seed": seed}, on_epoch)
        train_seconds = time.time() - t0
        log.info("training finished in %.1fs", train_seconds)

        # ---- calibration on validation logits
        temperature = 1.0
        if val_pairs and cfg["train"].get("calibrate", True):
            val_logits = model.logits([t for t, _ in val_pairs])
            temperature = fit_temperature(val_logits, model.encode_labels(lab for _, lab in val_pairs))
        log.info("temperature %.3f", temperature)

        inference = {**cfg.get("inference", {})}
        model.save(out_dir, extra={"temperature": temperature, "ucs_version": tax.version,
                                   "inference": inference, "ttucs_version": __version__,
                                   "data_fingerprint": data_fingerprint(cfg), "seed": seed})
        pipe = TTUCS.from_pretrained(out_dir)

        # ---- evaluation
        metrics, details = evaluate_splits(pipe, splits)
        parser_metrics = parser_field_accuracy(splits.get("test_slates_synth", []), pipe.parser)
        print(format_table(metrics))
        summary = {"metrics": metrics, "parser": parser_metrics, "temperature": temperature,
                   "train_seconds": train_seconds, "output_dir": str(out_dir),
                   "history": {k: list(map(float, v)) for k, v in (history or {}).items()}}
        report_dir = out_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        for split, d in details.items():
            per_class_report(pipe, d, report_dir / f"per_class_{split}.csv")
            top_confusions(pipe, d, report_dir / f"confusions_{split}.csv")
            category_confusion_png(pipe, d, report_dir / f"category_confusion_{split}.png",
                                   title=f"{split}: UCS category confusion (row-normalised)")
        if splits.get("test_realistic"):
            predictions_csv(pipe, splits["test_realistic"], report_dir / "predictions_test_realistic.csv")

        if mlf is not None:
            for split, m in metrics.items():
                mlf.log_metrics({f"{split}.{k}": float(v) for k, v in m.items()})
            mlf.log_metrics({**parser_metrics, "temperature": temperature,
                             "train_seconds": train_seconds})
            mlf.log_artifacts(str(report_dir), artifact_path="reports")
            mlf.log_artifact(str(out_dir / "config.json"), artifact_path="reports")
            if cfg["mlflow"].get("log_model", True):
                from ttucs.mlflow_model import log_pyfunc

                reg = cfg["mlflow"].get("registered_model_name") if cfg["mlflow"].get(
                    "register_model", True) else None
                examples = [e.text for e in splits.get("test_realistic", [])[:3]] or None
                info = log_pyfunc(out_dir, registered_model_name=reg, example_texts=examples)
                summary["model_uri"] = info.model_uri
                summary["registered_model_version"] = getattr(info, "registered_model_version", None)
            summary["mlflow_run_id"] = run.info.run_id
            (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
