"""YAML experiment configuration with defaults and dotted command-line overrides.

A tiny, dependency-free take on the Hydra/Lightning-CLI pattern::

    cfg = load_config("configs/bow.yaml", ["train.epochs=3", "data.boom.enabled=false"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "ucs_version": "8.2.1",
    "output_dir": "models/ttucs-bow",
    "data": {
        "exclude_categories": ["ARCHIVED"],
        "synthetic": {
            "enabled": True,
            "target_per_class": 150,   # synthetic tops up real data to this many examples
            "min_per_class": 60,       # always generate at least this many per CatID
            "max_synonym_df": 25,
            "cache": "data/processed/synthetic.jsonl",
        },
        "boom": {
            "enabled": True,
            "path": "data/raw/boom_one.csv",
            "use_for_training": True,
            "max_per_class": 300,      # cap dominant classes (OBJCont has >1000 rows)
            "text_fields": ["description", "fxname"],
        },
        "label_documents": True,       # add UCS explanation/synonym docs as training text
        "slate_augment_fraction": 0.25,
        "split": {"val": 0.1, "test": 0.1},
        "realistic_slates": "examples/realistic_slates.jsonl",
        "processed_dir": "data/processed",
    },
    "model": {
        "name": "bow",
        "embedding_dim": 128,
        "num_buckets": 262144,
        "word_ngrams": 2,
        "char_ngrams": [3, 5],
        "dropout": 0.2,
        "category_loss_weight": 0.3,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "hidden_dim": 512,
    },
    "train": {
        "epochs": 12,
        "batch_size": 256,
        "lr": 1.0e-2,
        "weight_decay": 1.0e-5,
        "label_smoothing": 0.05,
        "patience": 3,
        "num_threads": 8,
        "calibrate": True,
    },
    "inference": {"unknown_threshold": 0.12, "category_threshold": 0.5, "top_k": 5},
    "mlflow": {
        "enabled": True,
        "tracking_uri": "sqlite:///mlflow.db",
        "experiment": "ttucs",
        "register_model": True,
        "registered_model_name": "ttucs",
        "log_model": True,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = DEFAULTS
    if path:
        with open(path, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    return apply_overrides(cfg, overrides)


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config for MLflow params: {'a': {'b': 1}} -> {'a.b': 1}."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
