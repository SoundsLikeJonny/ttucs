"""Autotrain building blocks: provenance tags, the training-history plot, the autotrain config."""

from __future__ import annotations

import re
from pathlib import Path

from ttucs.config import load_config
from ttucs.provenance import changed_vs, code_fingerprint, git_info, run_tags
from ttucs.reporting import training_history_png

ROOT = Path(__file__).resolve().parents[1]


def test_code_fingerprint_is_stable():
    a = code_fingerprint()
    assert re.fullmatch(r"[0-9a-f]{12}", a)
    assert code_fingerprint() == a


def test_git_info_and_run_tags_shape():
    info = git_info()
    assert set(info) == {"sha", "branch", "dirty"}
    assert info["sha"] == "unknown" or re.fullmatch(r"[0-9a-f]{7,}", info["sha"])
    tags = run_tags()
    assert set(tags) == {"git_sha", "git_branch", "git_dirty", "code_fingerprint"}
    assert tags["git_dirty"] in ("true", "false", "unknown")
    assert tags["code_fingerprint"] == code_fingerprint()


def test_changed_vs_champion_record():
    assert changed_vs({}) is True
    assert changed_vs({"code_fingerprint": "deadbeefdead"}) is True
    assert changed_vs({"code_fingerprint": code_fingerprint()}) is False


def test_autotrain_config_excludes_realistic_slates():
    cfg = load_config(ROOT / "configs" / "autotrain.yaml")
    assert cfg["data"]["realistic_slates"] is None
    assert cfg["mlflow"]["enabled"] is True and cfg["mlflow"]["run_name"] == "autotrain"
    assert cfg["data"]["synthetic"]["target_per_class"] == 300
    assert cfg["output_dir"] == "models/ttucs-bow-autotrain"


def test_training_history_png(tmp_path):
    history = {"train_loss": [0.9, 0.6, 0.4], "val_top1": [0.5, 0.62, 0.7], "best_val_top1": [0.7]}
    out = training_history_png(history, tmp_path / "hist.png", title="t")
    assert out is not None and out.is_file() and out.stat().st_size > 1000
    assert training_history_png({}, tmp_path / "none.png") is None
    assert not (tmp_path / "none.png").exists()
