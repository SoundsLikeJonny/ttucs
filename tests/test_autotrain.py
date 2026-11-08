"""scripts/autotrain.py: versioning, gating, records, changelog edits and CLI dry-run paths."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("autotrain", ROOT / "scripts" / "autotrain.py")
at = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("autotrain", at)
_spec.loader.exec_module(at)

METRICS = {"dev.top1": 0.789, "dev.top5": 0.925, "dev.macro_f1": 0.788, "dev.ece": 0.069,
           "test_synth.top1": 0.908, "test_synth.top5": 0.983, "test_synth.macro_f1": 0.904,
           "test_slates_synth.top1": 0.900, "test_slates_synth.top5": 0.976,
           "test_slates_synth.macro_f1": 0.884}
PARSER = {"parser_description_exact": 0.921, "parser_description_token_f1": 0.978}
CHAMPION = {"weights_version": "0.2.0", "note": "test", "evaluated_at": "2026-11-07T00:00:00Z",
            "git_sha": "abc12345", "code_fingerprint": "aaaa", "config_hash": "bbbb", "seed": 42,
            "metrics": dict(METRICS), "parser": dict(PARSER)}


def _candidate(**over: float) -> dict:
    metrics = dict(METRICS)
    metrics.update(over)
    return {"metrics": metrics, "parser": dict(PARSER), "output_dir": "models/x",
            "mlflow_run_id": "r1", "seed": 42, "git_sha": "abc12345", "trained_at": "t"}


def test_next_beta_version():
    assert at.next_beta_version("0.2.0") == "0.2.1b1"
    assert at.next_beta_version("0.2.1b1") == "0.2.1b2"
    assert at.next_beta_version("1.10.3") == "1.10.4b1"
    with pytest.raises(ValueError):
        at.next_beta_version("dev")


def test_upper_bound():
    assert at._upper_bound("0.2.0") == "0.3"
    assert at._upper_bound("0.2.1b1") == "0.3"
    assert at._upper_bound("1.10.3") == "1.11"


def test_replace_once(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text('a version = "0.2.0" b\n', encoding="utf-8")
    at._replace_once(f, 'version = "0.2.0"', 'version = "0.2.1b1"')
    assert 'version = "0.2.1b1"' in f.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        at._replace_once(f, "zzz", "x")
    f.write_text("x x\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        at._replace_once(f, "x", "y")


def test_gate_pass():
    passed, report = at.gate(CHAMPION, _candidate(**{"dev.top1": 0.812}))
    assert passed and "RESULT: PASSED" in report


def test_gate_fails_on_insufficient_gain():
    passed, report = at.gate(CHAMPION, _candidate(**{"dev.top1": 0.792}))
    assert not passed and "RESULT: FAILED" in report


def test_gate_fails_on_guardrail_regression():
    assert not at.gate(CHAMPION, _candidate(**{"dev.top1": 0.82, "dev.top5": 0.890}))[0]
    assert not at.gate(CHAMPION, _candidate(**{"dev.top1": 0.82, "dev.ece": 0.090}))[0]
    assert not at.gate(CHAMPION, _candidate(**{"dev.top1": 0.82, "test_synth.macro_f1": 0.850}))[0]


def test_gate_ece_improvement_passes():
    assert at.gate(CHAMPION, _candidate(**{"dev.top1": 0.82, "dev.ece": 0.050}))[0]


def test_gate_fails_on_parser_regression_and_missing_metrics():
    cand = _candidate(**{"dev.top1": 0.82})
    cand["parser"] = {"parser_description_exact": 0.921, "parser_description_token_f1": 0.900}
    assert not at.gate(CHAMPION, cand)[0]
    del cand["metrics"]["dev.ece"]
    assert not at.gate(CHAMPION, cand)[0]


def test_leakage_gate_blocks_new_violations_only(monkeypatch):
    status = {"phrase": 3, "cross": 0, "soft": 10,
              "phrase_details": ["d1", "d2", "brand new"], "cross_details": []}
    monkeypatch.setattr(at, "leakage_status", lambda: status)
    base = {"phrase": 2, "cross": 0, "phrase_details": ["d1", "d2"], "cross_details": []}
    bad = at.leakage_gate(base)
    assert not bad["passed"] and "NEW catid-phrase match" in bad["report"]
    ok = at.leakage_gate({"phrase": 3, "cross": 0, "phrase_details": status["phrase_details"],
                          "cross_details": []})
    assert ok["passed"]
    crossed = dict(status, cross=1, cross_details=["dev:0 real:1 shares 4-gram"])
    monkeypatch.setattr(at, "leakage_status", lambda: crossed)
    assert not at.leakage_gate(base)["passed"]


def test_candidate_record():
    summary = {"output_dir": "models/x", "mlflow_run_id": "abc", "train_seconds": 10,
               "metrics": {"test_synth": {"n": 100, "top1": 0.9, "top5": 0.98, "macro_f1": 0.89,
                                          "ece": 0.02},
                           "test_slates_synth": {"top1": 0.88, "top5": 0.97, "macro_f1": 0.87,
                                                 "ece": 0.03}},
               "parser": {"parser_description_exact": 0.92, "parser_description_token_f1": 0.97,
                          "parser_other_acc": 0.5}}
    dev_m = {"n": 332, "top1": 0.81, "top5": 0.93, "macro_f1": 0.80, "ece": 0.05}
    rec = at.candidate_record({"seed": 42}, summary, dev_m, {"sha": "ab", "branch": "b"}, "fp12345")
    assert rec["metrics"]["dev.top1"] == 0.81 and rec["metrics"]["test_synth.top1"] == 0.9
    assert "ece" not in rec["parser"] and "parser_other_acc" not in rec["parser"]
    assert set(rec["parser"]) == set(PARSER)
    assert rec["code_fingerprint"] == "fp12345" and rec["seed"] == 42


def test_parse_sweep():
    assert at.parse_sweep(["train.epochs=8,15"]) == [{"train.epochs": 8}, {"train.epochs": 15}]
    combos = at.parse_sweep(["train.epochs=8,15", "model.dropout=0.1,0.3"])
    assert len(combos) == 4 and {"train.epochs": 8, "model.dropout": 0.1} in combos
    with pytest.raises(ValueError):
        at.parse_sweep(["badspec"])


def test_changelog_entries(tmp_path):
    pkg = tmp_path / "README.md"
    pkg.write_text("# pkg\n\n## Changelog\n\n### 0.2.0 - 2026-11-07\n- old\n", encoding="utf-8")
    at._pkg_changelog_entry("0.2.1b1", "dev top1 0.789 -> 0.810", pkg)
    text = pkg.read_text(encoding="utf-8")
    assert "### 0.2.1b1 - " in text and "dev top1 0.789 -> 0.810" in text
    assert text.index("### 0.2.1b1") < text.index("### 0.2.0")

    ch = tmp_path / "CHANGELOG.md"
    ch.write_text("# Changelog\n\n## [Unreleased]\n\n### Added\n- something\n\n"
                  "## [0.3.0] - 2026-11-07\n", encoding="utf-8")
    at._changelog_entry("0.2.1b1", "dev top1 0.789 -> 0.810", ch)
    text = ch.read_text(encoding="utf-8")
    assert "### Changed\n- Beta weights `ttucs-model-bow` 0.2.1b1" in text
    assert text.index("### Changed") < text.index("## [0.3.0]")
    at._changelog_entry("0.2.1b2", "second beta", ch)
    text = ch.read_text(encoding="utf-8")
    assert "second beta" in text and text.count("### Changed\n") == 1


def test_comparison_png(tmp_path):
    out = at.comparison_png(CHAMPION, _candidate(**{"dev.top1": 0.812}),
                            tmp_path / "cmp.png")
    assert out is not None and out.is_file() and out.stat().st_size > 1000


def test_resolve_published_dir_prefers_exact_version(monkeypatch, tmp_path):
    from ttucs import hub

    good = tmp_path / "good"
    good.mkdir()
    (good / "config.json").write_text(json.dumps({"model_version": "0.2.0"}), encoding="utf-8")
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "config.json").write_text(json.dumps({"model_version": "0.1.1"}), encoding="utf-8")
    monkeypatch.setattr(hub, "packaged_model_dir", lambda name="bow": good)
    assert at._resolve_published_dir(hub.MODELS["bow"]) == good
    monkeypatch.setattr(hub, "packaged_model_dir", lambda name="bow": stale)
    monkeypatch.setattr(hub, "cached_model_dir", lambda name="bow": good)
    assert at._resolve_published_dir(hub.MODELS["bow"]) == good
    monkeypatch.setattr(hub, "cached_model_dir", lambda name="bow": tmp_path / "missing")
    monkeypatch.setattr(hub, "is_offline", lambda: True)
    assert at._resolve_published_dir(hub.MODELS["bow"]) is None


def test_main_needs_champion(monkeypatch, capsys):
    monkeypatch.setattr(at, "load_champion", lambda path=None: None)
    assert at.main([]) == 2
    assert "init-champion" in capsys.readouterr().out


def test_main_nothing_to_do(monkeypatch, capsys):
    import ttucs.provenance as prov

    monkeypatch.setattr(at, "load_champion", lambda path=None: CHAMPION)
    monkeypatch.setattr(prov, "changed_vs", lambda champion: False)
    assert at.main([]) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_main_dry_run_and_leakage_block(monkeypatch, capsys):
    import ttucs.provenance as prov

    monkeypatch.setattr(at, "load_champion", lambda path=None: CHAMPION)
    monkeypatch.setattr(prov, "changed_vs", lambda champion: True)
    monkeypatch.setattr(at, "leakage_gate",
                        lambda base: {"passed": True, "report": "leak ok", "status": {}})
    assert at.main([]) == 0
    out = capsys.readouterr().out
    assert "change detected" in out and "dry run" in out
    monkeypatch.setattr(at, "leakage_gate",
                        lambda base: {"passed": False, "report": "leak bad", "status": {}})
    assert at.main([]) == 1
