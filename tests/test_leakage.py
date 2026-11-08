"""scripts/check_leakage.py: bank-phrase and dev-vs-realistic overlap checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_leakage", ROOT / "scripts" / "check_leakage.py")
cl = importlib.util.module_from_spec(_spec)
sys.modules["check_leakage"] = cl  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(cl)

BANKS = {
    "catid_phrases": {"DOORWood": ["wooden barn door", "screen door slam"]},
    "subcategory_actions": {"BREAK": ["crash and debris"]},
    "contexts": ["in a small room"],
}


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_phrase_overlap_detected_and_short_phrases_ignored():
    dev = [("dev:0", "a wooden barn door closes")]
    realistic = [("real:0", "quiet street at night")]
    v = cl.check_phrases(BANKS, dev, realistic)
    assert len(v) == 1 and "wooden barn door" in v[0].detail and v[0].check == "phrase"
    assert cl.check_phrases({"catid_phrases": {"DOORWood": ["barn door"]}}, dev, realistic) == []


def test_cross_4gram_and_substring():
    dev = [("dev:1", "sedan starts up idles then drives away")]
    realistic = [("real:1", "location street, sedan starts up idles then drives away fast, take 2")]
    v = cl.check_cross(dev, realistic)
    assert {x.check for x in v} == {"cross_4gram"}
    # a dev text too short to have a 4-gram is still caught by the substring branch
    v = cl.check_cross([("dev:3", "sedan idles")], [("real:3", "street corner, sedan idles at the curb")])
    assert {x.check for x in v} == {"substring"}
    assert cl.check_cross([("dev:2", "waves on rocks")], [("real:2", "dog barking at night")]) == []


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cl, "load_banks", lambda: BANKS)
    dev = tmp_path / "dev.jsonl"
    _jsonl(dev, [{"text": "large metal gate shuts hard", "label": "METAGate", "group": "dev:0"}])
    clean = tmp_path / "real.jsonl"
    _jsonl(clean, [{"text": "bird singing at dawn", "label": "AMBBird", "group": "real:0"}])
    assert cl.main(["--dev", str(dev), "--test", str(clean)]) == 0
    assert "leakage check clean" in capsys.readouterr().out
    bad = tmp_path / "bad.jsonl"
    _jsonl(bad, [{"text": "a wooden barn door closes", "label": "DOORWood", "group": "dev:0"}])
    assert cl.main(["--dev", str(bad), "--test", str(clean)]) == 1
    assert "LEAKAGE CHECK FAILED" in capsys.readouterr().out
