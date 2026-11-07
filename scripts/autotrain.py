"""MLflow-gated auto-train loop for ttucs.

The loop (AGENTS.md sections 1, 6 and 9):

1. Fingerprint the training code and resources (``ttucs.provenance``) and compare with
   ``configs/champion.json``; exit "nothing to do" when nothing changed.
2. Leakage checks: NEW per-CatID phrase matches in the hand-written sets, or NEW dev/realistic
   4-gram or substring overlap, block the round. Generic synth-bank phrases are reported only -
   the hand-written sets already contain some of those (see champion.json ``leakage_baseline``).
3. Train a candidate with ``configs/autotrain.yaml``. That config never loads the held-out
   realistic slates: the loop only sees the tuning signals (dev descriptions + synthetic splits).
   Every round is logged to MLflow with git/fingerprint tags and a training-history plot;
   ``mlflow ui`` shows the runs and curves.
4. Gate the candidate against the champion: dev top-1 must improve by at least ``--min-gain``
   (default 0.015, about 5 of the 332 dev descriptions) and the guardrails (dev top-5 /
   macro-F1 / ECE, synthetic top-1 / top-5 / macro-F1, parser exact / token-F1) must hold.
5. On failure, print a failure report (gate table + top confusions) to guide the next change.
6. On success, with ``--release``: bump a PEP 440 beta version, build the deterministic weights
   tarball + wheel, upload the tarball as a GitHub Release asset, update every pinned file
   (``hub.MODELS``, ``RELEASE_SHA256``, the dependency range, the package version, changelogs,
   ``champion.json``) and open a PR with the comparison numbers. It never merges, never pushes
   ``main`` and never publishes to PyPI - those stay with the owner (AGENTS.md rules 1 and 9).
   The realistic slates are then scored once, manually, before a beta is promoted to final.

Champion metrics are seeded from the published weights (``--init-champion``) and refreshed by
every beta release, so each gate compares against the current best.

Usage::

    python scripts/autotrain.py --init-champion          # seed configs/champion.json
    python scripts/autotrain.py                          # dry run: report the change + plan
    python scripts/autotrain.py --train                  # one candidate round, no release
    python scripts/autotrain.py --train --confirm-seed   # re-run the winner with seed 7
    python scripts/autotrain.py --train --sweep train.epochs=8,15 model.dropout=0.1,0.3
    python scripts/autotrain.py --train --release        # passing gate -> beta PR (needs gh)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = ROOT / "configs" / "champion.json"
AUTOTRAIN_CFG_PATH = ROOT / "configs" / "autotrain.yaml"
DEV_PATH = ROOT / "examples" / "dev_descriptions.jsonl"
REALISTIC_PATH = ROOT / "examples" / "realistic_slates.jsonl"
PKG_DIR = ROOT / "packages" / "ttucs-model-bow"

GATE_METRIC = "dev.top1"
DEFAULT_MIN_GAIN = 0.015          # ~5 of the 332 dev descriptions (AGENTS.md section 6)
GUARD_TOL = 0.005
DEFAULT_SEED = 7                  # second seed for confirming small gains
MAX_ROUNDS = 5                    # per invocation; protects the small dev set from over-tuning
REPORT_KEYS = ("dev.top1", "dev.top5", "dev.macro_f1", "dev.ece",
               "test_synth.top1", "test_synth.top5", "test_synth.macro_f1",
               "test_slates_synth.top1", "test_slates_synth.top5", "test_slates_synth.macro_f1")
#: metric -> tolerance. Positive: candidate must be >= champion - tol. Negative (ECE):
#: candidate must be <= champion + |tol|.
GUARDRAILS = {"dev.top5": GUARD_TOL, "dev.macro_f1": GUARD_TOL, "dev.ece": -GUARD_TOL,
              "test_synth.top1": GUARD_TOL, "test_synth.top5": GUARD_TOL,
              "test_synth.macro_f1": GUARD_TOL, "test_slates_synth.top1": GUARD_TOL,
              "test_slates_synth.top5": GUARD_TOL}
PARSER_GUARDRAILS = {"parser_description_exact": GUARD_TOL,
                     "parser_description_token_f1": GUARD_TOL}

#: ttucs-model-bow 0.2.0 published numbers (README / MODEL_CARD); fallback for --init-champion
#: when the installed weights package cannot be evaluated live.
PUBLISHED_METRICS = {"dev.top1": 0.789, "dev.top5": 0.925, "dev.macro_f1": 0.788, "dev.ece": 0.069,
                     "test_synth.top1": 0.908, "test_synth.top5": 0.983, "test_synth.macro_f1": 0.904,
                     "test_slates_synth.top1": 0.900, "test_slates_synth.top5": 0.976,
                     "test_slates_synth.macro_f1": 0.884}
PUBLISHED_PARSER = {"parser_description_exact": 0.921, "parser_description_token_f1": 0.978}


# --------------------------------------------------------------------------- small helpers
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_hash(cfg: dict) -> str:
    """Stable short hash of the effective config (the champion records it for reference)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def load_champion(path: Path = CHAMPION_PATH) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_champion(record: dict, path: Path = CHAMPION_PATH) -> Path:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old[:60]!r}... in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


_BETA_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:b(\d+))?$")


def next_beta_version(current: str) -> str:
    """``0.2.0 -> 0.2.1b1``; ``0.2.1b2 -> 0.2.1b3`` (PEP 440 pre-release numbering)."""
    m = _BETA_RE.match(current.strip())
    if not m:
        raise ValueError(f"cannot derive a beta version from {current!r}")
    x, y, z, b = m.groups()
    return f"{x}.{y}.{int(z) + 1}b1" if b is None else f"{x}.{y}.{z}b{int(b) + 1}"


def _upper_bound(version: str) -> str:
    """The ``<X.Y`` of the dependency range for a weights version (0.2.x -> 0.3)."""
    x, y = _BETA_RE.match(version.strip()).groups()[:2]
    return f"{x}.{int(y) + 1}"


def _leakage():
    """Load scripts/check_leakage.py as a module (scripts/ is not a package)."""
    if "check_leakage" in sys.modules:
        return sys.modules["check_leakage"]
    spec = importlib.util.spec_from_file_location("check_leakage", ROOT / "scripts" / "check_leakage.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_leakage"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(cmd: list[str], **kwargs: Any) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({out.returncode}):\n{out.stderr.strip()}")
    return out.stdout.strip()


# --------------------------------------------------------------------------- leakage gate
def leakage_status() -> dict:
    """Overlap counts between the banks and the hand-written sets.

    ``phrase`` counts per-CatID bank-phrase matches (hard), ``cross`` counts dev/realistic
    4-gram or substring overlaps (hard), ``soft`` counts generic synth-bank phrase matches
    (reported only - a few pre-exist in the hand-written sets).
    """
    cl = _leakage()
    from ttucs.synth.generator import load_banks

    banks = load_banks()
    dev, realistic = cl.read_texts(DEV_PATH), cl.read_texts(REALISTIC_PATH)
    hard = cl.check_phrases({"catid_phrases": banks.get("catid_phrases") or {}}, dev, realistic)
    all_phrase = cl.check_phrases(banks, dev, realistic)
    cross = cl.check_cross(dev, realistic)
    return {"phrase": len(hard), "cross": len(cross), "soft": len(all_phrase) - len(hard),
            "phrase_details": sorted(v.detail for v in hard),
            "cross_details": sorted(v.detail for v in cross),
            "soft_details": sorted(v.detail for v in all_phrase if v not in hard)}


def leakage_gate(baseline: dict) -> dict:
    """Pass unless the counts got WORSE than the champion baseline. Pre-existing matches do
    not block a round; they are the owner's to clean up (recorded in champion.json)."""
    status = leakage_status()
    base_phrase = set(baseline.get("phrase_details") or [])
    base_cross = set(baseline.get("cross_details") or [])
    new_phrase = [d for d in status["phrase_details"] if d not in base_phrase]
    new_cross = [d for d in status["cross_details"] if d not in base_cross]
    passed = not new_phrase and not new_cross
    lines = [f"leakage: catid-phrase matches {status['phrase']} (baseline {len(base_phrase)}), "
             f"dev/realistic overlaps {status['cross']} (baseline {len(base_cross)}), "
             f"generic-bank warnings {status['soft']}"]
    for kind, items in (("NEW catid-phrase match", new_phrase), ("NEW dev/realistic overlap", new_cross)):
        for d in items:
            lines.append(f"  {kind}: {d}")
    if not passed:
        lines.append("  fix these first (AGENTS.md section 6): never copy hand-set wording into banks")
    return {"passed": passed, "status": status, "report": "\n".join(lines)}


# --------------------------------------------------------------------------- candidate
def train_candidate(cfg: dict, run_name: str) -> dict:
    """One training round via ttucs.train.train (MLflow logging happens inside it)."""
    from ttucs.train import train

    summary = train({**cfg, "mlflow": {**cfg.get("mlflow", {}), "run_name": run_name}})
    print(f"trained candidate in {summary.get('train_seconds', 0):.0f}s -> {summary['output_dir']}"
          f" (mlflow run {summary.get('mlflow_run_id')})")
    return summary


def dev_metrics(model_dir: str | Path) -> dict:
    """Metrics on the 332 hand-written dev descriptions (parse=False: they are descriptions)."""
    from ttucs.datasets.build import read_jsonl
    from ttucs.evaluate import evaluate_split
    from ttucs.pipeline import TTUCS

    m, _ = evaluate_split(TTUCS.from_pretrained(model_dir), read_jsonl(DEV_PATH), parse=False)
    return m


def candidate_record(cfg: dict, summary: dict, dev_m: dict, git: dict, fingerprint: str) -> dict:
    """The comparable record of one trained candidate (same shape as champion.json)."""
    keep = ("top1", "top5", "macro_f1", "ece")
    metrics: dict[str, float] = {}
    for split in ("test_synth", "test_slates_synth"):
        metrics.update({f"{split}.{k}": float(v) for k, v in summary["metrics"].get(split, {}).items()
                        if k in keep})
    metrics.update({f"dev.{k}": float(v) for k, v in dev_m.items() if k in keep})
    parser = {k: float(v) for k, v in (summary.get("parser") or {}).items() if k in PARSER_GUARDRAILS}
    return {"trained_at": _now(), "git_sha": git["sha"], "git_branch": git["branch"],
            "code_fingerprint": fingerprint, "config_hash": config_hash(cfg),
            "seed": int(cfg["seed"]), "metrics": metrics, "parser": parser,
            "output_dir": summary["output_dir"], "mlflow_run_id": summary.get("mlflow_run_id")}


# --------------------------------------------------------------------------- gate
def gate(champion: dict, candidate: dict, min_gain: float = DEFAULT_MIN_GAIN) -> tuple[bool, str]:
    """Dev top-1 must gain >= min_gain; guardrail metrics must hold. Returns (passed, report)."""
    cm, pm = champion["metrics"], candidate["metrics"]
    rows: list[tuple[str, float | None, float | None, float | None, bool]] = []
    passed = True

    def get(d: dict, key: str) -> float | None:
        return float(d[key]) if key in d else None

    gain = (get(pm, GATE_METRIC) or 0.0) - (get(cm, GATE_METRIC) or 0.0)
    ok = get(pm, GATE_METRIC) is not None and gain >= min_gain
    rows.append((GATE_METRIC, get(cm, GATE_METRIC), get(pm, GATE_METRIC), gain, ok))
    passed &= ok
    for key, tol in GUARDRAILS.items():
        c, p = get(cm, key), get(pm, key)
        if c is None or p is None:
            rows.append((key, c, p, None, False))
            passed = False
            continue
        delta = p - c
        ok = delta >= -tol if tol > 0 else delta <= -tol
        rows.append((key, c, p, delta, ok))
        passed &= ok
    cp, pp = champion.get("parser") or {}, candidate.get("parser") or {}
    for key in PARSER_GUARDRAILS:
        c, p = get(cp, key), get(pp, key)
        if c is None or p is None:
            rows.append((key, c, p, None, False))
            passed = False
            continue
        ok = p >= c - PARSER_GUARDRAILS[key]
        rows.append((key, c, p, p - c, ok))
        passed &= ok
    return passed, format_gate_table(rows, passed, min_gain)


def format_gate_table(rows: list[tuple[str, float | None, float | None, float | None, bool]],
                      passed: bool, min_gain: float) -> str:
    lines = [f"gate vs champion: {GATE_METRIC} must gain >= {min_gain:.3f}; "
             f"guardrail tolerance {GUARD_TOL:.3f} (ECE inverted)",
             f"{'metric':<28}{'champion':>10}{'candidate':>11}{'delta':>10}  ok"]
    for key, c, p, d, ok in rows:
        cs = f"{c:.4f}" if c is not None else "  n/a "
        ps = f"{p:.4f}" if p is not None else "  n/a "
        ds = f"{d:+.4f}" if d is not None else "     -"
        lines.append(f"{key:<28}{cs:>10}{ps:>11}{ds:>10}  {'PASS' if ok else 'FAIL'}")
    lines.append(f"RESULT: {'PASSED' if passed else 'FAILED'}")
    return "\n".join(lines)


def comparison_png(champion: dict, candidate: dict, path: Path) -> Path | None:
    """Champion-vs-candidate bar chart over the gate metrics (used in the beta PR)."""
    keys = [k for k in REPORT_KEYS if k in champion.get("metrics", {}) and k in candidate["metrics"]]
    if not keys:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ys = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ys - w / 2, [champion["metrics"][k] for k in keys], w, label="champion",
            color="#1f77b4")
    ax.barh(ys + w / 2, [candidate["metrics"][k] for k in keys], w, label="candidate",
            color="#ff7f0e")
    ax.set_yticks(ys, keys, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("score (top-1 / top-5 / macro-F1 / ECE)")
    ax.set_title(f"champion ({champion.get('weights_version', '?')}) vs candidate "
                 f"(seed {candidate.get('seed')}, {candidate.get('git_sha')})")
    ax.legend(loc="lower right", fontsize=8)
    for i, k in enumerate(keys):
        ax.annotate(f"{champion['metrics'][k]:.3f}", (champion["metrics"][k], i - w / 2),
                    ha="left", va="center", fontsize=7, color="#1f77b4")
        ax.annotate(f"{candidate['metrics'][k]:.3f}", (candidate["metrics"][k], i + w / 2),
                    ha="left", va="center", fontsize=7, color="#ff7f0e")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def failure_report(candidate: dict) -> str:
    """Where to look next after a failed gate: the candidate's worst confusions."""
    lines = ["", "failure report - guide for the next change:"]
    conf = Path(candidate["output_dir"]) / "reports" / "confusions_test_synth.csv"
    if conf.is_file():
        import csv

        with conf.open(encoding="utf-8") as fh:
            pairs = [row for row in csv.reader(fh)][1:11]
        if pairs:
            lines.append("  top confusions (synthetic test split): "
                         + "; ".join(f"{r[0]}->{r[1]} x{r[2]}" for r in pairs))
    lines += ["  - add hard-negative phrases for the confused siblings (resources/, never wording",
              "    copied from the dev or realistic sets - AGENTS.md sections 2 and 6)",
              "  - compare runs and curves in `mlflow ui` (experiment 'ttucs')",
              "  - try a bounded sweep: python scripts/autotrain.py --train --sweep "
              "train.epochs=8,15",
              "  - re-run after your change: python scripts/autotrain.py --train"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- champion seed
def _model_version(directory: Path | None) -> str | None:
    if directory is None or not (directory / "config.json").is_file():
        return None
    return json.loads((directory / "config.json").read_text(encoding="utf-8")).get("model_version")


def _resolve_published_dir(spec: Any) -> Path | None:
    """Find the exact published weights: installed package, cache, or a verified download
    (never when offline). ``None`` means --init-champion falls back to published numbers."""
    from ttucs import hub

    for candidate in (hub.packaged_model_dir("bow"), hub.cached_model_dir("bow")):
        if _model_version(candidate) == spec.version:
            return candidate
    if hub.is_offline():
        print("offline: cannot fetch the published weights - using the published numbers")
        return None
    print(f"downloading {spec.distribution} {spec.version} for the live champion evaluation")
    try:
        return hub.download_model("bow")
    except (hub.DownloadError, hub.ChecksumError) as exc:
        print(f"warning: {exc} - using the published numbers instead")
        return None


def init_champion(model_dir: Path | None = None) -> dict:
    """Seed configs/champion.json from the published weights (live evaluation when the exact
    published version can be resolved; the README/MODEL_CARD numbers otherwise).

    ``model_dir`` forces the weights to evaluate (e.g. an extracted ``ttucs-model-bow`` wheel,
    useful when the GitHub Release asset is unreachable)."""
    from ttucs import hub
    from ttucs.config import load_config
    from ttucs.datasets.build import read_jsonl
    from ttucs.evaluate import evaluate_split, parser_field_accuracy
    from ttucs.pipeline import TTUCS
    from ttucs.provenance import code_fingerprint, git_info
    from ttucs.train import load_or_build

    cfg = load_config(AUTOTRAIN_CFG_PATH)
    spec = hub.MODELS["bow"]
    metrics, parser_m = dict(PUBLISHED_METRICS), dict(PUBLISHED_PARSER)
    note = f"published numbers (README/MODEL_CARD) for {spec.distribution} {spec.version}"
    resolved = model_dir or _resolve_published_dir(spec)
    if resolved is not None:
        if _model_version(resolved) != spec.version:
            raise ValueError(f"{resolved} holds model_version {_model_version(resolved)}, "
                             f"hub pins {spec.version}")
        hub.verify_manifest(resolved)
        splits = load_or_build(cfg)
        pipe = TTUCS.from_pretrained(resolved)
        metrics = {}
        keep = ("top1", "top5", "macro_f1", "ece")
        for split in ("test_synth", "test_slates_synth"):
            m, _ = evaluate_split(pipe, splits[split], parse=split == "test_slates_synth")
            metrics.update({f"{split}.{k}": float(v) for k, v in m.items() if k in keep})
        dev_m, _ = evaluate_split(pipe, read_jsonl(DEV_PATH), parse=False)
        metrics.update({f"dev.{k}": float(v) for k, v in dev_m.items() if k in keep})
        parser_m = {k: float(v) for k, v in
                    parser_field_accuracy(splits["test_slates_synth"], pipe.parser).items()
                    if k in PARSER_GUARDRAILS}
        note = f"live evaluation of the resolved {spec.distribution} {spec.version} weights"

    leak = leakage_status()
    record = {"weights_version": spec.version, "note": note, "evaluated_at": _now(),
              "git_sha": git_info()["sha"], "code_fingerprint": code_fingerprint(),
              "config_hash": config_hash(cfg), "metrics": metrics, "parser": parser_m,
              "leakage_baseline": {k: leak[k] for k in
                                   ("phrase", "cross", "soft", "phrase_details", "cross_details")}}
    save_champion(record)
    print(f"wrote {CHAMPION_PATH}: weights {record['weights_version']} ({note})")
    for k in REPORT_KEYS:
        print(f"  {k:<28}{metrics.get(k, float('nan')):.4f}")
    return record


# --------------------------------------------------------------------------- sweep
def parse_sweep(specs: list[str]) -> list[dict[str, Any]]:
    """``train.epochs=8,15 model.dropout=0.1,0.3`` -> the cartesian product as override dicts."""
    per_key: list[list[tuple[str, Any]]] = []
    for spec in specs:
        key, _, values = spec.partition("=")
        if not values:
            raise ValueError(f"bad sweep spec {spec!r}: expected KEY=v1,v2")
        per_key.append([(key.strip(), yaml.safe_load(v)) for v in values.split(",")])
    return [{k: v for k, v in combo} for combo in itertools.product(*per_key)]


# --------------------------------------------------------------------------- release
def _pkg_changelog_entry(version: str, summary_line: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "## Changelog\n\n"
    if marker not in text:
        raise RuntimeError(f"no '## Changelog' section in {path}")
    entry = f"### {version} - {_now()[:10]} (beta, autotrain gate)\n- {summary_line}\n\n"
    path.write_text(text.replace(marker, marker + entry, 1), encoding="utf-8")


def _changelog_entry(version: str, summary_line: str, path: Path) -> None:
    """Add the beta under [Unreleased] (appends to an existing Changed group, else adds one)."""
    text = path.read_text(encoding="utf-8")
    unreleased = text.find("## [Unreleased]")
    nxt = text.find("\n## [", unreleased + 1)
    if unreleased == -1 or nxt == -1:
        raise RuntimeError(f"no [Unreleased] section found in {path}")
    block = text[unreleased:nxt]
    bullet = f"- Beta weights `ttucs-model-bow` {version} (autotrain gate passed): {summary_line}\n"
    if "\n### Changed\n" in block:
        new_block = block.replace("\n### Changed\n", "\n### Changed\n" + bullet, 1)
    else:
        new_block = block.rstrip() + "\n\n### Changed\n" + bullet
    path.write_text(text[:unreleased] + new_block + text[nxt:], encoding="utf-8")


def bump_package_version(version: str, root: Path = ROOT) -> None:
    """Bump ttucs-model-bow's own version fields (must happen before building its wheel)."""
    from ttucs import hub

    old = hub.MODELS["bow"]
    pkg = root / "packages" / "ttucs-model-bow"
    _replace_once(pkg / "pyproject.toml", f'version = "{old.version}"', f'version = "{version}"')
    _replace_once(pkg / "src" / "ttucs_model_bow" / "__init__.py",
                  f'__version__ = "{old.version}"', f'__version__ = "{version}"')


def pin_release_files(version: str, sha256_hex: str, tarball: str, summary_line: str,
                      root: Path = ROOT) -> None:
    """Pin the built release everywhere (AGENTS.md section 8): hub.MODELS, RELEASE_SHA256,
    the dependency range and both changelogs."""
    from ttucs import hub

    old = hub.MODELS["bow"]
    pkg = root / "packages" / "ttucs-model-bow"
    hub_py = root / "src" / "ttucs" / "hub.py"
    _replace_once(hub_py, f'version="{old.version}"', f'version="{version}"')
    _replace_once(hub_py, f'sha256="{old.sha256}"', f'sha256="{sha256_hex}"')
    _replace_once(hub_py, f'archive="{old.archive}"', f'archive="{tarball}"')
    (pkg / "RELEASE_SHA256").write_text(f"{sha256_hex}  {tarball}\n", encoding="utf-8")
    _replace_once(root / "pyproject.toml",
                  f"ttucs-model-bow>={old.version},<{_upper_bound(old.version)}",
                  f"ttucs-model-bow>={version},<{_upper_bound(version)}")
    _pkg_changelog_entry(version, summary_line, pkg / "README.md")
    _changelog_entry(version, summary_line, root / "CHANGELOG.md")


# --------------------------------------------------------------------------- beta release
def champion_from_candidate(candidate: dict, weights_version: str, note: str) -> dict:
    record = dict(candidate)
    record.update({"weights_version": weights_version, "note": note, "evaluated_at": _now()})
    return record


def pr_body(champion: dict, candidate: dict, gate_report: str, version: str, tarball: str,
            digest: str) -> str:
    chart = Path(candidate["output_dir"]) / "reports" / "beta_comparison.png"
    return f"""## autotrain beta: `ttucs-model-bow` {version}

Gate passed against champion {champion.get('weights_version', '?')}
(seed {candidate.get('seed')}, {candidate.get('trained_at')}).

```
{gate_report}
```

- weights tarball `{tarball}` (sha256 `{digest}`), attached to the `model-bow-v{version}` Release
- comparison chart: `{chart}` (local artefact; `models/` is not committed)
- MLflow run `{candidate.get('mlflow_run_id')}` in experiment `ttucs` (see `mlflow ui`)

### Testing the beta
After merging, `ttucs download` (or a fresh `pip install -e .`) resolves the beta weights, because
`hub.MODELS` now pins {version}. PyPI publishing stays a manual `v*` tag (docs/RELEASING.md).

### Before promoting to a final release
- score the 80 held-out realistic slates once, manually:
  `ttucs eval -m {candidate['output_dir']}` (the loop never sees them)
- update the README / MODEL_CARD results tables with those numbers (this PR does not touch them)
- re-seed the champion: `python scripts/autotrain.py --init-champion`
"""


def beta_release(champion: dict, candidate: dict, gate_report: str,
                 dry_run: bool = False) -> int:
    """Bump a beta version, build the weights, upload the Release asset, open a PR.

    Never merges, never pushes ``main``, never publishes to PyPI (AGENTS.md rules 1 and 9).
    """
    from ttucs import hub
    from ttucs.hub import sha256_file

    old = hub.MODELS["bow"]
    version = next_beta_version(old.version)
    tarball_name = f"model-bow-v{version}.tar.gz"
    dev_n = sum(1 for line in DEV_PATH.open(encoding="utf-8") if line.strip())
    summary_line = (f"dev top-1 {champion['metrics']['dev.top1']:.3f} -> "
                    f"{candidate['metrics']['dev.top1']:.3f} (n={dev_n}), dev macro-F1 "
                    f"{champion['metrics']['dev.macro_f1']:.3f} -> "
                    f"{candidate['metrics']['dev.macro_f1']:.3f}, "
                    f"synthetic top-1 {champion['metrics']['test_synth.top1']:.3f} -> "
                    f"{candidate['metrics']['test_synth.top1']:.3f}")
    print(f"release: preparing beta ttucs-model-bow {version} ({summary_line})")

    bump_package_version(version)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    print(f"release: building the tarball + wheel into {dist}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_model_package.py"),
                    "--model-dir", candidate["output_dir"], "--tarball", "--wheel",
                    "--outdir", str(dist)], check=True)
    tb = dist / tarball_name
    digest = sha256_file(tb)
    pin_release_files(version, digest, tarball_name, summary_line)
    save_champion(champion_from_candidate(candidate, version, "beta (autotrain gate passed)"))
    print(f"release: pinned sha256 {digest}; refreshed {CHAMPION_PATH.name}")

    pinned = [PKG_DIR / "pyproject.toml", PKG_DIR / "src" / "ttucs_model_bow" / "__init__.py",
              PKG_DIR / "RELEASE_SHA256", PKG_DIR / "README.md", ROOT / "src" / "ttucs" / "hub.py",
              ROOT / "pyproject.toml", ROOT / "CHANGELOG.md", CHAMPION_PATH]
    if dry_run:
        print("release: DRY RUN - nothing committed, uploaded or opened. Files to commit:")
        for p in pinned:
            print(f"  {p.relative_to(ROOT)}")
        print(f"then: gh release create model-bow-v{version} {tb} --target <branch>, open a PR")
        return 0

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    body_path = dist / f"autotrain-pr-{version}.md"
    body_path.write_text(pr_body(champion, candidate, gate_report, version, tarball_name, digest),
                         encoding="utf-8")
    print(f"release: committing the pinned files on {branch}")
    _run(["git", "add", *[str(p) for p in pinned]])
    _run(["git", "add", "-u", "src/ttucs", "configs", "packages"])
    _run(["git", "commit", "-m", f"feat: autotrain beta ttucs-model-bow {version}"])
    print(f"release: pushing {branch}")
    _run(["git", "push", "-u", "origin", branch])
    print(f"release: creating GitHub Release model-bow-v{version} with asset {tarball_name}")
    _run(["gh", "release", "create", f"model-bow-v{version}", str(tb), "--target", branch,
          "--title", f"ttucs-model-bow {version} beta weights",
          "--notes", f"Beta weights from the autotrain loop (gate passed). {summary_line}"])
    print("release: opening the beta PR")
    _run(["gh", "pr", "create", "--base", "main", "--head", branch,
          "--title", f"Beta weights ttucs-model-bow {version} (autotrain)",
          "--body-file", str(body_path)])
    print("release: done - PR opened; the owner reviews, scores the realistic slates once, "
          "merges and decides whether to promote to a final release")
    return 0


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=AUTOTRAIN_CFG_PATH)
    ap.add_argument("--init-champion", action="store_true",
                    help="seed configs/champion.json from the published weights")
    ap.add_argument("--model-dir", type=Path,
                    help="with --init-champion: evaluate these weights (e.g. an extracted "
                         "ttucs-model-bow wheel) instead of resolving the published ones")
    ap.add_argument("--train", action="store_true", help="train a candidate round")
    ap.add_argument("--confirm-seed", action="store_true",
                    help="also train the winning setup with seed 7; both must pass the gate")
    ap.add_argument("--sweep", nargs="*", default=[], metavar="KEY=V1,V2",
                    help="bounded sweep over dotted config overrides, e.g. train.epochs=8,15")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                    help=f"train at most N rounds per invocation (default {MAX_ROUNDS})")
    ap.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN,
                    help=f"required dev top-1 gain (default {DEFAULT_MIN_GAIN})")
    ap.add_argument("--release", action="store_true",
                    help="on a passing gate: bump a beta, build, upload the Release asset, open a PR")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --release: local work only, skip git/gh/network")
    args = ap.parse_args(argv)

    if args.init_champion:
        init_champion(args.model_dir)
        return 0

    champion = load_champion()
    if champion is None:
        print(f"no {CHAMPION_PATH} - run `python scripts/autotrain.py --init-champion` first")
        return 2
    from ttucs.config import load_config
    from ttucs.provenance import changed_vs, code_fingerprint, git_info

    fingerprint = code_fingerprint()
    git = git_info()
    if not changed_vs(champion):
        print(f"nothing to do: code fingerprint {fingerprint} matches champion "
              f"(weights {champion['weights_version']}, seeded {champion['evaluated_at']})")
        return 0
    print(f"change detected: fingerprint {champion['code_fingerprint']} -> {fingerprint} "
          f"(branch {git['branch']}, {git['sha']}, dirty={git['dirty']})")

    leak = leakage_gate(champion.get("leakage_baseline") or {})
    print(leak["report"])
    if not leak["passed"]:
        return 1
    if not args.train:
        print("dry run - rerun with --train to train a candidate (see --sweep, --confirm-seed, --release)")
        return 0

    cfg = load_config(args.config)
    rounds = parse_sweep(args.sweep) or [{}]
    rounds = rounds[:max(1, args.max_rounds - (1 if args.confirm_seed else 0))]
    attempts: list[tuple[dict, dict]] = []
    for i, overrides in enumerate(rounds, 1):
        cfg_i = (load_config(args.config, [f"{k}={v}" for k, v in overrides.items()])
                 if overrides else cfg)
        print(f"round {i}/{len(rounds)}: {overrides or 'configs/autotrain.yaml as-is'}")
        summary = train_candidate(cfg_i, "autotrain" if len(rounds) == 1 else f"autotrain-r{i}")
        record = candidate_record(cfg_i, summary, dev_metrics(summary["output_dir"]), git, fingerprint)
        attempts.append((overrides, record))
    best_overrides, best = max(attempts, key=lambda a: a[1]["metrics"][GATE_METRIC])
    if len(attempts) > 1:
        for _, rec in sorted(attempts, key=lambda a: -a[1]["metrics"][GATE_METRIC]):
            print(f"  dev top1 {rec['metrics'][GATE_METRIC]:.4f}  run {rec['mlflow_run_id']}  "
                  f"config {rec['config_hash']}")
        print(f"best round: dev top1 {best['metrics'][GATE_METRIC]:.4f}")

    passed, report = gate(champion, best, args.min_gain)
    print(report)
    comparison_png(champion, best, Path(best["output_dir"]) / "reports" / "beta_comparison.png")

    if not passed:
        print(failure_report(best))
        out = Path(best["output_dir"]) / "reports" / "autotrain_failure.txt"
        out.write_text(report + "\n" + failure_report(best) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        return 1

    if args.confirm_seed:
        print(f"confirm-seed: re-training the winning setup with seed {DEFAULT_SEED}")
        cfg2 = load_config(args.config, [f"{k}={v}" for k, v in best_overrides.items()]
                           + [f"seed={DEFAULT_SEED}"])
        summary2 = train_candidate(cfg2, "autotrain-confirm")
        cand2 = candidate_record(cfg2, summary2, dev_metrics(summary2["output_dir"]), git, fingerprint)
        passed2, report2 = gate(champion, cand2, args.min_gain)
        print(report2)
        comparison_png(champion, cand2, Path(cand2["output_dir"]) / "reports" / "beta_comparison.png")
        if not passed2:
            print(f"confirm-seed FAILED (seed {DEFAULT_SEED}) - treat the gain as noise (AGENTS.md 6)")
            print(failure_report(best))
            return 1

    if args.release:
        return beta_release(champion, best, report, dry_run=args.dry_run)
    print("gate PASSED - rerun with --release to bump a beta version and open a PR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
