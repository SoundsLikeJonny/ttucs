# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Module-level shortcuts for a one-line start: `import ttucs as t; t.predict(...)` (and
  `t.predict_batch(...)`) load the default pretrained model lazily on first use - offline, and
  nothing loads at import time - so scripts no longer need an explicit `TTUCS.from_pretrained()`.
  The class API is unchanged for custom models or inference overrides.
- The slate parser now recognises Sennheiser MKH models when speech-to-text drops the "h"
  ("m k eighty forty" -> `Sennheiser MKH 8040`; 1-2 digit capsule numbers stay Schoeps), and the
  A-B technique spoken as "dual a b" / "a b stereo" (`Spaced Pair (AB)`).
- Cue-less slates now join all content-bearing leftover clauses into the description instead of
  keeping only the longest one, so multi-part ambience slates ("nearby traffic, crossing sign,
  children playing in a nearby park, wind") are classified in full. Perspective-only words
  (exterior/interior/...) are captured as perspective and kept out of the description.

### Changed
- Descriptions are now multi-clause: the parser joins the content clauses that follow a
  description cue ("recording traffic passing, children playing, wind" keeps all three; the same
  for cue-less slates), and `data.synthetic.multi_clause_prob` (0.2 in the release config) joins
  same-class description pairs so training matches what the parser feeds the model. Multi-clause
  dev texts scored 0.600 vs 0.816 single-clause before this change. Synthetic test sets change
  and are not comparable across versions.

### Added
- `MLFLOW_TRACKING_URI` now overrides `mlflow.tracking_uri` from the config, so runs can be sent to
  a remote tracking server without editing a config file. Offline mode (`TTUCS_OFFLINE=1` or
  `HF_HUB_OFFLINE=1`) ignores it and keeps the local SQLite store. The resolved URI is logged.
- `requirements.txt`: a pinned lock of the `[dev]` dependencies (Python 3.10+, Windows/macOS/Linux,
  CPU-only PyTorch) for reproducible development environments, generated with `uv pip compile`.
- `scripts/autotrain.py`, `configs/autotrain.yaml` and `configs/champion.json`: an MLflow-gated
  auto-train loop. It fingerprints the training code and resources, blocks rounds with new
  leakage, trains a candidate on the dev descriptions and synthetic splits only (the held-out
  realistic slates are never loaded), and gates it against the champion: dev top-1 must gain at
  least 0.015 and guardrails (dev top-5/macro-F1/ECE, synthetic top-1/top-5/macro-F1, parser
  exact/token-F1) must hold. With `--release` a passing candidate becomes a beta PR (pre-release
  version bump, verified Release asset, pinned sha, refreshed champion) - it never merges, never
  pushes `main` and never publishes to PyPI. `make autotrain` runs one round.
- MLflow training runs are now tagged with the git SHA/branch/dirty flag, a code fingerprint and
  the data fingerprint, and write a `training_history.png` (train loss + val top-1 per epoch).
- `scripts/check_leakage.py`: automates the leakage checks (bank phrases of 3+ words appearing
  verbatim in the hand-written sets; dev/realistic 4-gram or substring overlap); exits 1 on
  violations. Run it after editing `catid_phrases.yaml`, `synth_banks.yaml` or the dev set.

### Changed
- `configs/autotrain.yaml` sets `mlflow.log_model: false`: every logged model copies the ~63 MB
  weights into the artifact store, so exploratory rounds now log metrics and reports only. Turn it
  back on for a gated winner that should enter the registry.

### Fixed
- README install line now states the real weights size: the `ttucs-model-bow` 0.2.0 wheel is ~63 MB
  (fp16 safetensors ~68 MB), not the 0.1.1-era "~32 MB".
- CONTRIBUTING dev setup now covers Windows PowerShell 5.1 as well as bash, and no longer claims that
  weight-dependent tests skip: no test needs the real weights (they train tiny models in-test and
  bypass the installed package).
- Removed the stale "MOVE this file" header comment from `.github/workflows/publish.yml`.

## [0.3.0] - 2026-11-07

### Added
- `src/ttucs/resources/catid_phrases.yaml`: hand-written everyday phrases for 494 CatIDs, used by the
  synthetic generator with probability `synthetic.phrase_prob` (default 0.5).
- Object routing in `DescriptionGenerator`: objects named in a specific subcategory's Explanation or
  Synonyms go to that subcategory, and MISC/GENERAL classes only get the leftover objects.
- `examples/dev_descriptions.jsonl`: 332 hand-written descriptions for choosing settings, so the
  realistic slates stay a held-out test set.
- `ttucs eval --predictions` writes predictions for the first `--data` file when no realistic slates
  are evaluated.

### Changed
- Default weights are now `ttucs-model-bow` 0.2.0 (131,072 x 256, ~68 MB, 300 examples per CatID).
  On the 80 hand-written realistic slates, top-1 goes from 0.488 to 0.713 and top-5 from 0.813 to
  0.925; on the dev set, top-1 goes from 0.491 to 0.789. The dependency range is now
  `ttucs-model-bow>=0.2.0,<0.3`.

### Fixed
- Synthetic explanation fragments no longer include cross-references to other classes ("see ...",
  "not ...").

## [0.2.0] - 2026-11-06

### Added
- Pretrained weights ship as the data-only `ttucs-model-bow` 0.1.1 distribution (fp16 safetensors, SHA-256
  manifest, 131,072 x 128 compact model from `configs/bow-release.yaml`). It is a default dependency,
  so `pip install ttucs` works offline right after install. It is versioned separately from `ttucs`.
- `ttucs.hub`: offline-first model resolution (explicit path, installed package, local cache in
  `TTUCS_HOME` or the platform cache dir, then a SHA-256-verified GitHub Release download).
  `TTUCS_OFFLINE=1` or `HF_HUB_OFFLINE=1` disables all network access.
- CLI `ttucs info [--verify]` and `ttucs download [--url --sha256 --force]`.
- Model loading from `weights.safetensors` (fp16 or fp32) as well as `weights.pt`; training records
  its data sources in `config.json`.
- `scripts/build_model_package.py` (reproducible weights tarball and wheel), trusted-publishing
  workflow `.github/workflows/publish.yml`, and `docs/RELEASING.md`.

### Changed
- CI workflows moved from `ci/` to `.github/workflows/` (`ci.yml`, `publish.yml`), so GitHub runs them.
- Lean core install: scikit-learn, pandas, matplotlib and MLflow moved to the `[train]` extra.
  `train`, `eval` and `report` print an install hint without it.
- `TTUCS.from_pretrained()` with no arguments now loads the installed `ttucs-model-bow` weights.
- The default model is now the compact model trained on redistributable data only. Accuracy on the 80
  hand-written evaluation slates drops from 0.713 to 0.488 top-1 (0.913 to 0.813 top-5); synthetic
  test sets stay at about 0.86-0.87 top-1.
- README rewritten for PyPI with absolute links; model and data cards updated for the published model.

### Removed
- **Breaking:** third-party sound library metadata ingestion and its CLI fetch command. The published
  weights are trained only on synthetic data and the UCS list.

### Fixed
- License metadata uses an SPDX expression (`license = "MIT"`, `license-files`) instead of the
  deprecated TOML table and license classifier.

## [0.1.0] - 2026-11-06

### Added
- UCS v8.2.1 list (753 CatIDs, 82 categories) shipped as package data with CatID/alias lookup,
  UCS filename building and parsing.
- Rule-based speech-to-text slate parser (project, location, creator, date, microphones, mic setup,
  recorder, sample rate, bit depth, take/scene/slate, perspective, notes, description).
- LLM-free balanced synthetic description and slate generators; ingestion of third-party sound
  library metadata (local use only); group-based deterministic train/val/test splits.
- Models: hashed n-gram bag-of-embeddings (default, CPU), frozen MiniLM encoder + head, zero-shot
  label-document similarity.
- `TTUCS` Python API: top-k, hierarchical category scores, temperature calibration, unknown /
  category-MISC fallback, UCS filename + FXName + keyword suggestions, JSON serialisation.
- MLflow tracking: params, per-epoch and per-split metrics, per-class reports, confusion artifacts,
  pyfunc model with signature, model registry.
- CLI `ttucs generate-data|train|eval|predict|parse|report` plus a third-party metadata fetch command,
  YAML configs, tests, CI, pre-commit, Makefile, Dockerfile, model and data cards.

### Removed
- Legacy prototype (`main_web.py`, notebook, `.idea/`, `requirements.txt`).

[Unreleased]: https://github.com/SoundsLikeJonny/ttucs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/SoundsLikeJonny/ttucs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/SoundsLikeJonny/ttucs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/SoundsLikeJonny/ttucs/releases/tag/v0.1.0
