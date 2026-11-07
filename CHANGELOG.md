# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `requirements.txt`: a pinned lock of the `[dev]` dependencies (Python 3.10+, Windows/macOS/Linux,
  CPU-only PyTorch) for reproducible development environments, generated with `uv pip compile`.

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
