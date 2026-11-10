# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `docs/about/`: a 20-document guided tour of the repository for new contributors, from what UCS is
  and how a slate is parsed through bag-of-words features, every layer of `BowNet`, calibration,
  synthetic data, splits, training, evaluation, the two-package distribution and the release
  process. `docs/about/about_1_index.md` is the entry point.
- `ttucs serve`: the engine over loopback HTTP, so other tools can share one loaded model instead
  of paying a process start per call. Routes are `GET /health`, `POST /predict`, `POST /parse`,
  `GET /taxonomy/search`, `GET /taxonomy/{cat_id}` and `POST /shutdown`; every response carries
  `schema_version`. It binds `127.0.0.1` only, requires an `X-Ttucs-Token` header on everything
  except `/health`, sends no CORS headers, caps bodies at 1 MiB and 512 texts, and serialises
  inference behind one lock. The endpoint file (`$TTUCS_HOME/serve.json`, mode `0600`) carries the
  host, port, token and pid and is removed on clean shutdown. `--no-token` exists for tests and
  logs a warning. See `docs/serve-api.md` for the contract, including the launcher steps a client
  follows. No new dependencies and no network access of its own.
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
- Module-level shortcuts for a one-line start: `import ttucs as t; t.predict(...)` (and
  `t.predict_batch(...)`) load the default pretrained model lazily on first use - offline, and
  nothing loads at import time - so scripts no longer need an explicit `TTUCS.from_pretrained()`.
  The class API is unchanged for custom models or inference overrides.
- Slate parser: Sennheiser MKH models are now recognised when speech-to-text drops the "h"
  ("m k eighty forty" -> `Sennheiser MKH 8040`; 1-2 digit capsule numbers stay Schoeps), and the
  A-B technique spoken as "dual a b" / "a b stereo" parses as `Spaced Pair (AB)`.
- Cue-less slates now join all content-bearing leftover clauses into the description instead of
  keeping only the longest one, so multi-part ambience slates ("nearby traffic, crossing sign,
  children playing in a nearby park, wind") are classified in full. Perspective-only words
  (exterior/interior/...) are captured as perspective and kept out of the description.
- Opt-in description-generator styles for experimentation (default off, off in the release config):
  `data.synthetic.verbose_prob` emits longer action-narration sentences for sources and
  first-person scene observations for ambiences; `data.synthetic.multi_clause_prob` joins
  same-class description pairs into multi-clause training text.
- Opt-in clause weighting, `model.head_repeat` (default 1 = off): with a value above 1,
  comma-clauses are featurised independently and the first clause - the label driver, which the
  parser guarantees comes first - has its features repeated into the embedding-bag mean while
  context clauses count once. Off keeps featurisation bit-identical to the published model.

### Changed
- **Breaking:** `ttucs predict --json` now always prints a JSON list, one element per input. It
  previously printed a bare object for a single text and a list for several, which made every
  script branch on the input count. Consumers that indexed the single-object form need `[0]`.
  `ttucs serve` follows the same rule.
- `docs/assets/distribution.md` and its plots are regenerated from `configs/bow-release.yaml`, so the
  report describes the release training split (159,591 synthetic + 3,305 UCS label documents =
  162,896 examples, 179-260 per CatID) instead of the older `configs/bow.yaml` build. `make report`
  now uses the release config for the same reason.
- **Multi-clause descriptions** (parser behaviour ships with `ttucs`; retrained weights with the
  new data would be a separate `ttucs-model-bow` release): the parser joins the content clauses
  that follow a description cue
  ("recording traffic passing, children playing, wind" keeps all three; the same for cue-less
  slates), and the release config trains on `data.synthetic.multi_clause_prob: 0.2` so training
  matches what the parser feeds the model. Retrained on it, the multi-clause dev slice improves
  0.600 -> 0.673 (n=55), synthetic test 0.909 -> 0.915 (22,384 -> 20,259 examples after the
  generator change), realistic category accuracy 0.875 -> 0.900; held-out realistic top-1 is
  0.675 vs the published 0.713 (n=80, within noise - sibling-class confusions dominate).
  Synthetic test sets change with the generator and are not comparable across versions.
- `configs/autotrain.yaml` sets `mlflow.log_model: false`: every logged model copies the ~63 MB
  weights into the artifact store, so exploratory rounds now log metrics and reports only. Turn it
  back on for a gated winner that should enter the registry.
- `configs/bow-release.yaml` enables MLflow tracking for release training runs
  (`mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse them).
- Experiment retrain rounds (2026-11-08/09) with the new knobs did not beat the multi-clause
  configuration: `verbose_prob: 0.3` cost multi-clause dev accuracy (0.673 -> 0.582, n=55) and
  `head_repeat: 2` netted flat, so the release config keeps `verbose_prob: 0.0` and
  `head_repeat: 1`. Two controlled rounds changed held-out top-1 by at most one slate (n=80).

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
