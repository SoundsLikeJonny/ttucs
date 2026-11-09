# The repository map

Every path below exists in the repository at the time of writing. The "explained in" column points
at the document in this folder that covers it in depth.

## Top level

| path | what it is | explained in |
|---|---|---|
| `README.md` | pitch, install, quick start, results table, architecture diagram | [about_what-is-ttucs.md](about_what-is-ttucs.md) |
| `AGENTS.md` | the rulebook: hard rules, conventions, release process, current state | [about_development-and-release.md](about_development-and-release.md) |
| `MODEL_CARD.md` | what the weights are, how they were trained, how well they score | [about_evaluation.md](about_evaluation.md) |
| `DATA_CARD.md` | what data was used, licences, leakage policy | [about_synthetic-data.md](about_synthetic-data.md) |
| `CONTRIBUTING.md` | dev setup, style, PR expectations | [about_development-and-release.md](about_development-and-release.md) |
| `CHANGELOG.md` | Keep a Changelog history, newest first, `## [Unreleased]` at the top | [about_development-and-release.md](about_development-and-release.md) |
| `pyproject.toml` | packaging, dependencies, extras, ruff and pytest config | [about_distribution-and-hub.md](about_distribution-and-hub.md) |
| `Makefile` | shortcuts: `lint`, `test`, `smoke`, `release-model`, `autotrain`, ... | [about_development-and-release.md](about_development-and-release.md) |
| `requirements.txt` | pinned lock of the `[dev]` dependencies | [about_development-and-release.md](about_development-and-release.md) |
| `LICENSE` | MIT | — |

## `src/ttucs/` — the package

| path | what it is | explained in |
|---|---|---|
| `__init__.py` | lazy module API: `import ttucs as t; t.predict(...)` | [about_pipeline-and-outputs.md](about_pipeline-and-outputs.md) |
| `_version.py` | the single source of the `ttucs` version | [about_development-and-release.md](about_development-and-release.md) |
| `_extras.py` | clear errors when `[train]` / `[mlflow]` extras are missing | [about_training-and-config.md](about_training-and-config.md) |
| `config.py` | `DEFAULTS`, YAML merge, dotted overrides, `flatten` | [about_training-and-config.md](about_training-and-config.md) |
| `taxonomy.py` | UCS list loading, lookup, aliases, `UCSFilename` build/parse | [about_ucs-taxonomy.md](about_ucs-taxonomy.md) |
| `text.py` | `preprocess`, `make_fx_name`, `make_keywords`, `initials`, `abbreviate` | [about_text-features.md](about_text-features.md), [about_pipeline-and-outputs.md](about_pipeline-and-outputs.md) |
| `slate/__init__.py` | re-exports `SlateFields`, `SlateParser`, `parse_slate`, `normalize_transcript` | [about_slate-parser.md](about_slate-parser.md) |
| `slate/normalize.py` | STT cleanup: fillers, numbers, homophones, spelled letters | [about_normalisation.md](about_normalisation.md) |
| `slate/parser.py` | the three-pass slate parser and `SlateFields` | [about_slate-parser.md](about_slate-parser.md) |
| `slate/gear.py` | the gear lexicon (`MICS`, `RECORDERS`, `SETUPS`, `MIC_TYPES`, `ALL_GEAR`) | [about_gear-lexicon.md](about_gear-lexicon.md) |
| `models/__init__.py` | imports the model modules so they register themselves | [about_model-architecture.md](about_model-architecture.md) |
| `models/base.py` | `UCSModel` interface, `MODEL_REGISTRY`, `train_loop`, `fit_temperature` | [about_training-loop.md](about_training-loop.md) |
| `models/bow.py` | `HashedNgramFeaturizer`, `BowNet`, `BowClassifier` — the published model | [about_model-architecture.md](about_model-architecture.md) |
| `models/zeroshot.py` | TF-IDF similarity to UCS label documents (no training) | [about_model-architecture.md](about_model-architecture.md) |
| `models/encoder.py` | optional frozen MiniLM + MLP head (not trained or verified) | [about_model-architecture.md](about_model-architecture.md) |
| `pipeline.py` | `TTUCS`, `Prediction`, `Candidate`, decision rule, filename | [about_pipeline-and-outputs.md](about_pipeline-and-outputs.md) |
| `hub.py` | offline-first model resolution, download, manifest, fp16 export | [about_distribution-and-hub.md](about_distribution-and-hub.md) |
| `synth/generator.py` | `DescriptionGenerator`, `SlateGenerator`, `Example`, `load_banks` | [about_synthetic-data.md](about_synthetic-data.md) |
| `datasets/build.py` | split assembly, label documents, augmentation, JSONL cache | [about_datasets-and-splits.md](about_datasets-and-splits.md) |
| `train.py` | the training entry point: data, fit, calibrate, evaluate, log | [about_training-and-config.md](about_training-and-config.md) |
| `evaluate.py` | metrics, ECE, parser field accuracy, reports | [about_evaluation.md](about_evaluation.md) |
| `reporting.py` | class-distribution report and training-history plot | [about_training-and-config.md](about_training-and-config.md) |
| `provenance.py` | git state, code fingerprint, run tags | [about_training-and-config.md](about_training-and-config.md) |
| `mlflow_model.py` | the MLflow `pyfunc` flavour | [about_training-and-config.md](about_training-and-config.md) |
| `cli.py` | the `ttucs` command: `predict`, `parse`, `info`, `download`, `generate-data`, `train`, `eval`, `report` | [about_pipeline-and-outputs.md](about_pipeline-and-outputs.md) |
| `resources/ucs_v8.2.1.csv` | the UCS list (753 rows) | [about_ucs-taxonomy.md](about_ucs-taxonomy.md) |
| `resources/ucs_v8.2.1_top_level.csv` | top-level category explanations | [about_ucs-taxonomy.md](about_ucs-taxonomy.md) |
| `resources/synth_banks.yaml` | modifiers, contexts, subcategory actions, per-category objects/actions | [about_synthetic-data.md](about_synthetic-data.md) |
| `resources/catid_phrases.yaml` | hand-written everyday head phrases for 494 CatIDs | [about_synthetic-data.md](about_synthetic-data.md) |

## `packages/ttucs-model-bow/` — the weights distribution

| path | what it is | explained in |
|---|---|---|
| `pyproject.toml` | the `ttucs-model-bow` distribution, versioned independently | [about_distribution-and-hub.md](about_distribution-and-hub.md) |
| `src/ttucs_model_bow/__init__.py` | `__version__` and the model directory location | [about_distribution-and-hub.md](about_distribution-and-hub.md) |
| `src/ttucs_model_bow/model/` | `config.json`, `weights.safetensors`, `manifest.json` (built, not committed) | [about_distribution-and-hub.md](about_distribution-and-hub.md) |
| `RELEASE_SHA256` | the sha256 of the release tarball, checked by tests and CI | [about_distribution-and-hub.md](about_distribution-and-hub.md) |
| `README.md` | short description and its own weights changelog | [about_distribution-and-hub.md](about_distribution-and-hub.md) |

## `configs/` — experiment configurations

| path | what it is | explained in |
|---|---|---|
| `bow-release.yaml` | the published weights: 131,072 buckets x 256 dims, 300 per CatID | [about_training-and-config.md](about_training-and-config.md) |
| `bow.yaml` | larger research config: 524,288 buckets x 192 dims, 150 per CatID | [about_training-and-config.md](about_training-and-config.md) |
| `smoke.yaml` | tiny config used by CI to prove training still runs | [about_development-and-release.md](about_development-and-release.md) |
| `autotrain.yaml` | the gated auto-train loop; realistic slates set to `null` | [about_evaluation.md](about_evaluation.md) |
| `champion.json` | the reference metrics and leakage baseline the gate compares against | [about_evaluation.md](about_evaluation.md) |
| `zeroshot.yaml`, `minilm.yaml` | the two non-published models | [about_model-architecture.md](about_model-architecture.md) |

## `scripts/` — development tooling

| path | what it is | explained in |
|---|---|---|
| `build_model_package.py` | export fp16 weights, deterministic tarball, wheel, size check | [about_distribution-and-hub.md](about_distribution-and-hub.md) |
| `build_taxonomy.py` | regenerate the UCS CSVs from the official spreadsheets | [about_ucs-taxonomy.md](about_ucs-taxonomy.md) |
| `check_leakage.py` | bank-phrase and 4-gram overlap checks, exit 1 on violations | [about_evaluation.md](about_evaluation.md) |
| `autotrain.py` | fingerprint, leakage gate, train, gate, optional beta PR | [about_evaluation.md](about_evaluation.md) |

## `examples/` — the hand-written evaluation sets

| path | what it is | explained in |
|---|---|---|
| `dev_descriptions.jsonl` | 332 hand-written descriptions, the tuning set | [about_evaluation.md](about_evaluation.md) |
| `realistic_slates.jsonl` | 80 hand-written full slates, the held-out test set | [about_evaluation.md](about_evaluation.md) |

## `tests/` — 113 tests

`test_parser.py` (13), `test_normalize.py` (4), `test_text.py` (5), `test_taxonomy.py` (10),
`test_synth_and_splits.py` (16), `test_model_pipeline.py` (8), `test_hub.py` (13),
`test_config.py` (4), `test_cli.py` (2), `test_leakage.py` (3), `test_autotrain.py` (17),
`test_autotrain_units.py` (6). See [about_development-and-release.md](about_development-and-release.md).

## `.github/workflows/`

| path | what it is |
|---|---|
| `ci.yml` | ruff, pytest, smoke train and packaging check on Python 3.10–3.12 |
| `publish.yml` | Trusted Publishing to TestPyPI (manual) and PyPI (tag `v*`) |

## `docs/`

| path | what it is |
|---|---|
| `RELEASING.md` | the full release runbook |
| `assets/` | generated class-distribution report and plots |
| `about/` | this folder |

## Never committed

`data/`, `models/`, `dist/`, `build/`, `mlruns/`, `mlflow.db`, `*.pt`, `*.safetensors`, tarballs,
credentials, and the agent-platform marker `.abacus.donotdelete`. Stage files explicitly; never
`git add -A` (AGENTS.md rule 3). The weights travel as a GitHub Release asset instead.

## Reading order for a new contributor

1. [about_what-is-ttucs.md](about_what-is-ttucs.md) — why the tool exists.
2. [about_ucs-taxonomy.md](about_ucs-taxonomy.md) — the label space.
3. [about_slate-anatomy.md](about_slate-anatomy.md) → [about_slate-parser.md](about_slate-parser.md)
   — what goes in and how it is split.
4. [about_bag-of-words.md](about_bag-of-words.md) → [about_model-architecture.md](about_model-architecture.md)
   — what the model is.
5. [about_pipeline-and-outputs.md](about_pipeline-and-outputs.md) — what comes out.
6. Then data, training, evaluation and distribution as needed.
