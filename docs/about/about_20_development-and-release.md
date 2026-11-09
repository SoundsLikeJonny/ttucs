# Development, testing and release

This document is the working manual: how to set up, what the checks are, how versions move, and the
exact release sequence. It summarises AGENTS.md sections 4, 8 and 9 and `docs/RELEASING.md`, and
points at the code each step touches.

## Setup

The owner works on Windows with PowerShell 5.1, so both forms are given.

bash:

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e packages/ttucs-model-bow -e ".[dev]"
pre-commit install
```

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e packages/ttucs-model-bow -e ".[dev]"
pre-commit install
```

The local `packages/ttucs-model-bow` has **no weights** until you build them, so in dev either pass
`-m models/...` or install the released weights. `make install` does the pip line.

## The checks

| check | command | what it covers |
|---|---|---|
| lint | `ruff check src tests scripts` | line length 110, py310 target, rules in `pyproject.toml` |
| tests | `pytest -q` | 113 tests |
| smoke train | `make smoke` | `configs/smoke.yaml`: 12 per class, 2 epochs, 65,536 x 32 |
| packaging | `python -m build --outdir dist . && twine check --strict dist/*` | the wheel is installable and metadata is valid |
| leakage | `python scripts/check_leakage.py` | bank phrases and 4-grams |
| model pins | `pytest -q tests/test_hub.py` | sha, versions and dependency range agree |

CI (`.github/workflows/ci.yml`) runs lint, tests, the smoke train and the packaging check on Python
3.10, 3.11 and 3.12 for every push to `main` and every PR. It installs CPU torch and the local
weightless model package, so tests that need real weights skip when the weights are absent.

## The tests

| file | tests | covers |
|---|---:|---|
| `test_parser.py` | 13 | cues, trimming, multi-clause, gear fragments, perspectives, dates |
| `test_normalize.py` | 4 | fillers, spelled letters, homophones, numbers |
| `test_text.py` | 5 | preprocess, FX names, keywords, initials, abbreviate |
| `test_taxonomy.py` | 10 | loading, resolve, aliases, filenames, parse errors |
| `test_synth_and_splits.py` | 16 | generator determinism, routing, grouped splits, augmentation |
| `test_model_pipeline.py` | 8 | featuriser, model save/load, pipeline decisions |
| `test_hub.py` | 13 | resolution order, offline, manifest, download, pins |
| `test_config.py` | 4 | defaults, deep merge, dotted overrides, flatten |
| `test_cli.py` | 2 | predict and parse end to end |
| `test_leakage.py` | 3 | the leakage script |
| `test_autotrain.py` | 17 | gate, champion, versions, leakage baseline |
| `test_autotrain_units.py` | 6 | helpers |

Add a test with every behaviour change. For parser or normaliser changes, use the exact transcript
that failed, not a cleaned-up version of it.

## Style

- ruff, line length 110, target py310.
- Type hints on public functions; `from __future__ import annotations` at the top of modules.
- Keep dependencies minimal: anything heavy goes in an extra (`[train]`, `[transformers]`,
  `[mlflow]`), never in the core install.
- No comments that restate the code; comments explain *why*.
- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`.
- Branches: `feat/...`, `fix/...`, `docs/...`, `chore/...`.

## Git rules

1. Work on a feature branch and open a PR to `main`.
2. Never merge a PR yourself, never push to `main`, never force-push `main`. The owner reviews and
   merges.
3. Never commit `data/`, `models/`, `dist/`, `build/`, `mlruns/`, `mlflow.db`, `*.pt`,
   `*.safetensors`, tarballs, credentials or `.abacus.donotdelete`. Stage files explicitly; never
   `git add -A` or `git add .`.
4. Never paste tokens into commits, issues, PRs, logs or chat. PyPI publishing uses Trusted
   Publishing and needs no token.
5. Do not add the owner's personal details to the repo.

## Versioning

SemVer 2.0.0, independently for the two distributions:

| distribution | single source of truth |
|---|---|
| `ttucs` | `src/ttucs/_version.py` (`pyproject.toml` reads it) |
| `ttucs-model-bow` | `packages/ttucs-model-bow/pyproject.toml` **and** `src/ttucs_model_bow/__init__.py` |

Bump the weights version **only when the weights change**. When they do, update all of these
together:

- `hub.MODELS["bow"]` — version, sha256, archive name;
- `packages/ttucs-model-bow/RELEASE_SHA256`;
- the dependency range in `pyproject.toml` (`ttucs-model-bow>=X.Y.0,<X.(Y+1)`);
- `MODEL_CARD.md` and the README results table;
- the changelog in `packages/ttucs-model-bow/README.md`.

`tests/test_hub.py` fails if they disagree.

## Changelog

`CHANGELOG.md` follows Keep a Changelog 1.1.0:

- newest first, with `## [Unreleased]` at the top;
- a line under `[Unreleased]` for every user-visible change;
- groups in the order Added, Changed, Deprecated, Removed, Fixed, Security, omitting empty ones;
- breaking changes marked **Breaking:**;
- compare links at the bottom kept current;
- accuracy changes give before/after numbers and name the test set.

At release time, rename `[Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and leave a fresh empty
`[Unreleased]`.

## Release sequence

1. **PR** with the version bumps, sha, docs and CHANGELOG. The owner merges it.
2. **If the weights changed**: train, build, upload.
   ```bash
   ttucs train -c configs/bow-release.yaml
   python scripts/build_model_package.py --model-dir models/ttucs-bow-release --tarball --wheel
   ```
   Upload `dist/model-bow-vX.Y.Z.tar.gz` as GitHub Release `model-bow-vX.Y.Z`. The tag must not
   start with `v`; do not mark it "Latest"; upload the exact file that was hashed.
3. **Optional TestPyPI dry run**: *Actions → Publish → Run workflow*.
4. **The owner tags** the merged commit:
   ```bash
   git tag -a vX.Y.Z -m "ttucs X.Y.Z"
   git push origin vX.Y.Z
   ```
5. **`publish.yml`** checks that the tag equals `ttucs.__version__`, downloads and verifies the
   weights asset, builds both packages and uploads via Trusted Publishing with `skip-existing`.
6. **GitHub Release** for `vX.Y.Z` with the CHANGELOG section as notes.

## Offline verification before claiming a release works

AGENTS.md section 5 requires the real user path:

1. build the wheels (`pip wheel --no-deps -w <dir> .` plus the built `ttucs_model_bow-*.whl`);
2. install both into a **fresh venv** with CPU torch;
3. run with no network (Linux: `unshare -rn ...`), `TTUCS_OFFLINE=1` and an empty `TTUCS_HOME`:
   - `ttucs info --verify` reports `source package`, the right `model_version` and
     `manifest_verified True`;
   - `ttucs predict "<a full slate>"` gives a sensible CatID and parsed fields;
   - `TTUCS.from_pretrained().predict(...)` works too;
4. after uploading a release asset, re-download it, compare the sha256, and test `ttucs download`
   from an empty cache.

## Where to change what

| you want to change | edit | retrain? | weights bump? |
|---|---|---|---|
| a microphone or recorder name | `slate/gear.py` | no | no |
| a slate cue word | `slate/parser.py` | no | no |
| an STT homophone | `slate/normalize.py` | no | no |
| a decision threshold default | `config.py` `inference` | no | no |
| filename or keyword rules | `text.py` | no | no |
| training phrases | `resources/catid_phrases.yaml` | yes | yes |
| vocabulary banks | `resources/synth_banks.yaml` | yes | yes |
| model shape or loss | `models/bow.py`, configs | yes | yes |
| a new model | `models/<name>.py` + `@register` + config | yes | new package |
| a new UCS version | `scripts/build_taxonomy.py`, `UCS_VERSION` | yes | yes |
| docs | `docs/`, `README.md`, cards | no | no |

## Current state

As of 2026-11-08 (AGENTS.md section 11): PyPI has `ttucs` 0.2.0 and 0.3.0 and `ttucs-model-bow`
0.1.1 and 0.2.0; tags `v0.1.0`, `v0.2.0`, `v0.3.0`, `model-bow-v0.1.1`, `model-bow-v0.2.0`; PR #12
is open with the parser fixes, the lazy module API and multi-clause training data, CI green, 113
tests, no new leakage, and no weights bump. Four retrain rounds found only multi-clause data durably
helpful; held-out realistic top-1 stayed at 0.675 against the published 0.713.

That section of AGENTS.md is the one to update when the state changes.
