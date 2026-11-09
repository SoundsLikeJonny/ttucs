# AGENTS.md

Guidance for AI coding agents (and new contributors) working on **ttucs**. Read this before changing
anything. Human-facing docs: [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md),
[MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md), [docs/RELEASING.md](docs/RELEASING.md).

## 1. What this project is and why it exists

ttucs ("Text to UCS") turns **spoken field-recording slates** into
[Universal Category System](https://universalcategorysystem.com/) (UCS v8.2.1) metadata.

A recordist says something like *"Project Thick As Thieves, location Squamish forest, recording a
wooden door creak slow, MKH 416 into a MixPre, 96k 24 bit, take 3"*. Speech-to-text turns it into text,
and ttucs returns:

- a CatID (`DOORCreak`), Category/SubCategory and a calibrated confidence, plus the top-k alternatives;
- the parsed slate fields (project, location, mics, recorder, sample rate, bit depth, take, ...);
- a suggested UCS filename and keywords.

**Owner's goals (keep these in mind when making trade-offs):**

- A simple, minimal, genuinely useful tool for sound designers and recordists. Prefer small, clear
  code and a small install over features.
- `pip install ttucs` must work out of the box and **inference must run fully offline**.
- Everything published, including the weights, must be **redistributable** (MIT code, MIT weights,
  public UCS list).
- The main quality goal is **accuracy on real slate wording**, not on synthetic test sets.
- Next lever on the roadmap: fine-tuning on a user's *own* labelled library metadata (Soundminer or
  BaseHead exports) and real slate recordings. This is not built in yet.

## 2. Hard rules

1. **Git.** Work on a feature branch and open a PR to `main`. **Never merge a PR yourself, never
   push to `main`, and never force-push `main`.** The owner reviews and merges.
2. **No third-party sound-library metadata.** Do not add, download, train on or reference any
   commercial or third-party sound-effects library's metadata. This applies to the repo, the
   published weights, docs, commit messages and PRs, so do not name such libraries at all. Training
   data is synthetic text plus the UCS list only. (Users may train private models on their own data;
   that must never reach the repo or the published weights.)
3. **Never commit** `data/`, `models/`, `dist/`, `build/`, `mlruns/`, `mlflow.db`, `*.pt`,
   `*.safetensors`, tarballs, credentials, or the agent-platform marker file `.abacus.donotdelete`.
   Stage files explicitly; do not use `git add -A` or `git add .`.
4. **Offline first.** Nothing may touch the network at import time or during prediction.
   `TTUCS_OFFLINE=1` (or `HF_HUB_OFFLINE=1`) must guarantee zero network access.
5. **Only the parsed description is classified.** Slate fields (location, gear, take, ...) are
   returned but must never feed the classifier.
6. **Evaluation hygiene.** `examples/realistic_slates.jsonl` is the **held-out test set**. Never
   train on it, tune on it, or copy its wording into training resources. Tune on
   `examples/dev_descriptions.jsonl` and synthetic validation (see section 6).
7. **Truthful reporting.** Never say something "works", is "verified" or is "released" unless you ran
   it and saw the result. Test the real user path: a clean venv, the real wheel and the CLI, under no
   network when offline behaviour matters. State exactly what was and was not checked. Lead with
   failures and caveats.
8. **Changelog and SemVer** for every user-visible change (section 8).
9. **Secrets.** Never paste tokens into commits, issues, PRs, logs or chat. PyPI publishing uses
   Trusted Publishing and needs no token.
10. **Privacy.** Do not add the owner's personal details (address, contact info and so on) to the
    repo. Author metadata stays as it is.

## 3. How it works (high level)

```
speech-to-text slate
   │
   ▼
ttucs.slate (rule-based parser + gear lexicon + number/word normalisation)
   ├── slate fields ─────────────────────────────┐
   └── description ─▶ classifier (ttucs.models) ─┴─▶ Prediction
                       CatID + Category heads,      (cat_id, confidence, decision,
                       temperature calibration      alternatives, filename, keywords, slate)
```

- **Taxonomy** (`ttucs.taxonomy`, `resources/ucs_v8.2.1*.csv`): 753 CatIDs; 738 are used because the
  `ARCHIVED` workflow category is excluded.
- **Decision logic:** the `decision` field is `confident`, `category_fallback` (falls back to the
  category's MISC CatID when the CatID is unsure but the category is clear) or `unknown`. It is
  driven by `inference.unknown_threshold` and `inference.category_threshold`.
- **Models** (`ttucs.models`, registered with `@register("name")`):
  - `bow`: hashed word 1–2-gram plus char 3–5-gram bag of embeddings, a residual MLP, and CatID and
    Category heads. **This is the published model.**
  - `zeroshot`: TF-IDF similarity to UCS Explanations and Synonyms. A cold-start floor that needs no
    training.
  - `minilm` (encoder): frozen MiniLM plus an MLP head. Optional and **not trained or verified yet**.
- **Training data** (`ttucs.synth`, `ttucs.datasets.build`):
  - `DescriptionGenerator` builds library-style descriptions per CatID from:
    - UCS Explanations and Synonyms;
    - `resources/synth_banks.yaml` (modifiers, contexts, actions, objects);
    - `resources/catid_phrases.yaml` (hand-written everyday head phrases for 494 CatIDs, used with
      `phrase_prob`).
  - **Object routing:** objects named in a specific subcategory go to that subcategory, and
    MISC/GENERAL classes only get the leftover objects.
  - `SlateGenerator` wraps descriptions into spoken-style slates, used for augmentation and to
    test the parser against ground truth.
  - UCS label documents are added as extra training text.
  - Splits are deterministic (seed 42) and grouped, so variants of one description never straddle
    splits.
  - Synthetic text should **read like professional sound-library metadata** (concise source +
    action + modifiers, e.g. "metal gate slam heavy"), plus everyday recordist phrasing.
- **Config** (`ttucs.config`): `DEFAULTS` are merged with a YAML file, then dotted CLI overrides,
  e.g. `ttucs train -c configs/bow.yaml train.epochs=3`. `configs/bow-release.yaml` defines the
  published weights.
- **Tracking:** MLflow (SQLite `mlflow.db`) is optional and lives in the `[train]` extra.
- **Distribution and model hub** (`ttucs.hub`):
  - Two PyPI distributions:
    - `ttucs`: code, UCS list and CLI.
    - `ttucs-model-bow`: data only (`config.json`, fp16 `weights.safetensors`, `manifest.json` with
      SHA-256 hashes). It lives in `packages/ttucs-model-bow/`.
  - `ttucs` depends on `ttucs-model-bow` with a compatible range.
  - Model resolution order:
    1. an explicit path;
    2. the installed `ttucs_model_bow` package;
    3. the cache (`$TTUCS_HOME` or the platform cache dir);
    4. a SHA-256-verified download of the GitHub Release asset, only when not offline.
  - `MODELS["bow"]` in `hub.py` pins the weights version, archive name and sha256.

### Repository map

| path | what |
|---|---|
| `src/ttucs/cli.py` | `ttucs` CLI: `predict`, `parse`, `info`, `download`, `generate-data`, `train`, `eval`, `report` |
| `src/ttucs/pipeline.py` | `TTUCS` API (`from_pretrained`, `predict`, `predict_batch`, `top_k`, `save_pretrained`, `from_mlflow`) |
| `src/ttucs/slate/` | parser, gear lexicon, text normalisation |
| `src/ttucs/synth/generator.py` | synthetic description and slate generators, object routing |
| `src/ttucs/datasets/build.py` | builds and caches synthetic, label-document and realistic splits |
| `src/ttucs/models/` | `base.UCSModel`, `bow`, `zeroshot`, `encoder` |
| `src/ttucs/train.py`, `evaluate.py`, `reporting.py` | training loop with calibration, metrics (top-1/5, macro-F1, category accuracy, ECE, parser metrics), plots |
| `src/ttucs/hub.py` | offline-first model resolution, download, fp16 export, manifest checks |
| `src/ttucs/_extras.py` | clear install hints when `[train]` extras are missing |
| `src/ttucs/resources/` | UCS CSVs, `synth_banks.yaml`, `catid_phrases.yaml` (shipped via the `*.csv`/`*.yaml` package-data glob) |
| `configs/` | `bow-release.yaml` (published), `bow.yaml` (larger research config), `smoke.yaml` (CI), `minilm.yaml`, `zeroshot.yaml` |
| `examples/` | `realistic_slates.jsonl` (80, held-out test), `dev_descriptions.jsonl` (332, tuning) |
| `packages/ttucs-model-bow/` | data-only weights distribution, `RELEASE_SHA256` |
| `scripts/build_model_package.py` | deterministic weights tarball and wheel (wheel must be at most 90 MB) |
| `scripts/build_taxonomy.py` | regenerates the UCS resources |
| `.github/workflows/` | `ci.yml` (ruff, pytest, smoke train, packaging check on Python 3.10–3.12), `publish.yml` (Trusted Publishing) |

## 4. Environment and commands

The owner works on **Windows (PowerShell 5.1)** with **WebStorm**. When you give the owner commands,
provide PowerShell versions:

- no `&&`: put each command on its own line;
- `$env:NAME = "value"` for environment variables;
- `.venv\Scripts\Activate.ps1` to activate a venv;
- no backslash line continuations.

Dev setup (bash):

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch is enough
pip install -e packages/ttucs-model-bow -e ".[dev]"                  # local, weightless model package first
pre-commit install
```

Dev setup (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e packages/ttucs-model-bow -e ".[dev]"
pre-commit install
```

The local `packages/ttucs-model-bow` has **no weights** until you build them. To run predictions in
dev, either point at a trained model dir (`-m models/...`) or install the released weights.

| task | command |
|---|---|
| lint and test (must pass before any PR) | `ruff check src tests scripts` then `pytest -q` (or `make lint test`) |
| CI smoke train | `make smoke` (`configs/smoke.yaml`) |
| train the release model (CPU, ~3–5 min) | `ttucs train -c configs/bow-release.yaml` → `models/ttucs-bow-release` |
| evaluate (default test splits + realistic slates) | `ttucs eval -c configs/bow-release.yaml -m models/ttucs-bow-release` |
| evaluate on the dev set | `... eval ... --data examples/dev_descriptions.jsonl --predictions dev.csv` |
| build weights tarball and wheel | `python scripts/build_model_package.py --model-dir models/ttucs-bow-release --tarball --wheel` |
| check the resolved model | `ttucs info --verify` |
| auto-train, gate and beta-release loop | `make autotrain` (`python scripts/autotrain.py --train`; dry run without `--train`; `--release` opens a beta PR) |

Style: ruff with line length 110, target py310 (rules in `pyproject.toml`). Use type hints and keep
dependencies minimal: anything heavy goes in an extra (`[train]`, `[transformers]`), never in the core
install. Add tests for new behaviour.

## 5. Install and offline verification (do this before claiming a release works)

1. Build the wheels: `pip wheel --no-deps -w <dir> .` plus the built `ttucs_model_bow-*.whl`.
2. Install both into a **fresh venv** with CPU torch.
3. Run with no network (Linux: `unshare -rn ...`), `TTUCS_OFFLINE=1` and an empty `TTUCS_HOME`:
   - `ttucs info --verify` must report `source package`, the right `model_version` and
     `manifest_verified True`;
   - `ttucs predict "<a full slate>"` must give a sensible CatID and parsed fields;
   - check the Python API `TTUCS.from_pretrained().predict(...)` too.
4. After uploading a release asset, re-download it and compare the sha256, and test `ttucs download`
   from an empty cache.

## 6. Accuracy work and evaluation discipline

- **Ranking of signals** (most to least honest):
  1. the 80 hand-written realistic slates;
  2. the 332 hand-written dev descriptions;
  3. synthetic test sets.

  Synthetic sets come from the same generator as the training data and overstate accuracy. They
  also change whenever the generator changes, so they are **not comparable across versions**.
- **Choose settings on dev plus synthetic validation only.** Score the realistic slates once, at
  the end.
- **Small n.** With n=80, one slate is 1.25 points. Check important comparisons with a second seed
  (`seed=7`) before trusting a gain of a few points.
- **Leakage checks** whenever you edit `catid_phrases.yaml`, `synth_banks.yaml` or the dev set:
  - no phrase of 3 or more words may appear verbatim in the dev or test sets;
  - no dev text may share a 4-gram or substring with a realistic slate.
- **Report overlap honestly.** Give accuracy on the realistic-slate subset with no bank phrase of its
  own CatID next to the headline number.
- **Retraining is deterministic** for the same code, config and seed. However, the weights tarball
  sha256 changes whenever `config.json` changes; it records `ttucs_version`, so a version bump
  changes it.
- **Known remaining confusions:**
  - VEH* vs DOOR* (vehicle doors);
  - AMB* ambiences vs their source class (e.g. AMBInsc vs ANMLInsc);
  - CRWD* siblings (CRWDCheer vs CRWDSprt);
  - FIRECrkl vs FIREBurn.

  Confidence is over-confident on real phrasing (ECE ≈ 0.14) because calibration uses synthetic
  validation data.
- **Size limits.** The fp16 weights wheel must stay **≤ 90 MB** (enforced by the build script;
  PyPI's limit is 100 MB). The current release is 131,072 buckets × 256 dims ≈ 63 MB.
- **Ideas not yet tried:**
  - calibrating on the dev set;
  - more phrase coverage for the 244 CatIDs that have no phrases yet;
  - hard-negative phrases for sibling classes;
  - an AMB-vs-source disambiguation rule;
  - a user fine-tuning path for their own metadata.

## 7. Naming conventions

| thing | name |
|---|---|
| code distribution / import | `ttucs` / `ttucs` |
| weights distribution / import | `ttucs-model-bow` / `ttucs_model_bow` |
| code release tag (triggers PyPI publish) | `v<ttucs version>`, e.g. `v0.3.0` |
| weights release tag (must **not** start with `v`) | `model-bow-v<weights version>`, e.g. `model-bow-v0.2.0` |
| weights release asset | `model-bow-v<version>.tar.gz` (+ `.sha256`) |
| release model output dir | `models/ttucs-bow-release`; processed data in `data/processed-release` |
| UCS filename produced | `CatID_FX Name_CreatorID_SourceID[_UserData]`, e.g. `DOORCreak_Wooden Door Creak Slow_NONE_ThickAsThieves_MKH416` |
| branches | `feat/...`, `fix/...`, `docs/...`, `chore/...` |
| commits | [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`) |

## 8. Versioning and changelog

- **SemVer 2.0.0** for both distributions, versioned **independently**:
  - `ttucs` version: `src/ttucs/_version.py` (single source; `pyproject.toml` reads it).
  - `ttucs-model-bow` version: `packages/ttucs-model-bow/pyproject.toml` **and**
    `packages/ttucs-model-bow/src/ttucs_model_bow/__init__.py`. Bump it **only when the weights
    change**.
- **When the weights change**, update all of these together:
  - `hub.MODELS["bow"]` (version, sha256, archive);
  - `packages/ttucs-model-bow/RELEASE_SHA256`;
  - the dependency range in `pyproject.toml` (`ttucs-model-bow>=X.Y.0,<X.(Y+1)`);
  - `MODEL_CARD.md` and the README results table;
  - the changelog section in `packages/ttucs-model-bow/README.md`.

  `tests/test_hub.py` checks that the sha and versions agree.
- **`CHANGELOG.md` follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/):**
  - Newest first, with `## [Unreleased]` at the top. Add a line there for every user-visible change.
  - At release time, rename it to `## [X.Y.Z] - YYYY-MM-DD` and leave a fresh, empty
    `## [Unreleased]`.
  - Group entries under `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`,
    `### Security`, in that order, omitting empty groups. Mark breaking changes with **Breaking:**.
  - Keep the compare links at the bottom up to date:
    `[Unreleased]: .../compare/vX.Y.Z...HEAD`, `[X.Y.Z]: .../compare/vPREV...vX.Y.Z`.
  - When accuracy changes, give concrete before/after numbers and name the test set.
- `packages/ttucs-model-bow/README.md` has its own short changelog (`### X.Y.Z - YYYY-MM-DD`).

## 9. Release process (summary; full steps in docs/RELEASING.md)

1. PR with the version bumps, sha, docs and CHANGELOG, then the owner merges it.
2. If the weights changed: train → build → upload `model-bow-vX.Y.Z.tar.gz` as GitHub Release
   `model-bow-vX.Y.Z`. Do not mark it "Latest".
   - **The sha in the repo must match the exact uploaded file.** Upload the file that was built and
     hashed; never rebuild after pinning the sha.
3. Optional TestPyPI dry run: *Actions → Publish → Run workflow*.
4. The owner tags the merged commit and pushes the tag:
   - `git tag -a vX.Y.Z -m "ttucs X.Y.Z"`
   - `git push origin vX.Y.Z`
5. `publish.yml` checks that the tag equals `ttucs.__version__`. It then downloads and verifies the
   weights asset, builds both packages and uploads them via Trusted Publishing (`skip-existing`).
6. Create the GitHub Release for `vX.Y.Z` with the CHANGELOG section as notes.

## 10. Working with the owner

- Be concise, concrete and honest. Give numbers with their n and caveats. Label estimates as
  estimates.
- Give Windows PowerShell 5.1 commands (section 4) when asking the owner to run something.
- Open a PR and stop; the owner reviews and merges. Ask before doing anything outward-facing
  (publishing, tagging, creating releases) unless it was explicitly requested.
- The owner is a senior technical sound designer and knows UCS, field recording and audio tooling
  well. Do not over-explain the domain; do explain ML trade-offs briefly.

## 11. Current state (update this section when it changes)

As of 2026-11-08:

- **PyPI:** `ttucs` 0.2.0 and 0.3.0; `ttucs-model-bow` 0.1.1 and 0.2.0.
- **Tags:** `v0.1.0`, `v0.2.0`, `v0.3.0`, `model-bow-v0.1.1`, `model-bow-v0.2.0`.
- **GitHub Releases:** `v0.2.0`, `v0.3.0` (Latest) and the two `model-bow-*` weights releases all
  exist. `v0.1.0` has a tag but no Release.
- **Published weights** `ttucs-model-bow` 0.2.0 (fp16, 131,072 × 256, 300 synthetic examples per
  CatID, phrase bank plus routing):

  | split | n | top-1 | top-5 | macro-F1 | ECE |
  |---|---:|---:|---:|---:|---:|
  | realistic slates (held out) | 80 | 0.713 | 0.925 | 0.723 | 0.138 |
  | dev descriptions | 332 | 0.789 | 0.925 | 0.788 | 0.069 |
  | synthetic slates | 2,000 | 0.900 | 0.976 | 0.884 | 0.023 |
  | synthetic descriptions | 22,384 | 0.908 | 0.983 | 0.904 | 0.020 |

  - The previous 0.1.1 weights scored 0.488 / 0.813 on the realistic slates.
  - On the 29 realistic slates with no bank-phrase overlap, top-1 went from 0.552 to 0.724.
  - Parser metrics: description exact match 0.921, token-F1 0.978.
- **Autotrain loop (added 2026-11-07):** `scripts/autotrain.py` + `configs/champion.json` retrain on
  code/resource changes, gate against the champion on dev + synthetic data only, and can open beta
  PRs (`--release`); the realistic slates stay out of the loop. Known pre-existing leakage
  (recorded as the baseline in `configs/champion.json`; the gate blocks only NEW violations): the
  dev set contains "sci fi door" (dev:144) and "sci fi computer" (dev:149) verbatim from
  `catid_phrases.yaml`, and 10 generic 3-word `synth_banks.yaml` entries (contexts/objects) match
  dev or realistic texts.
- **PR #12 (opened 2026-11-08, awaiting owner review):** parser fixes (MKH without the "h",
  spoken "dual a b" -> `Spaced Pair (AB)`, multi-clause descriptions: cue'd slates absorb the
  clauses that follow them, cue-less slates join all content clauses; perspective-only words and
  mic-brand fragments kept out), module-level `import ttucs as t; t.predict(...)` lazy API,
  multi-clause training data (`data.synthetic.multi_clause_prob: 0.2` in the release config),
  MLflow tracking on in `bow-release.yaml`. CI green, 113 tests, no new leakage (12 known baseline
  violations only). No weights bump: `ttucs-model-bow` stays 0.2.0 until the owner promotes a
  retrain (recommended candidate = round-2 config).
- **Four controlled retrain rounds (2026-11-08/09):** only multi-clause data durably helped
  (dev multi-clause slice 0.600 -> 0.673, n=55; synthetic test 0.909 -> 0.915; realistic
  category-acc 0.875 -> 0.900). Held-out realistic top-1 stuck at 0.675 across all rounds
  (published 0.713, n=80) - sibling-class confusion is the binding constraint. Parked opt-in
  knobs (commits local to the branch, default off, off in the release config):
  `data.synthetic.verbose_prob` (longer narration styles; round 3 cost multi-clause dev 0.673 ->
  0.582) and `model.head_repeat` (clause weighting via feature repetition; round 4 netted flat,
  and the parser's first clause is not always the label driver). Next levers: sibling
  hard-negative phrases, category-gated fallback rules, second seed before trusting small
  deltas.
