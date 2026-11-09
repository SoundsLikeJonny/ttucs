# Training, configuration and tracking

This document covers the `train:` half of the tool: how a config becomes a model directory, what
`ttucs train` does step by step, and how runs are tracked and reported.

## Configuration: `ttucs.config`

`DEFAULTS` in `src/ttucs/config.py` is a plain nested dict. A YAML file is deep-merged over it, and
dotted CLI overrides are applied last:

```bash
ttucs train -c configs/bow-release.yaml train.epochs=3 data.synthetic.phrase_prob=0.6
```

```python
cfg = load_config("configs/bow-release.yaml", ["train.epochs=3"])
```

`deep_merge` recurses into dicts and replaces everything else, so a YAML file only states what it
changes. `flatten(cfg)` produces `{"train.epochs": 3, ...}` for MLflow params and for the
`-o KEY=VALUE` CLI form. Values are parsed as YAML, so `false`, `3` and `[3, 5]` arrive as the
right types.

There is no schema validation beyond what the code indexes; a typo in a key is silently ignored
unless something reads it. When adding a knob, add it to `DEFAULTS` so it is discoverable.

### The knobs that matter

| key | default | release | effect |
|---|---:|---:|---|
| `seed` | 42 | 42 | data generation, batch order, splits |
| `ucs_version` | 8.2.1 | 8.2.1 | which taxonomy CSV to load |
| `data.synthetic.target_per_class` | 150 | 300 | examples per CatID |
| `data.synthetic.phrase_prob` | 0.5 | 0.5 | share using `catid_phrases.yaml` |
| `data.synthetic.multi_clause_prob` | 0.0 | 0.2 | share joined into `"a, b"` pairs |
| `data.synthetic.verbose_prob` | 0.0 | 0.0 | share of long narration (parked, off) |
| `data.slate_augment_fraction` | 0.25 | 0.25 | share re-parsed through a synthetic slate |
| `data.split` | 0.1 / 0.1 | same | val / test fractions |
| `model.num_buckets` | 262144 | 131072 | hash buckets = embedding rows |
| `model.embedding_dim` | 128 | 256 | embedding width |
| `model.head_repeat` | 1 | 1 | headline-clause repeat (parked, off) |
| `model.category_loss_weight` | 0.3 | 0.3 | auxiliary category loss weight |
| `train.epochs` | 12 | 15 | max epochs |
| `train.lr` | 1e-2 | 1e-2 | OneCycle peak learning rate |
| `train.patience` | 3 | 3 | early-stopping patience |
| `train.calibrate` | true | true | fit the temperature after training |
| `inference.unknown_threshold` | 0.12 | 0.12 | decision floor |
| `inference.category_threshold` | 0.5 | 0.5 | fallback threshold |

`configs/bow.yaml` is the larger research config (524,288 buckets x 192 dims, 150 per class);
`configs/smoke.yaml` is the CI config (65,536 x 32, 12 per class, 2 epochs); `configs/autotrain.yaml`
mirrors the release config but sets `realistic_slates: null`.

## `ttucs train`: the steps

`train.train(cfg, rebuild_data)`:

1. **Seed everything** (`random`, `numpy`, `torch`) and load the taxonomy.
2. **Load or build the splits** (`load_or_build`, fingerprint-checked —
   [about_16_datasets-and-splits.md](about_16_datasets-and-splits.md)).
3. **Build the label lists**: `labels` = trainable CatIDs in taxonomy order, `cats` = their
   categories. These lists are saved into `config.json`, so the model is self-describing.
4. **Instantiate the model** from `MODEL_REGISTRY[cfg["model"]["name"]]`.
5. **Open an MLflow run** (unless disabled) and log params, tags and counts.
6. **Fit** with `model.fit(train_pairs, val_pairs, {**cfg["train"], "seed": seed}, on_epoch)`;
   `on_epoch` streams `train_loss`, `val_top1` and `epoch_seconds` to MLflow.
7. **Calibrate**: `fit_temperature(val_logits, val_labels)` -> a scalar `T`.
8. **Save** the model directory with `extra={temperature, ucs_version, inference, ttucs_version,
   data_fingerprint, seed, training_data}`.
9. **Evaluate** every `test_*` split through a real `TTUCS` pipeline, plus parser field accuracy on
   `test_slates_synth`.
10. **Write reports** into `<output_dir>/reports/`: `metrics.json`, `training_history.png`,
    `per_class_<split>.csv`, `confusions_<split>.csv`, `category_confusion_<split>.png`, and
    `predictions_test_realistic.csv`.
11. **Log to MLflow**: per-split metrics, parser metrics, the reports directory, `config.json`, and
    optionally the model as a `pyfunc` (with signature and registration).

The returned summary dict is what `ttucs train --json` prints and what `scripts/autotrain.py`
consumes.

## The model directory

`UCSModel.save(dir, extra)` writes:

```
<output_dir>/
  config.json          model_name, labels, label_categories, model config, temperature,
                       ucs_version, ttucs_version, data_fingerprint, seed, training_data, inference
  weights.pt           torch state dict (training format)
  reports/             metrics.json, plots, CSVs
```

`weights.pt` is the training artefact. The release format is produced later by
`hub.export_model`, which writes `weights.safetensors` in fp16 plus a `manifest.json`
([about_19_distribution-and-hub.md](about_19_distribution-and-hub.md)).

## MLflow

- Default tracking URI: `sqlite:///mlflow.db` (a local file, git-ignored).
- `MLFLOW_TRACKING_URI` overrides it, **unless** offline mode is on, in which case the SQLite
  default is forced (`resolve_tracking_uri`).
- Experiment `ttucs`, run name from `mlflow.run_name`.
- Tags include `ttucs_version`, `ucs_version`, `model`, `python`, `torch`, the git state from
  `provenance.run_tags()` and the `data_fingerprint`.
- `mlflow.log_model: false` in `autotrain.yaml` because every logged model copies the ~63 MB
  weights into the artifact store; exploratory rounds log metrics and reports only.
- `ttucs.mlflow_model.log_pyfunc` registers a `pyfunc` flavour that loads a `TTUCS` pipeline, so a
  registered model version can be served and reloaded with `TTUCS.from_mlflow("models:/ttucs/1")`.

MLflow lives in the `[train]` extra. `_extras.require_mlflow` raises a clear error with the exact
`pip install` line if it is missing, and `-o mlflow.enabled=false` turns tracking off for a run.

## Provenance

`src/ttucs/provenance.py` records what produced a run:

| function | returns |
|---|---|
| `code_fingerprint(root)` | sha1 over the tracked source files, so a run can be tied to code |
| `git_info(root)` | branch, commit, dirty flag (empty strings outside a checkout) |
| `run_tags(root)` | the subset logged as MLflow tags |
| `changed_vs(champion, root)` | whether code/resources changed since the champion was recorded |

`scripts/autotrain.py` uses `changed_vs` to decide whether a retrain is needed at all.

## Reports

`src/ttucs/reporting.py`:

- `distribution_report(train, tax, out_dir)` writes `docs/assets/distribution.md` plus
  `class_distribution.png` and `category_sources.png` — examples per CatID, per-source counts,
  imbalance ratio. `ttucs report` runs it.
- `training_history_png(history, path, title)` plots `train_loss` and `val_top1` per epoch.

`evaluate.py` writes the per-split CSVs and confusion plots listed above.

## Cost and determinism

- Release config on a laptop CPU: ~3–5 minutes for 15 epochs over ~180k examples.
- Same code, config and seed reproduce the same weights on the same machine.
- The weights tarball sha256 still changes whenever `config.json` changes, because it records
  `ttucs_version`.

## Common failures

| symptom | cause | fix |
|---|---|---|
| `MissingExtraError: training needs ...` | `[train]` not installed | `pip install "ttucs[train]"` |
| `MissingExtraError: MLflow tracking needs ...` | no `mlflow` | install `[train]` or `-o mlflow.enabled=false` |
| data rebuilt every run | a `data` key or the version changed | expected; check the fingerprint in the log |
| `KeyError` on a config key | typo, or a knob missing from `DEFAULTS` | add it to `DEFAULTS` |

Next: [about_18_evaluation.md](about_18_evaluation.md) covers the metrics and the gate.
