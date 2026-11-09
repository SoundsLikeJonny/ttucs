# Evaluation, leakage and the autotrain gate

Accuracy claims are the most dangerous part of a project like this, because synthetic test sets
flatter the model. This document explains the metrics, the three test sets and their ranking, the
leakage checks, and the automated gate that decides whether a retrain is allowed to become a beta.

## The metrics

`evaluate.classification_metrics(probs, y, label_to_cat, threshold)` returns:

| metric | definition | what it tells you |
|---|---|---|
| `n` | number of examples | the denominator, always report it |
| `top1` | share where argmax is correct | the headline accuracy |
| `top5` | share where the truth is in the top 5 | how useful the alternatives list is |
| `macro_f1` | macro F1 over classes present in the split | balance across classes |
| `category_acc` | share where the predicted *category* is correct | how often the fallback would be right |
| `ece` | expected calibration error, 15 bins | whether the confidences mean anything |
| `mean_confidence` | mean top-1 probability | compare with `top1` to see over-confidence |
| `coverage` | share with confidence >= threshold | how much the model would answer |
| `selective_top1` | accuracy among covered examples | accuracy at that coverage |

`coverage` and `selective_top1` are only computed when a threshold is passed; `evaluate_split`
passes `pipe.unknown_threshold`. They are the honest way to choose a threshold: pick the
coverage/accuracy trade-off you want rather than a round number.

## The three test sets, ranked by honesty

AGENTS.md section 6 ranks the signals:

1. **`examples/realistic_slates.jsonl` — 80 hand-written full slates.** The held-out test set.
   Never trained on, never tuned on, scored once at the end.
2. **`examples/dev_descriptions.jsonl` — 332 hand-written descriptions.** The tuning set. Used for
   every comparison during development.
3. **Synthetic splits** (`test_synth`, `test_slates_synth`). Same generator as the training data, so
   they overstate accuracy and are **not comparable across versions**, because the generator
   changes.

Published 0.2.0 numbers (MODEL_CARD, README):

| split | n | top-1 | top-5 | macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| realistic slates (held out) | 80 | 0.713 | 0.925 | 0.723 | 0.138 |
| dev descriptions | 332 | 0.789 | 0.925 | 0.788 | 0.069 |
| synthetic slates | 2,000 | 0.900 | 0.976 | 0.884 | 0.023 |
| synthetic descriptions | 22,384 | 0.908 | 0.983 | 0.904 | 0.020 |

With n=80, one slate is 1.25 points, so a difference of a few points is noise; important
comparisons are repeated with a second seed (`seed=7`) before being trusted.

## Parser metrics

`parser_field_accuracy(examples, parser)` scores the parser against the `SlateGenerator` ground
truth stored in `meta["truth"]`:

- exact match per field after normalisation (`description`, `project`, `location`, `creator`,
  `microphones`, `mic_setup`, `recorder`, `sample_rate`, `bit_depth`, `take`, `date`);
- `parser_description_token_f1` and `parser_description_exact` for the description, which is graded
  by token F1 because word order and glue vary.

Published: description exact match 0.921, token-F1 0.978.

## ECE

```python
for lo, hi in pairwise(linspace(0, 1, 16)):
    mask = (conf > lo) & (conf <= hi)
    ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
```

The gap between accuracy and confidence, averaged over confidence bins. 0 is perfect. The model is
well calibrated on synthetic data (0.020–0.023) and over-confident on real phrasing (0.138),
because the temperature is fitted on synthetic validation data. That is stated in the MODEL_CARD
and in AGENTS.md, and calibrating on the dev set is listed as an untried idea.

## Leakage

`scripts/check_leakage.py` enforces AGENTS.md section 6:

- no phrase of 3 or more words from `catid_phrases.yaml` or `synth_banks.yaml` may appear verbatim
  in `examples/dev_descriptions.jsonl` or `examples/realistic_slates.jsonl`;
- no dev text may share a 4-gram or substring with a realistic slate.

It exits 1 on a violation and prints the offending pairs. `tests/test_leakage.py` runs it in CI.

Known pre-existing violations are recorded as `leakage_baseline` in `configs/champion.json`: the dev
set contains "sci fi door" (dev:144) and "sci fi computer" (dev:149) verbatim from
`catid_phrases.yaml`, plus 10 generic 3-word bank entries. They are the owner's to clean up; the
gate blocks only **new** violations.

## The autotrain loop

`scripts/autotrain.py` (with `configs/autotrain.yaml` and `configs/champion.json`) is the gated
retrain loop. `make autotrain` runs it; without `--train` it is a dry run.

### Steps

1. **Fingerprint** the code and resources (`provenance.changed_vs`). If nothing changed since the
   champion, exit "nothing to do".
2. **Leakage gate** (`leakage_gate`): fail only if the violation counts got worse than the
   champion's baseline.
3. **Train** a candidate with `configs/autotrain.yaml` (release shape, `realistic_slates: null`).
4. **Score** the candidate on dev descriptions and the synthetic validation split
   (`dev_metrics`), never on the realistic slates.
5. **Gate** (`gate`) against `configs/champion.json`: `dev.top1` must gain at least
   `--min-gain` (default 0.015, about five of the 332 dev descriptions), and the other metrics must
   not regress beyond their tolerances
   (ECE must not get worse). A failure prints a gate table plus the candidate's worst confusions
   (`failure_report`) so the next change has a target.
6. **On pass with `--release`**: bump to the next beta version, update the pins, and open a PR with
   the comparison numbers and a champion-vs-candidate chart. It never merges, never pushes to
   `main`, and never publishes.

### Champion

`configs/champion.json` holds the reference record: metrics, parser metrics, config hash, git
state, weights version and the leakage baseline. `--init-champion` seeds it from the published
weights (downloading them only when not offline, otherwise falling back to the published numbers).
Every beta release refreshes it, so each gate compares against the current best.

### Why the realistic slates are excluded

The loop runs unattended and repeatedly. If it could see the 80 held-out slates, they would become
a tuning set within a few rounds and the headline number would stop meaning anything. Setting
`realistic_slates: null` makes that structurally impossible, not just a convention.

## Running evaluation by hand

```bash
ttucs eval -c configs/bow-release.yaml -m models/ttucs-bow-release
ttucs eval -m models/ttucs-bow-release --data examples/dev_descriptions.jsonl --predictions dev.csv
ttucs eval -m models/ttucs-bow-release --data examples/realistic_slates.jsonl --parse --json out.json
```

- `--data` can be repeated; each file becomes a split named after the file.
- `--parse` runs the slate parser first (use it for slates, not for descriptions).
- `--predictions` writes a CSV of text, truth, prediction and confidence.
- `--json` writes the metrics dict.
- `--mlflow` logs the evaluation to the tracking server.

## Reporting rules

When you report a number:

- name the split and its `n`;
- say whether it is synthetic (and therefore not comparable across versions);
- for the realistic slates, give the subset with no bank-phrase overlap next to the headline
  (29 slates: 0.552 -> 0.724 when the phrase bank landed);
- label estimates as estimates, and never claim a gain smaller than the noise floor without a
  second seed.

Next: [about_19_distribution-and-hub.md](about_19_distribution-and-hub.md) ships the result.
