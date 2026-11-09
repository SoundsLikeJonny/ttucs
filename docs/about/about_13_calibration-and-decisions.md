# Calibration and the decision rule

The model outputs 738 logits. Turning those into a CatID, a confidence and a decision is done in
`TTUCS.probabilities` and `TTUCS._build` (`src/ttucs/pipeline.py`). This document explains the
temperature, the softmax, the ECE, and the three decisions.

## Step 1 — temperature scaling

```python
logits = self.model.logits(descriptions) / self.temperature
```

A neural network's softmax is usually **over-confident**: it says 0.98 when it is right 80% of the
time. Temperature scaling (Guo et al., 2017) divides the logits by a single learned scalar `T`
before the softmax:

- `T > 1` softens the distribution (less confident);
- `T < 1` sharpens it;
- `T = 1` leaves it unchanged.

`T` is fitted **after training**, on the validation split, by minimising cross-entropy over `T`
alone (`fit_temperature` in `models/base.py`, LBFGS, clamped to `[0.05, 20]`). It cannot change
which class wins — dividing by a positive scalar preserves the argmax — so accuracy is untouched
and only the confidences move.

The fitted value is stored in the model's `config.json` as `temperature` and loaded by
`TTUCS.from_pretrained`. The published 0.2.0 weights have `temperature = 0.8772`, i.e. slightly
sharper than raw.

## Step 2 — softmax

```python
logits = logits - logits.max(axis=1, keepdims=True)   # numerical stability
p = np.exp(logits); p = p / p.sum(axis=1, keepdims=True)
```

`p` is a `(n, 738)` array of calibrated CatID probabilities summing to 1 per row. The max
subtraction prevents `exp` overflow; it does not change the result.

## Step 3 — category probabilities

```python
cat_p = np.bincount(self.model.label_to_cat, weights=p, minlength=len(self.model.categories))
```

The CatID probabilities are summed into the 81 categories. This is a *marginalisation*, not the
category head's own output: it answers "how much probability mass is in DOORS?", which is what the
fallback rule needs. (The category head is used during training as an auxiliary loss, not at
inference.)

## Step 4 — the decision rule

Two thresholds, both in `inference:` in the config and overridable per call:

| threshold | default | meaning |
|---|---:|---|
| `unknown_threshold` | 0.12 | below this top-1 confidence, the CatID is not trusted |
| `category_threshold` | 0.50 | at or above this category mass, the category is trusted |

```
if best.confidence >= unknown_threshold:
    decision = "confident"
elif cat_p[top_category] >= category_threshold and the category has a MISC CatID:
    decision = "category_fallback"   # cat_id becomes that category's MISC CatID
else:
    decision = "unknown"
```

- **`confident`** — the CatID is returned as is.
- **`category_fallback`** — the subcategory is uncertain but the category is clear, so the CatID
  becomes the category's `MISC` entry (UCS's own convention for "this category, unspecified").
  The model's best guess stays in `alternatives[0]`, so nothing is lost.
- **`unknown`** — neither is trusted. The best guess is still returned (and a filename is still
  suggested), but `is_unknown` is true and the user should review it.

The `MISC` map is built once in `TTUCS.__init__` from the taxonomy: `{category: cat_id}` for every
entry whose subcategory is `MISC`.

## Step 5 — what the `Prediction` carries

| field | meaning |
|---|---|
| `cat_id`, `category`, `subcategory`, `cat_short` | the chosen entry (possibly the MISC fallback) |
| `confidence` | calibrated probability of the chosen CatID |
| `category_confidence` | calibrated category mass of the chosen category |
| `decision` | `confident` / `category_fallback` / `unknown` |
| `alternatives` | top-k `Candidate`s with their confidences |
| `categories` | top-k categories with their mass |
| `description`, `slate`, `fx_name`, `keywords`, `filename`, `filename_parts` | the rest |

## Measuring calibration: ECE

`evaluate.expected_calibration_error` bins predictions by confidence into 15 bins and averages
`|accuracy - confidence|` per bin, weighted by bin size:

```
ECE = sum over bins  (n_bin / n) * |accuracy_bin - mean_confidence_bin|
```

- ECE 0 means the confidences are exactly right; ECE 0.14 means the model is on average 14 points
  more confident than it should be.

Published 0.2.0 figures (from MODEL_CARD):

| split | n | ECE |
|---|---:|---:|
| synthetic descriptions | 22,384 | 0.020 |
| synthetic slates | 2,000 | 0.023 |
| dev descriptions | 332 | 0.069 |
| realistic slates (held out) | 80 | **0.138** |

The pattern is the important part: calibration is excellent on the synthetic distribution it was
fitted on and poor on real phrasing, because `T` is fitted on synthetic validation data. This is a
known limitation, stated in AGENTS.md section 6 and in the MODEL_CARD, and the obvious fix
(calibrating on the dev set) is listed as an idea not yet tried.

## Why the thresholds are low

`unknown_threshold: 0.12` looks permissive, and it is: with 738 classes, a uniform distribution
gives 0.0014 per class, so 0.12 is already ~85x uniform. The threshold is a floor against
near-uniform outputs, not a confidence guarantee. The honest guard is the `decision` field plus the
ECE caveat: a `confident` label on real phrasing is a suggestion, not a promise.

## Observed behaviour (published 0.2.0 weights, run in this repository)

| input | cat_id | decision | confidence | category confidence |
|---|---|---|---:|---:|
| `metal gate slam heavy` | `DOORGate` | confident | 0.948 | 0.979 |
| `glass bottle smash on concrete` | `GLASBrk` | confident | 0.940 | 0.976 |
| `recording traffic passing by, children playing, wind` | `CRWDChld` | confident | 0.910 | 0.911 |
| `a quiet room hum` | `AMBRoom` | confident | 0.459 | 0.548 |
| `a dull thud` | `ROCKImpt` | confident | 0.379 | 0.383 |
| `a faint hum` | `VOXWhsp` | confident | 0.750 | 0.765 |
| `the thing` | `CREAHmn` | confident | 0.975 | 0.978 |
| `something odd` | `SCIMech` | confident | 0.775 | 0.800 |

The last three rows are the caveat made concrete: vague or empty descriptions still produce high
confidences, because the model has never been trained to say "I do not know". The thresholds do not
catch them. Treat `decision` as a floor, not a certificate, and prefer the top-k list when the
description is thin.

## Tuning the thresholds

- `ttucs predict --threshold 0.3 ...` overrides `unknown_threshold` for one call.
- `TTUCS.from_pretrained(unknown_threshold=..., category_threshold=...)` overrides both.
- `evaluate.classification_metrics(..., threshold=...)` also reports `coverage` (share of inputs
  above the threshold) and `selective_top1` (accuracy among those), which is the honest way to
  choose a threshold: pick the coverage/accuracy trade-off you want.
- Changing the defaults is a user-visible change and needs a CHANGELOG entry.

Next: [about_14_pipeline-and-outputs.md](about_14_pipeline-and-outputs.md) shows how a `Prediction` is
assembled and printed.
