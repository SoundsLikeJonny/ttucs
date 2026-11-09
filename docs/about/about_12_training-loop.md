# The training loop

`train_loop` in `ttucs/models/base.py` is a single generic function that trains any `nn.Module` the
models expose. `BowClassifier.fit` prepares the data and hands it three callables: a batch
generator, a forward function and an evaluation function. This document explains each moving part
and the reasoning behind it.

## The contract

```python
train_loop(module, batches, forward, evaluate, train_cfg, cat_weight, on_epoch=None) -> history
```

| argument | what it must do |
|---|---|
| `module` | the `nn.Module` to train (`BowNet`) |
| `batches(train: bool)` | yield tuples whose **last two items** are `y` (CatID ids) and `y_cat` (Category ids); also expose `num_batches()` for the scheduler |
| `forward(batch)` | return `(catid_logits, category_logits_or_None)` |
| `evaluate()` | return a dict with at least `val_top1` |
| `train_cfg` | the `train:` section of the config |
| `cat_weight` | weight of the auxiliary category loss |

`BowClassifier.fit` builds exactly that:

```python
def batches(train_mode=True):
    idx = list(range(len(feats))); rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        ids, offs = HashedNgramFeaturizer.collate([feats[j] for j in b])
        yield ids, offs, torch.from_numpy(y[b]), torch.from_numpy(yc[b])
```

Features are computed **once** before training (`feats = [...]`) and only collated per batch, which
is why an epoch over ~180k examples takes seconds.

## Two optimisers, one model

```python
sparse = [m.weight for m in module.modules()
          if isinstance(m, (nn.Embedding, nn.EmbeddingBag)) and m.sparse]
dense  = [p for p in module.parameters() if id(p) not in sparse_ids]
opts = [AdamW(dense, lr=lr, weight_decay=1e-5)]
if sparse: opts.append(SparseAdam(sparse, lr=lr))
```

- The 33.5M-row embedding table gets **SparseAdam**: only the rows touched by a batch receive a
  gradient and an update. A dense AdamW step would touch all 33.5M rows every batch, roughly 50x
  slower.
- Everything else (the residual block and both heads, ~276k parameters) gets **AdamW** with
  `weight_decay: 1e-5`.
- Both optimisers are stepped every batch, and both have their own scheduler.

## Schedule

```python
scheds = [OneCycleLR(o, max_lr=lr, total_steps=epochs * num_batches, pct_start=0.1) for o in opts]
```

`OneCycleLR` warms up for the first 10% of steps, peaks at `lr` (0.01 in the release config), then
decays. It needs the total step count up front, which is why `batches.num_batches` exists. A high
peak learning rate is what makes 15 epochs enough for a sparse table.

## Loss

```python
ce = nn.CrossEntropyLoss(label_smoothing=0.05)
loss = ce(logits, y)
if cat_logits is not None and cat_weight > 0:
    loss = loss + cat_weight * ce(cat_logits, y_cat)
```

- **Cross-entropy** on the 738 CatID logits.
- **Label smoothing 0.05**: the target distribution is 0.95 on the true class and 0.05 spread over
  the rest. It stops the model from driving logits to extremes on synthetic text, which would make
  calibration worse.
- **Auxiliary category loss** at weight 0.3: the same cross-entropy on the 81 Category logits. It
  gives a coarser, easier signal that helps sibling classes (`DOORGate` vs `DOORCreak`) share
  representation.

## Epoch loop and early stopping

```python
for epoch in range(epochs):
    module.train(); ... for batch in batches(True): zero_grad; loss.backward(); step; sched.step()
    module.eval(); metrics = evaluate()
    if metrics["val_top1"] > best + 1e-4: best, bad, best_state = ..., 0, deepcopy(state_dict)
    else: bad += 1; if bad >= patience: break
module.load_state_dict(best_state)
```

- Validation is `val_top1` on the validation split (10% of the data, grouped so variants of one
  description never straddle splits — [about_16_datasets-and-splits.md](about_16_datasets-and-splits.md)).
- The best state is kept in memory as a clone of the state dict and restored at the end, so the
  returned model is the best epoch, not the last.
- `patience: 3` stops after three epochs without a gain of more than 0.0001.
- `history` records `train_loss` and `val_top1` per epoch; `train.py` turns it into
  `training_history.png` and MLflow metrics.

## Reproducibility

- `torch.set_num_threads(train_cfg["num_threads"])` (8 by default) fixes the thread count.
- The batch order uses `random.Random(seed)` with `seed: 42` from the config.
- The data generator is seeded separately (seed 42, grouped splits).
- Retraining with the same code, config and seed is deterministic on the same machine; the
  weights tarball sha256 changes whenever `config.json` changes, because it records
  `ttucs_version`.

## What the loop does not do

- No gradient clipping (not needed at this size and learning rate).
- No mixed precision (CPU fp32 training; fp16 is only the export format).
- No distributed training (a laptop CPU is the target).
- No data augmentation inside the loop (augmentation happens in `datasets/build.py`).

## Cost

On a laptop CPU with the release config (300 synthetic examples per CatID, ~180k training
examples, 15 epochs, batch 256): roughly 3–5 minutes, which is the figure in AGENTS.md and the
README. The smoke config (`configs/smoke.yaml`: 12 per class, 2 epochs, 65,536 buckets x 32 dims)
runs in well under a minute in CI.

Next: [about_13_calibration-and-decisions.md](about_13_calibration-and-decisions.md) turns logits into
confidences and a decision.
