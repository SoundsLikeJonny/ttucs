# Datasets and splits

`ttucs/datasets/build.py` turns the generator's texts into the named splits the trainer and the
evaluator consume, and caches them as JSONL. This document explains each split, how the split
boundaries are drawn, and why the grouping matters.

## The `Example` record

Every split is a list of `Example`:

```python
@dataclass
class Example:
    text: str          # the description, or a full slate transcript for slate splits
    label: str         # the CatID
    source: str        # "synthetic" | "ucs_doc" | "slate:synthetic" | ...
    group: str         # split key: variants sharing a group never straddle splits
    meta: dict         # {"truth": {...}} for slates, {"slate": ..., "augmented": True} for augmentation
```

`write_jsonl` / `read_jsonl` round-trip it, so a built dataset is a plain, inspectable file.

## The splits `build_datasets` produces

| split | contents | used for |
|---|---|---|
| `train` | synthetic descriptions + UCS label documents, optionally slate-augmented | fitting |
| `val` | 10% of the synthetic descriptions | early stopping **and** temperature calibration |
| `test_synth` | 10% of the synthetic descriptions | synthetic description accuracy |
| `test_slates_synth` | 2,000 of `test_synth` wrapped into slates | synthetic slate accuracy + parser metrics |
| `test_realistic` | `examples/realistic_slates.jsonl` (80) | the headline held-out number |

`evaluate_splits` only looks at names starting with `test`, and parses first for the slate splits
(`SLATE_SPLITS`), so a slate split is scored exactly the way the CLI would score it.

## Grouped splitting

The naive approach — shuffle and slice — leaks. Two descriptions generated from the same template
and the same head noun are near-duplicates; if one lands in train and the other in test, the test
score is inflated.

`group_split` hashes the group key instead:

```python
def split_bucket(group, seed):
    h = hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()
    return int(h[:12], 16) / 16**12          # deterministic uniform in [0, 1)

b = split_bucket(ex.group or ex.text, seed)
test if b < test else val if b < test + val else train
```

- The bucket depends only on `(seed, group)`, so it is stable across runs, machines and operating
  systems — no `random.shuffle` ordering to depend on.
- Synthetic examples are grouped four at a time: `group=f"synth:{cat_id}:{i // 4}"`. Four
  consecutive samples from one CatID share a bucket, so a template variant cannot appear on both
  sides of the boundary.
- Label documents are grouped per CatID (`doc:{cat_id}`), so all of a class's UCS text stays
  together.
- Slate splits inherit the group of the description they wrapped, so a slate and its description
  never straddle.

The split fractions come from `data.split: {val: 0.1, test: 0.1}`.

## Balanced counts

```python
counts = {cid: max(scfg["min_per_class"], scfg["target_per_class"]) for cid in labels}
```

Every trainable CatID gets the same budget — 300 in `bow-release.yaml` and `autotrain.yaml`, 150 in
the `DEFAULTS`, 12 in `smoke.yaml`. `min_per_class` (60) is a floor for configs that set a lower
target. Balance is what makes macro-F1 meaningful and stops rare classes from being ignored.

## Label documents

`label_documents(taxonomy, labels)` adds, per CatID:

- the UCS Explanation;
- `"<category> <subcategory>"` lower-cased;
- chunks of up to 8 synonyms, lower-cased, up to 24 synonyms.

These are short, encyclopaedic texts, tagged `source="ucs_doc"`, grouped per CatID. They give every
class some training signal even if the banks are thin, and they are the same documents the
zero-shot model uses.

## Slate augmentation

```python
slate_augment(examples, fraction=0.25, seed)
```

For 25% of the non-`ucs_doc` training examples, the description is wrapped into a synthetic slate
and then **parsed back**, and the parser's description replaces the original text. The example
keeps its label and group, and `meta` records the slate and `augmented: True`.

Why: at inference the classifier sees the parser's output, not the generator's text. If the parser
drops a word or keeps a stray clause, the model should have seen that during training. This is the
cheapest way to close the train/inference distribution gap without labelling real slates.

## The synthetic slate test set

`make_slate_testset(examples, seed, n=2000)` samples 2,000 `test_synth` examples, wraps each in a
slate, and stores the generator's ground truth in `meta["truth"]`. Two uses:

1. classification accuracy on slate-shaped input (`test_slates_synth`);
2. `parser_field_accuracy`, which compares the parser's output field by field against that truth.

## Caching and the fingerprint

`ttucs train` calls `load_or_build(cfg, rebuild)`:

- `data_fingerprint(cfg)` is a sha1 of `{seed, ucs_version, data, ttucs_version}`;
- if `data/processed/manifest.json` exists with the same fingerprint, the JSONL files are reused;
- otherwise `generate_data` rebuilds every split, writes `<split>.jsonl` plus `manifest.json`
  (fingerprint, versions, per-split example and label counts), and returns them.

So changing `target_per_class`, `phrase_prob`, `multi_clause_prob`, the seed or the `ttucs` version
invalidates the cache automatically; changing only `train.epochs` does not. `--rebuild-data` forces
a rebuild. `ttucs generate-data` writes the same files without training.

The synthetic cache is also written separately to `data.processed.synthetic.cache`
(`data/processed/synthetic.jsonl`), which is what the distribution report reads.

## What is never in a split

- `examples/realistic_slates.jsonl` is only ever loaded as `test_realistic`, and
  `configs/autotrain.yaml` sets `realistic_slates: null` so the autotrain loop cannot see it at all.
- No third-party library metadata, ever (AGENTS.md rule 2).
- `data/` is git-ignored; splits are rebuilt, not committed.

Next: [about_training-and-config.md](about_training-and-config.md) consumes these splits.
