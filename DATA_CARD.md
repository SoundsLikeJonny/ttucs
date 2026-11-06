# Data card

Data behind the published `ttucs-model-bow` weights. Everything is either in the repository or
generated from it, so the weights can be redistributed.

| source | used for | in repo? |
|---|---|---|
| UCS v8.2.1 Full List (753 CatIDs, Explanations, Synonyms) | label space, synthetic vocab, label documents | yes (`src/ttucs/resources`) |
| Synthetic descriptions (`ttucs.synth`) | training/val/test; 150 per CatID (min 60) | generated |
| Synthetic slates (`SlateGenerator`) | augmentation (25%), parser + slate eval with ground truth | generated |
| `examples/realistic_slates.jsonl` (80 hand-written slates) | sanity test set, never trained on | yes |

- 738 CatIDs are used (the ARCHIVED workflow category is excluded). Synthetic generation is balanced,
  so every CatID has training examples.
- Splits are deterministic (seed 42) and grouped so variants of one description never straddle splits.
- Generated data is fingerprinted from the config and cached in `processed_dir`.
- Run `ttucs report` for `docs/assets/distribution.md` with plots.

**Your own data:** real library metadata is the biggest lever for accuracy on real slates, but there
is no built-in option for it yet. To experiment, retrain from source with `pip install "ttucs[train]"`
and extend `ttucs.datasets.build`. Keep non-redistributable metadata out of the repository and out of
any weights you publish.
