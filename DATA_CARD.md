# Data card

Data behind the published `ttucs-model-bow` weights. Everything is either in the repository or
generated from it, so the weights can be redistributed.

| source | used for | in repo? |
|---|---|---|
| UCS v8.2.1 Full List (753 CatIDs, Explanations, Synonyms) | label space, synthetic vocab, label documents | yes (`src/ttucs/resources`) |
| Synthetic descriptions (`ttucs.synth`) | training/val/test; 300 per CatID (min 60) | generated |
| `src/ttucs/resources/catid_phrases.yaml` (hand-written everyday phrases for 494 CatIDs) | synthetic vocab (`phrase_prob` 0.5) | yes |
| Synthetic slates (`SlateGenerator`) | augmentation (25%), parser + slate eval with ground truth | generated |
| `examples/dev_descriptions.jsonl` (332 hand-written descriptions) | dev set for choosing settings, never trained on | yes |
| `examples/realistic_slates.jsonl` (80 hand-written slates) | held-out test set, never trained or tuned on | yes |

- The phrase bank and the dev set are separate files from the realistic slates, but they were written
  by the same author who had seen those slates. Phrases of three or more words that appeared verbatim
  in the dev or test sets were removed, and no dev text shares a 4-gram with a test slate. Short
  everyday wording ("door slam", "dog bark") still overlaps, which flatters the hand-written results
  somewhat; see the model card.
- 738 CatIDs are used (the ARCHIVED workflow category is excluded). Synthetic generation is balanced,
  so every CatID has training examples.
- Splits are deterministic (seed 42) and grouped so variants of one description never straddle splits.
- Generated data is fingerprinted from the config and cached in `processed_dir`.
- Run `ttucs report` for `docs/assets/distribution.md` with plots.
- Generator knobs available for experimentation (all default off; the release config ships
  `multi_clause_prob: 0.2` since the 0.2.0-era card above was written): `multi_clause_prob` joins
  same-class description pairs into multi-clause text, `verbose_prob` emits longer action-narration /
  scene-observation sentences. Both change the synthetic test sets, so results are not comparable
  across generator versions.

**Your own data:** real library metadata is the biggest lever for accuracy on real slates, but there
is no built-in option for it yet. To experiment, retrain from source with `pip install "ttucs[train]"`
and extend `ttucs.datasets.build`. Keep non-redistributable metadata out of the repository and out of
any weights you publish.
