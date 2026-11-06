# Data card

| source | used for | in repo? |
|---|---|---|
| UCS v8.2.1 Full List (753 CatIDs, Explanations, Synonyms) | label space, synthetic vocab, label documents | yes (`src/ttucs/resources`) |
| Synthetic descriptions (`ttucs.synth`) | training/val/test; tops every CatID up to 150 (min 60) | generated |
| Synthetic slates (`SlateGenerator`) | augmentation, parser + slate eval with ground truth | generated |
| BOOM ONE metadata sheet (40,325 rows, 553 CatIDs) | real-text training (cap 300/CatID) and held-out eval | **no**: `ttucs fetch-boom`, (c) BOOM Library |
| Pro Sound Effects HF dataset | not used: gated (login required), not retrieved | no |
| `examples/realistic_slates.jsonl` (80 hand-written slates) | sanity test set | yes |

**BOOM ONE analysis:** 201 UCS CatIDs are missing (e.g. ARCHIVED 15, BOATS 14, AMBIENCE 12, MUSICAL 9).
Classes are heavily imbalanced: median 34 rows per CatID, max 1,063 (OBJCont). There are 13,851
duplicate descriptions. Splits are grouped by FXName stem so takes never straddle splits.
Run `ttucs report` for `docs/assets/distribution.md` with plots.
