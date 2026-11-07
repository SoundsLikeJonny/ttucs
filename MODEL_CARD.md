# Model card: ttucs-model-bow 0.2.0

- **Task:** text (slate description) → one of 738 UCS v8.2.1 CatIDs (ARCHIVED workflow category excluded).
- **Distribution:** PyPI package `ttucs-model-bow` (data only), installed automatically with `ttucs`.
  It holds `config.json`, fp16 `weights.safetensors` (67.7 MB) and a SHA-256 `manifest.json`.
  Inference runs locally on CPU with no network access.
- **Architecture:** hashed word uni/bi-grams + char 3–5-grams (131,072 buckets, dim 256, sparse
  EmbeddingBag) → residual MLP → CatID head + auxiliary Category head (loss weight 0.3). SparseAdam/AdamW,
  OneCycle, label smoothing 0.05, temperature scaling (T=0.877 on validation). Config:
  [`configs/bow-release.yaml`](https://github.com/SoundsLikeJonny/ttucs/blob/main/configs/bow-release.yaml).
- **Training data:** redistributable sources only: synthetic descriptions (300 per CatID) built from
  templates, UCS synonyms and a hand-written bank of everyday library phrasing for ~490 CatIDs
  (`resources/catid_phrases.yaml`), UCS label documents, and 25% of examples wrapped in synthetic slates
  and re-parsed. 180,633 training / 21,688 validation examples. `config.json` records the sources under `training_data`.
- **Metrics** (test splits, same seed; fp16 matches fp32 to 4 decimals):

  | split | n | top-1 | top-5 | macro-F1 | ECE |
  |---|---|---|---|---|---|
  | synthetic descriptions | 22,384 | 0.908 | 0.983 | 0.904 | 0.020 |
  | synthetic slates | 2,000 | 0.900 | 0.976 | 0.884 | 0.023 |
  | hand-written dev descriptions (tuning set) | 332 | 0.789 | 0.925 | 0.788 | 0.069 |
  | **hand-written realistic slates (held out)** | 80 | **0.713** | **0.925** | 0.723 | 0.138 |

  0.1.1 scored 0.488 top-1 / 0.813 top-5 on the realistic slates and 0.491 / 0.807 on the dev set.
  Settings (phrase bank, object routing, 300 per CatID, 256 dims) were chosen on the dev set and
  synthetic validation only. Caveats: the phrase bank, the dev set and the realistic slates were all
  written by hand for this project, so they share vocabulary (51 of the 80 realistic slates contain a
  bank phrase of their own CatID; bank phrases of 3+ words that occur verbatim in either set were
  removed). On the 29 slates with no such match, top-1 rose from 0.552 to 0.724. The synthetic test
  sets changed with the generator, so they are not comparable across versions.

  The unpublished 0.1.0 research model (524,288 × 192, fp32, 403 MB) also used non-redistributable
  real library metadata and scored 0.713 top-1 on the realistic slates, the same as this model. For
  0.1.1 the gap came from the real-world training text, not from model size: a 524,288 × 192 model
  trained on the 0.1.1 redistributable data scored 0.538. With n=80, differences of a few slates are
  noise (one slate = 1.25 points).
- **Limitations:** English only. Trained only on synthetic text, so real slate wording is out of
  distribution: expect the right CatID in the top 5 more often than at top 1. Remaining confusions are
  mostly close UCS siblings (CRWDCheer vs CRWDSprt, FIRECrkl vs FIREBurn, AMB* ambiences vs the
  source class, e.g. AMBInsc vs ANMLInsc). Confidence is calibrated on synthetic validation data, so it
  is over-confident on real slates (ECE 0.14). The parser is rule-based and gear
  vocabulary is limited to the lexicon. Training on real library metadata helped a lot in the
  0.1.0 experiments (see above).
- **Intended use:** a metadata *suggestion* tool with a human in the loop, not autonomous labelling.
- **License:** MIT.
