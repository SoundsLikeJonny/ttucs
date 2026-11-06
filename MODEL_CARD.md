# Model card: ttucs-model-bow 0.1.1

- **Task:** text (slate description) → one of 738 UCS v8.2.1 CatIDs (ARCHIVED workflow category excluded).
- **Distribution:** PyPI package `ttucs-model-bow` (data only), installed automatically with `ttucs`.
  It holds `config.json`, fp16 `weights.safetensors` (33.8 MB) and a SHA-256 `manifest.json`.
  Inference runs locally on CPU with no network access.
- **Architecture:** hashed word uni/bi-grams + char 3–5-grams (131,072 buckets, dim 128, sparse
  EmbeddingBag) → residual MLP → CatID head + auxiliary Category head (loss weight 0.3). SparseAdam/AdamW,
  OneCycle, label smoothing 0.05, temperature scaling (T=0.899 on validation). Config:
  [`configs/bow-release.yaml`](https://github.com/SoundsLikeJonny/ttucs/blob/main/configs/bow-release.yaml).
- **Training data:** redistributable sources only: synthetic descriptions (templates + UCS synonyms,
  150 per CatID), UCS label documents, and 25% of examples wrapped in synthetic slates and re-parsed.
  91,797 training / 10,964 validation examples. `config.json` records the sources under `training_data`.
- **Metrics** (test splits, same seed; fp16 matches fp32 to 4 decimals):

  | split | n | top-1 | top-5 | macro-F1 | ECE |
  |---|---|---|---|---|---|
  | synthetic descriptions | 11,244 | 0.874 | 0.972 | 0.861 | 0.020 |
  | synthetic slates | 2,000 | 0.859 | 0.970 | 0.841 | 0.031 |
  | hand-written realistic slates | 80 | 0.488 | 0.813 | 0.509 | 0.182 |

  The unpublished 0.1.0 research model (524,288 × 192, fp32, 403 MB) also used non-redistributable
  real library metadata and scored 0.713 top-1 on the realistic slates. Most of that gap comes from the
  real-world training text, not from the smaller size: a 524,288 × 192 model trained on the same
  redistributable data scored 0.538. With n=80, differences of a few slates are noise.
- **Limitations:** English only. Trained only on synthetic text, so real slate wording is out of
  distribution: expect the right CatID in the top 5 more often than at top 1. Material-vs-action
  classes (e.g. DOORWood vs DOORCreak) get confused. Confidence is calibrated on synthetic validation
  data, so it is over-confident on real slates (ECE 0.18). The parser is rule-based and gear
  vocabulary is limited to the lexicon. Training on real library metadata helped a lot in the
  0.1.0 experiments (see above).
- **Intended use:** a metadata *suggestion* tool with a human in the loop, not autonomous labelling.
- **License:** MIT.
