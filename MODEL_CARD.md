# Model card: ttucs-bow v0.1.0

- **Task:** text (slate description) → one of 738 UCS v8.2.1 CatIDs (ARCHIVED workflow category excluded).
- **Architecture:** hashed word uni/bi-grams + char 3–5-grams (524,288 buckets, dim 192, sparse
  EmbeddingBag) → residual MLP → CatID head + auxiliary Category head (loss weight 0.3). SparseAdam/AdamW,
  OneCycle, label smoothing 0.05, temperature scaling (T=0.919 on validation).
- **Training data:** 109,105 examples: synthetic (template + UCS synonyms), BOOM ONE descriptions/FXNames
  (capped at 300 per CatID), and UCS label documents; 25% wrapped in synthetic slates and re-parsed.
- **Metrics:** see README. Top-1 is 0.713 on 80 hand-written realistic slates versus 0.81–0.87 on
  in-distribution sets.
- **Limitations:** English only. Trained on synthetic and one vendor's metadata. Ambiguous classes such
  as UCS material-vs-action (e.g. DOORWood vs DOORCreak) get confused. 201 CatIDs have no real examples.
  Confidence is calibrated on in-distribution validation data, so it is over-confident on real slates
  (ECE 0.16). The parser is rule-based and gear vocabulary is limited to the lexicon.
- **Intended use:** a metadata *suggestion* tool with a human in the loop, not autonomous labelling.
- **Tracking:** MLflow experiment `ttucs`, registered model `ttucs` (pyfunc, input `text`).
