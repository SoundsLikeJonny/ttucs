# ttucs-model-bow

These are the pretrained weights for [ttucs](https://github.com/SoundsLikeJonny/ttucs), the Text-to-UCS
slate classifier. The package contains data only: `config.json`, fp16 `weights.safetensors` and a
SHA-256 `manifest.json`. It has no code dependencies and makes no network calls.

You normally don't install it directly: `pip install ttucs` pulls it in. ttucs then finds it offline:

```python
from ttucs import TTUCS

model = TTUCS.from_pretrained()          # resolves the installed ttucs-model-bow weights
print(model.predict("wooden door creak slow, take 3").cat_id)
```

`ttucs info --verify` shows where the weights were loaded from and checks them against the manifest.

## Model

- **Architecture:** hashed word 1–2-gram + char 3–5-gram bag of embeddings (131,072 buckets × 256 dims),
  a residual MLP, then UCS CatID and Category heads. Temperature-calibrated. 738 UCS v8.2.1 CatIDs.
- **Training data:** synthetic descriptions and slates generated from the UCS v8.2.1 list, plus UCS
  label documents. No third-party library data is used, so the weights can be redistributed.
- **Metrics and limitations:** see the
  [model card](https://github.com/SoundsLikeJonny/ttucs/blob/main/MODEL_CARD.md).
- **License:** MIT (weights). The UCS list is a public-domain initiative by Tim Nielsen, Justin Drury,
  Kai Paquin and contributors (<https://universalcategorysystem.com/>).

## Versioning

This package is versioned independently of `ttucs` and bumps only when the weights change. `ttucs`
declares a compatible range (`ttucs-model-bow>=0.2.0,<0.3`). The model's `config.json` records
`min_ttucs_version`, which ttucs checks at load time.

## Changelog

### 0.2.0 - 2026-11-07
- Retrained with a per-CatID everyday phrase bank, object routing to specific subcategories, 300
  examples per CatID and 256-dim embeddings (131,072 × 256, ~68 MB). Hand-written realistic slates:
  top-1 0.488 → 0.713, top-5 0.813 → 0.925. Still synthetic + UCS data only.

### 0.1.1 - 2026-11-06
- First published weights: synthetic + UCS data only, compact (131,072 × 128), fp16 safetensors, SHA-256 manifest.
