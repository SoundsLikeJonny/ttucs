# ttucs: Text to UCS

Turn spoken field-recording slates (speech-to-text output) into
[Universal Category System](https://universalcategorysystem.com/) metadata: CatID, Category/SubCategory,
confidence, parsed slate fields, a suggested UCS filename, and keywords. Built with PyTorch and MLflow.

```python
from ttucs import TTUCS

model = TTUCS.from_pretrained("models/ttucs-bow")
p = model.predict("um okay project thick as thieves location squamish forest recording a wooden door "
                  "creak slow mkh four sixteen into a mix pre six ninety six k twenty four bit take three")
p.cat_id, p.category_full, p.confidence   # ('DOORCreak', 'DOORS-CREAK', 0.993)
p.filename                                # 'DOORCreak_Wooden Door Creak Slow_NONE_ThickAsThieves_MKH416'
p.slate["sample_rate"], p.slate["take"]   # (96000, '3')
p.to_json()
```

Only the parsed **description** goes to the classifier. The other slate fields are returned but are
never used for training.

## Install & quick start

```bash
pip install -e ".[dev]"            # extras: [mlflow], [transformers], [all]
ttucs fetch-boom                   # optional: BOOM ONE metadata, local research use only
ttucs generate-data -c configs/bow.yaml
ttucs train -c configs/bow.yaml    # logs to MLflow (sqlite:///mlflow.db) and registers model "ttucs"
ttucs eval -m models/ttucs-bow
ttucs predict "glass bottle smash on concrete, exterior, take two"
ttucs parse "project nightfall, location tofino beach, waves on rocks, ninety six k, take one"
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

API features: `predict_batch`, `top_k`, aggregated category scores, temperature calibration, a
`decision` field (`confident` / `category_fallback` to the category's MISC CatID / `unknown`),
`from_mlflow("models:/ttucs/1")`, and `save_pretrained`.

## Models

| config | model | notes |
|---|---|---|
| `configs/bow.yaml` | hashed word+char n-gram bag of embeddings, CatID + Category heads | default; CPU; ~2 min training |
| `configs/minilm.yaml` | frozen all-MiniLM-L6-v2 + MLP head, initialised from label-document embeddings | optional, needs `[transformers]` (**not trained/verified in v0.1.0**) |
| `configs/zeroshot.yaml` | TF-IDF similarity to UCS Explanation+Synonyms | cold-start floor, no training |

## Results (BoW, seed 42, 10 epochs, 738 trainable CatIDs)

| test split | n | top-1 | top-5 | macro-F1 | category acc |
|---|---:|---:|---:|---:|---:|
| BOOM ONE held-out (real library text) | 5364 | 0.814 | 0.928 | 0.741 | 0.889 |
| synthetic descriptions | 8739 | 0.867 | 0.971 | 0.847 | 0.960 |
| synthetic spoken slates (parser → model) | 2000 | 0.864 | 0.968 | 0.841 | 0.952 |
| **hand-written realistic slates** | 80 | **0.713** | **0.913** | 0.705 | 0.875 |

Slate parser on synthetic slates: description token-F1 0.970, project 0.966, location 0.812,
sample rate 0.730, take 1.000.

**Read these numbers with caution.** Synthetic test sets come from the same generator as the training
data, and BOOM test rows come from the same library as the BOOM training rows, so both overstate
real-world accuracy. The 80 hand-written slates are the most honest signal, but the set is tiny.
See [MODEL_CARD.md](MODEL_CARD.md) and [DATA_CARD.md](DATA_CARD.md).

## Roadmap
- Fine-tune on your own labelled library metadata (Soundminer/BaseHead exports) and real slate recordings.
- Speech-to-text integration (e.g. Whisper) with word timestamps → slate-to-region alignment.
- Active learning: log low-confidence predictions and user corrections, then retrain.
- Train and evaluate the MiniLM path; consider a learned slate parser (token tagging).

## Credits & license
Code: MIT (c) 2025 Jon Evans. UCS category list v8.2.1 is by Tim Nielsen, Justin Drury, Kai Paquin
and contributors, a public-domain initiative (https://universalcategorysystem.com/). BOOM ONE metadata
is (c) BOOM Library and is not included in this repo.
