# ttucs: Text to UCS

ttucs turns spoken field-recording slates (speech-to-text output) into
[Universal Category System](https://universalcategorysystem.com/) metadata. For each slate it returns a
CatID, Category/SubCategory, a confidence, the parsed slate fields, a suggested UCS filename, and
keywords. It is built with PyTorch and MLflow, and inference runs locally and offline.

```bash
pip install ttucs          # also installs the pretrained weights (ttucs-model-bow, ~32 MB)
```

```python
from ttucs import TTUCS

model = TTUCS.from_pretrained()   # offline: loads the installed weights, no network
p = model.predict("Project Thick As Thieves, location Squamish forest, recording a wooden door creak slow, "
                  "Sennheiser MKH 416 and Sound Devices MixPre, 96k 24 bit, take 3")
p.cat_id, p.category_full, p.confidence
p.filename                        # e.g. 'DOORCreak_Wooden Door Creak Slow_NONE_ThickAsThieves_MKH416'
p.slate["sample_rate"], p.slate["take"]
p.to_json()
```

Only the parsed **description** goes to the classifier. The other slate fields are returned but are
never used for classification.

## Install

| what | command |
|---|---|
| inference (API + `ttucs predict/parse/info/download`) | `pip install ttucs` |
| training, evaluation, reports, MLflow | `pip install "ttucs[train]"` |
| MiniLM encoder model | `pip install "ttucs[transformers]"` |
| development | `pip install -e packages/ttucs-model-bow -e ".[dev]"` |

PyTorch is required. On machines without a GPU, install the much smaller CPU-only wheel first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Weights and offline use

The weights ship as a separate, data-only distribution,
[`ttucs-model-bow`](https://github.com/SoundsLikeJonny/ttucs/tree/main/packages/ttucs-model-bow).
`ttucs` depends on it, so `pip install ttucs` fetches it automatically. Wheels can't run code at install
time, which is why the weights are their own package. After installation nothing touches the network.

`TTUCS.from_pretrained()` (or `from_pretrained("bow")`) looks for a model in this order:

1. an explicit local directory: `TTUCS.from_pretrained("models/my-model")`;
2. the installed `ttucs_model_bow` package;
3. the user cache: `$TTUCS_HOME`, or the platform cache dir (e.g. `~/.cache/ttucs`), filled by `ttucs download`;
4. a SHA-256-verified download of the
   [GitHub Release asset](https://github.com/SoundsLikeJonny/ttucs/releases), **only** when offline mode is off.

Set `TTUCS_OFFLINE=1` (`HF_HUB_OFFLINE=1` is honoured too) to guarantee no network access. If no weights
are found, the error lists exactly what to install or run.

```bash
ttucs info [--verify]        # resolved model path/source, versions, offline mode (and sha256 check)
ttucs download --model bow   # fill the cache, e.g. for air-gapped images (--dest DIR, --url MIRROR --sha256 HEX)
```

`from_mlflow("models:/ttucs/1")` still loads models from an MLflow registry (needs `[train]` or `[mlflow]`).

## CLI

```bash
ttucs predict "glass bottle smash on concrete, exterior, take two"   # --json, -k 5, --file slates.txt
ttucs parse "project nightfall, location tofino beach, waves on rocks, ninety six k, take one"
ttucs generate-data -c configs/bow-release.yaml                       # [train]
ttucs train -c configs/bow-release.yaml                               # [train]; MLflow optional
ttucs eval -m models/ttucs-bow-release                                # [train]
ttucs report -c configs/bow-release.yaml                              # [train]
```

The API also offers `predict_batch`, `top_k`, aggregated category scores, temperature calibration, a
`decision` field (`confident` / `category_fallback` to the category's MISC CatID / `unknown`), and
`save_pretrained`.

## Models

| config | model | notes |
|---|---|---|
| `configs/bow-release.yaml` | hashed word+char n-gram bag of embeddings (131,072 × 128), CatID + Category heads | **published weights** (fp16 safetensors, ~34 MB); CPU; ~1 min training |
| `configs/bow.yaml` | the same, larger (524,288 × 192) | ~400 MB fp32; no accuracy gain on synthetic data (see model card) |
| `configs/minilm.yaml` | frozen all-MiniLM-L6-v2 + MLP head | optional, needs `[transformers]` (**not trained/verified yet**) |
| `configs/zeroshot.yaml` | TF-IDF similarity to UCS Explanation+Synonyms | cold-start floor, no training |

## Results: published model (`ttucs-model-bow` 0.1.1, fp16)

| test split | n | top-1 | top-5 | macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| synthetic descriptions | 11,244 | 0.874 | 0.972 | 0.861 | 0.020 |
| synthetic spoken slates (parser → model) | 2,000 | 0.859 | 0.970 | 0.841 | 0.031 |
| **hand-written realistic slates** | 80 | **0.488** | **0.813** | 0.509 | 0.182 |

The fp16 export scored the same as the fp32 checkpoint on all three splits, to 4 decimals.

**Read these numbers with caution.** The synthetic test sets come from the same generator as the
training data and overstate real-world accuracy. The 80 hand-written slates are the most honest signal:
about half are right at top-1 and about four in five are in the top 5. The set is tiny, and confidence
is over-optimistic on real phrasing (ECE 0.18). The model is trained only on redistributable data
(synthetic text plus UCS label documents). A 0.1.0 development model that also used non-redistributable
real library metadata reached 0.713 top-1 on the same 80 slates. Training on your own labelled
library metadata is the most promising way to close that gap; it is on the roadmap and not yet a
built-in option. See
[MODEL_CARD.md](https://github.com/SoundsLikeJonny/ttucs/blob/main/MODEL_CARD.md) and
[DATA_CARD.md](https://github.com/SoundsLikeJonny/ttucs/blob/main/DATA_CARD.md).

## Roadmap
- Fine-tune on your own labelled library metadata (Soundminer/BaseHead exports) and real slate recordings.
- Speech-to-text integration (e.g. Whisper) with word timestamps → slate-to-region alignment.
- Active learning: log low-confidence predictions and user corrections, then retrain.
- Train and evaluate the MiniLM path; consider a learned slate parser (token tagging).

## Releasing
See [docs/RELEASING.md](https://github.com/SoundsLikeJonny/ttucs/blob/main/docs/RELEASING.md)
(Trusted Publishing to TestPyPI/PyPI, model release assets) and the
[CHANGELOG](https://github.com/SoundsLikeJonny/ttucs/blob/main/CHANGELOG.md).

## Credits & license
Code and published weights: MIT (c) 2025 Jon Evans. The UCS category list v8.2.1 is by Tim Nielsen,
Justin Drury, Kai Paquin and contributors, a public-domain initiative
(<https://universalcategorysystem.com/>).
