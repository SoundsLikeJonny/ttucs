# The model architecture, layer by layer

The published model is `bow` (`ttucs/models/bow.py`). This document walks through it from input
string to logits, with the exact shapes and parameter counts of the release configuration
(`configs/bow-release.yaml`: `num_buckets: 131072`, `embedding_dim: 256`, 738 labels, 81
categories, `dropout: 0.2`).

## The full picture

```
description string
   │  HashedNgramFeaturizer
   ▼
ids: int64[N]            offsets: int64[B]          (N = total n-grams, B = batch size)
   │  nn.EmbeddingBag(131072, 256, mode="mean", sparse=True)
   ▼
h0: float32[B, 256]                                 mean of the n-gram embeddings
   │  nn.Dropout(0.2)
   ▼
h1: float32[B, 256]
   │  residual block: Linear(256,256) -> GELU -> LayerNorm(256), added back
   ▼
h2: float32[B, 256]                                 h2 = h1 + hidden(h1)
   ├── nn.Linear(256, 738)  -> catid_logits: float32[B, 738]
   └── nn.Linear(256, 81)   -> category_logits: float32[B, 81]
```

## Layer 1 — `EmbeddingBag`

```python
self.emb = nn.EmbeddingBag(num_buckets, dim, mode="mean", sparse=True)
nn.init.uniform_(self.emb.weight, -0.05, 0.05)
```

- A lookup table with one 256-dim row per hash bucket: `131072 x 256 = 33,554,432` parameters.
- `mode="mean"` averages the rows of every id belonging to one example (using `offsets`).
- `sparse=True` means gradients are sparse: only the rows touched by a batch get a gradient. That
  is what makes `SparseAdam` possible in the training loop and keeps a CPU epoch fast.
- Initialised uniform in `[-0.05, 0.05]` so the initial bag mean is small and the head starts
  near zero.

This single layer is 99% of the parameters (33.5M of 33.8M) and 99% of the file size.

## Layer 2 — dropout

`nn.Dropout(0.2)` on the pooled vector. It randomly zeroes 20% of the 256 dimensions during
training, which stops the head from relying on a few dimensions and is the main regulariser here
(there is no weight decay on the embedding table).

## Layer 3 — residual block

```python
self.hidden = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
...
h = h + self.hidden(h)
```

- `Linear(256, 256)`: 65,536 weights + 256 biases.
- `GELU`: smooth ReLU variant, no parameters.
- `LayerNorm(256)`: 256 weights + 256 biases, normalises each example's vector.
- The residual add (`h + hidden(h)`) means the block learns a *correction* to the bag mean rather
  than replacing it, which trains more stably at this size.

## Layers 4 and 5 — the two heads

```python
self.head = nn.Linear(dim, n_labels)      # CatID:   256 x 738 = 188,928 + 738
self.cat_head = nn.Linear(dim, n_cats)    # Category: 256 x 81  =  20,736 + 81
```

Two outputs from the same 256-dim vector:

- **CatID head** — the primary task, 738 classes.
- **Category head** — an auxiliary task, 81 classes (the UCS top-level categories). It is trained
  with weight `model.category_loss_weight` (0.3 in the release config). The idea is hierarchical
  smoothing: a description that is clearly "a door" but ambiguous between `DOORGate` and
  `DOORCreak` still gets a useful gradient from the category head, and the category probabilities
  are what drive the `category_fallback` decision
  ([about_calibration-and-decisions.md](about_calibration-and-decisions.md)).

## Parameter count

Measured from the instantiated module (`BowNet(131072, 256, 738, 81, 0.2).named_parameters()`):

| layer | shape | parameters |
|---|---|---:|
| `emb.weight` | 131072 x 256 | 33,554,432 |
| `hidden.0.weight` | 256 x 256 | 65,536 |
| `hidden.0.bias` | 256 | 256 |
| `hidden.2.weight` | 256 | 256 |
| `hidden.2.bias` | 256 | 256 |
| `head.weight` | 738 x 256 | 188,928 |
| `head.bias` | 738 | 738 |
| `cat_head.weight` | 81 x 256 | 20,736 |
| `cat_head.bias` | 81 | 81 |
| **total** | | **33,831,219** |

At 2 bytes per parameter (fp16) that is **67.7 MB**, which is the `weights.safetensors` in the
published package and the reason the wheel-size limit is 90 MB.

## Inference path

`BowClassifier.logits(texts)`:

```python
self.net.eval()                                   # dropout off
for chunk in chunks(texts, 1024):
    ids, offs = self.featurizer.batch(chunk)
    out.append(self.net(ids, offs)[0].numpy())    # CatID logits only
```

It is wrapped in `@torch.no_grad()`, batches at 1024, and returns a `float32` array of shape
`(n, 738)`. The pipeline then divides by the temperature and softmaxes
([about_calibration-and-decisions.md](about_calibration-and-decisions.md)).

## The other two models

### `zeroshot` — `ZeroShotClassifier`

No training. Each CatID is represented by its UCS label document
(`"water surf. <explanation> <synonyms>"`), and the input is scored by cosine similarity to those
documents using TF-IDF over word 1–2-grams and char 3–5-grams (`FeatureUnion` of two
`TfidfVectorizer`s). Similarity is scaled by `similarity_scale` (20.0) into logits. It is the
cold-start floor and a sanity baseline: it needs `scikit-learn` (`[train]`), runs offline, and
knows nothing beyond the UCS text. An `sbert` backend exists behind `[transformers]`.

### `encoder` — `EncoderClassifier`

Optional. A frozen sentence-transformers encoder (default `all-MiniLM-L6-v2`, 384 dims) embeds
each text once; a small `EncoderHead` (dropout, `Linear(384,512)`, GELU, LayerNorm, dropout,
`Linear(512,384)`, residual add) feeds a CatID head and a Category head. The head's output layer is
initialised from embeddings of the UCS label documents, giving a zero-shot prior. It needs
`pip install "ttucs[transformers]"` and a one-time model download, and it is **not trained or
verified** — it is a research path, not part of the published tool.

## Why this architecture

| requirement | how it is met |
|---|---|
| CPU inference in milliseconds | one sparse lookup + a 256-dim MLP |
| small download | 63–68 MB fp16, one matrix |
| offline, no tokeniser files | hashing, no vocabulary |
| robust to STT spelling | character n-grams |
| hierarchical decisions | auxiliary Category head |
| trains in minutes | 33.8M params, sparse optimiser, 15 epochs |

Next: [about_training-loop.md](about_training-loop.md) shows how these weights are learned.
