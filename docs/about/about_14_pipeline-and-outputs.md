# The pipeline, the outputs and the CLI

`src/ttucs/pipeline.py` is the public API: it joins the parser, the model, the taxonomy and the text
utilities into one object, `TTUCS`, and one result type, `Prediction`. This document follows a
single call from `predict("...")` to the printed output.

## `TTUCS`: construction

```python
pipe = TTUCS.from_pretrained()            # default: the "bow" model, offline-first
pipe = TTUCS.from_pretrained("models/ttucs-bow-release")
pipe = TTUCS.from_pretrained(unknown_threshold=0.3)
pipe = TTUCS.from_mlflow("models:/ttucs/1")
```

`from_pretrained` does four things:

1. `hub.resolve_model(name)` finds a directory (explicit path, installed `ttucs-model-bow` package,
   cache, or a verified download — [about_19_distribution-and-hub.md](about_19_distribution-and-hub.md));
2. `hub.check_compatibility` refuses a model whose format or `ttucs` version range does not match;
3. `UCSModel.load(path)` rebuilds the model class from `config.json` and loads the weights;
4. it reads `temperature`, `inference.unknown_threshold` and `inference.category_threshold` from the
   model's own metadata, so a model carries its own inference settings.

`TTUCS.__init__` then caches two lookups: `_entries` (the `UCSEntry` for each model label, in label
order) and `_misc` (the MISC CatID per category).

## `predict_batch`: the whole path

```python
def predict_batch(self, texts, top_k=5, creator_id=None, source_id=None):
    slates = [self.parser.parse(t) for t in texts]
    probs  = self.probabilities([s.description or s.normalized_text or " " for s in slates])
    return [self._build(t, s, p, top_k, creator_id, source_id) for t, s, p in zip(...)]
```

1. **Parse** every input into `SlateFields`.
2. **Classify only the description.** If the parser found no description, the normalised text is
   used; if that is empty too, a single space (which featurises to bucket 0).
3. **Build** one `Prediction` per input.

`predict(text, ...)` is `predict_batch([text], ...)[0]`.

## `_build`: from probabilities to a `Prediction`

| step | code | result |
|---|---|---|
| rank CatIDs | `np.argsort(-p)[:top_k]` | `alternatives`: `Candidate(cat_id, category, subcategory, cat_short, confidence)` |
| rank categories | `np.bincount(label_to_cat, weights=p)` | `categories`: `[{category, confidence}]` |
| decide | the threshold rule | `decision`, possibly a MISC `cat_id` |
| FX name | `make_fx_name(description)` | `"Metal Gate Slam Heavy"` |
| filename | `_filename(...)` | a `UCSFilename` |
| keywords | `make_keywords(description, [subcategory, category])` | `['Metal', 'Gate', 'Slam', 'Heavy', 'Doors']` |

## The `Prediction` object

`Prediction` is a dataclass, JSON-serialisable via `to_dict()` / `to_json()`:

```python
p = pipe.predict("metal gate slam heavy", top_k=3)
p.cat_id            # 'DOORGate'
p.category_full     # 'DOORS-GATE'
p.confidence        # 0.948251
p.decision          # 'confident'
p.alternatives      # [Candidate('DOORGate', 0.948), Candidate('DOORCreak', 0.026), ...]
p.categories        # [{'category': 'DOORS', 'confidence': 0.979}, ...]
p.filename          # 'DOORGate_Metal Gate Slam Heavy_NONE_NONE'
p.is_unknown        # False
```

## Filenames

`_filename` builds a `UCSFilename`:

| block | source | fallback |
|---|---|---|
| CatID | the chosen entry | — |
| UserCategory | `INT`/`EXT` from `slate.perspective` | omitted |
| FXName | `make_fx_name(description)` | `"Untitled"` |
| CreatorID | `initials(slate.creator)` | `creator_id` argument, then `"NONE"` |
| SourceID | `abbreviate(slate.project)` | `source_id` argument, then `"NONE"` |
| UserData | `slate.user_data` | omitted |

`make_fx_name` keeps content words in order until 25 characters (UCS's recommendation), drops
duplicates, title-cases, and upper-cases short tokens containing digits. `initials("Jon Evans")`
gives `"JE"`; `abbreviate("Thick As Thieves")` gives `"ThickAsThieves"` (initials if longer than 16
characters). `sanitize_block` removes characters illegal on Windows, so the name is always writable.

## Keywords

`make_keywords` takes the content tokens of the description (stop-words removed via `STOPWORDS` in
`text.py`), adds the subcategory and category names, de-duplicates in order, title-cases and keeps
15.

## The module-level API

`src/ttucs/__init__.py` exposes a lazy shortcut:

```python
import ttucs as t
pred = t.predict("project nightfall, location tofino beach, waves crashing on rocks, take 3")
```

The first call builds the default `TTUCS` and caches it in a module global; nothing is loaded at
import time, so `import ttucs` stays instant and offline. `t.predict_batch(...)` works the same way.
For a custom model or inference overrides, use `TTUCS.from_pretrained` directly.

## The CLI

`ttucs` is an `argparse` program (`cli.py`) with a shared `-c/--config` and `-o KEY=VALUE` override
pair for the training commands:

| command | what it does | needs `[train]`? |
|---|---|---|
| `ttucs predict [TEXT ...] [-f FILE] [-m MODEL] [-k K] [--threshold T] [--creator-id ID] [--source-id ID] [--json]` | classify slates or descriptions | no |
| `ttucs parse [TEXT ...] [-f FILE] [--verbose]` | print `SlateFields` as JSON | no |
| `ttucs info [-m MODEL] [--verify] [--json]` | resolved model, versions, offline mode, manifest check | no |
| `ttucs download [--model bow] [--dest DIR] [--url URL] [--sha256 SHA] [--force]` | fetch and verify weights into the cache | no |
| `ttucs generate-data [--out DIR]` | build the splits | yes |
| `ttucs train [-c CONFIG] [--rebuild-data]` | train, calibrate, evaluate, log | yes |
| `ttucs eval -m MODEL [--data FILE ...] [--parse] [--predictions CSV] [--json FILE] [--mlflow]` | evaluate | yes |
| `ttucs report [--out DIR]` | class-distribution report and plots | yes |

`predict` prints a human-readable block per input (or JSON with `--json`):

```
> um okay so project nightfall uh location tofino beach waves crashing on the rocks mkh eighty forty ortf into a mix pre six ninety six k twenty four bit take one
  description : waves crashing on the rocks
  prediction  : WATRSurf  (WATER-SURF)  conf 0.755  [confident]
  top-k       : WATRSurf 0.755, WATRWav 0.061, ...
  filename    : WATRSurf_Waves Crashing Rocks_NONE_Nightfall_MKH8040-ORTF
  keywords    : Waves, Crashing, Rocks, Surf, Water
  slate       : {"project": "Nightfall", "location": "Tofino Beach", "microphones": ["Sennheiser MKH 8040"], ...}
```

`-f FILE` reads one transcript per line, so a whole session can be processed in one call.

## Error behaviour

- A missing model raises `hub.ModelNotFoundError` with a hint to run `ttucs download` or install
  `ttucs-model-bow`; `ttucs info` exits 1 and prints the error instead of raising.
- Missing training extras raise `MissingExtraError` from `_extras.py` with the exact `pip install`
  line.
- `parse_ucs_filename` raises `ValueError` for malformed names.

Next: [about_15_synthetic-data.md](about_15_synthetic-data.md) starts the data half of the tour.
