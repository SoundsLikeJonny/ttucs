# The UCS taxonomy: the label space

Everything ttucs predicts is a CatID from the [Universal Category System](https://universalcategorysystem.com/).
This document explains what that list contains, how it is shipped, how it is loaded, and the few
places where the code deliberately deviates from the raw spreadsheet.

## What UCS is

UCS is a public, community-maintained taxonomy for sound effects libraries. It is a flat list of
**Category / SubCategory** pairs, each with a short identifier called a **CatID**. Examples:

| Category | SubCategory | CatID | CatShort | Explanation (abridged) |
|---|---|---|---|---|
| WATER | SURF | `WATRSurf` | Surf | Waves breaking, surf on a beach |
| DOORS | GATE | `DOORGate` | Gate | A gate opening, closing or being struck |
| GLASS | BREAK | `GLASBrk` | Break | Glass breaking, shattering |
| CROWDS | CHILDREN | `CRWDChld` | Children | Crowds of children |

The CatID is the first block of a UCS filename, which is why it is the thing ttucs predicts:

```
CatID(-UserCategory)_(VendorCategory-)FXName_CreatorID_SourceID(_UserData)
WATRSurf_Waves Crashing Rocks_NONE_Nightfall_MKH8040-ORTF
```

UCS also defines a `MISC` subcategory per category ("this category, unspecified") and a
`GENERAL` subcategory in some categories. ttucs uses `MISC` for its `category_fallback` decision
([about_13_calibration-and-decisions.md](about_13_calibration-and-decisions.md)).

## The numbers

| quantity | value | where it comes from |
|---|---:|---|
| CatIDs in UCS v8.2.1 | 753 | `len(UCSTaxonomy.load())` |
| top-level categories | 82 | `len(UCSTaxonomy.load().categories)` |
| trainable CatIDs | 738 | 753 minus the `ARCHIVED` workflow category |
| model output classes | 738 | `config.json` -> `labels` |
| Category head classes | 81 | the categories that remain after excluding `ARCHIVED` |
| CatIDs with hand-written phrases | 494 | `resources/catid_phrases.yaml` |

The 753 vs 738 difference matters: `ARCHIVED` describes a library *workflow* state (stems, ADR,
work in progress), not a sound. Training a classifier to predict "this recording is archived"
from its description would be nonsense, so `WORKFLOW_CATEGORIES = frozenset({"ARCHIVED"})` is
excluded by `UCSTaxonomy.trainable()` and by `data.exclude_categories` in every config.

## Where the list lives

`src/ttucs/resources/ucs_v8.2.1.csv` ships as package data (the `*.csv` and `*.yaml` globs in
`pyproject.toml`). Columns: `Category, SubCategory, CatID, CatShort, Explanations, Synonyms`.
`ucs_v8.2.1_top_level.csv` holds the one-line explanation of each top-level category.

The CSV is generated from the official spreadsheets by `scripts/build_taxonomy.py`:

```bash
python scripts/build_taxonomy.py "UCS v8.2.1 Full List.xlsx" "UCS v8.2.1 Top Level Categories.xlsx" --version 8.2.1
```

It reads the sheet with `header=2`, keeps the first six columns, drops rows without a CatID,
asserts that no CatID is duplicated, and writes the CSV. The spreadsheets themselves are never
committed; only the derived CSV is.

## How it is loaded

`UCSTaxonomy.load(version)` returns a cached (`lru_cache(maxsize=4)`) object built from the CSV:

- `entries`: a list of frozen `UCSEntry` dataclasses (`cat_id`, `category`, `subcategory`,
  `cat_short`, `explanation`, `synonyms` tuple);
- `_by_id`, `_by_id_lower`, `_by_full`: three lookup dicts for exact, case-insensitive and
  `CATEGORY-SUBCATEGORY` lookups.

`resolve(cat_id)` tries, in order: the alias table, the exact CatID, the lower-cased CatID, and
`CATEGORY-SUBCATEGORY` (so `"WATER-SURF"` resolves to `WATRSurf`). The alias table exists because
real library metadata contains misspellings that are not in the official list:

```python
CATID_ALIASES = {"MACHOffi": "MACHOffc", "WOODHandl": "WOODHndl"}
```

`UCSEntry.category_full` gives `"WATER-SURF"`, the form UCS-aware databases use.
`UCSEntry.label_document()` gives a natural-language description of the class
(`"water surf. <explanation> <synonyms>"`), used by the zero-shot model and as extra training text.

## How the taxonomy is used downstream

| consumer | what it takes | why |
|---|---|---|
| `train.py` | `taxonomy.trainable(exclude)` | the label list and the Category list of the model |
| `synth/generator.py` | explanations, synonyms, subcategory names | to write synthetic descriptions per CatID |
| `datasets/build.py` | `label_document()` | extra training text for rare classes |
| `models/zeroshot.py` | `label_document()` | TF-IDF similarity without training |
| `pipeline.py` | `UCSEntry` per label, the `MISC` map | to fill `Prediction` fields and the fallback |
| `taxonomy.UCSFilename` | CatID, `NONE` defaults | to build and parse UCS filenames |

The model's label order is the taxonomy order filtered to trainable entries. That order is stored
in the model's `config.json` (`labels`), so a model is self-describing: `TTUCS.from_pretrained`
loads the taxonomy version recorded in the model and maps output index *i* to `labels[i]`.

## Filenames

`UCSFilename` is a dataclass that both builds and parses the UCS filename convention:

```python
UCSFilename(cat_id="WATRSurf", fx_name="Waves Crashing Rocks", creator_id="NONE",
            source_id="Nightfall", user_category="EXT", user_data="MKH8040-ORTF").build()
# 'WATRSurf-EXT_Waves Crashing Rocks_NONE_Nightfall_MKH8040-ORTF'
```

`build()` joins blocks with `_`, substitutes `NONE` for an empty CreatorID or SourceID, and
`parse_ucs_filename()` reverses it, raising `ValueError` for fewer than four blocks or (when a
taxonomy is passed) an unknown CatID. `sanitize_block()` strips characters that are illegal in
filenames on Windows, so a suggested name is always safe to write to disk.

## Changing the taxonomy

- A new UCS version means a new CSV (`ucs_v<version>.csv`), a new `UCS_VERSION` constant, and a
  retrain: the label list changes, so the weights change, so `ttucs-model-bow` gets a new version.
- Adding an alias is free: it only affects lookup, not the model.
- Never edit the CSV by hand; regenerate it with `scripts/build_taxonomy.py` so the diff is
  traceable to the official list.
