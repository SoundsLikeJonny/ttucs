# Text features: what the model actually sees

The classifier never sees words. It sees a list of **integer bucket ids**, one per n-gram, produced
by `HashedNgramFeaturizer` in `ttucs/models/bow.py`. This document explains that transformation,
because it is where most of the model's robustness (and most of its blind spots) comes from.

## The input

The description string, e.g. `"metal gate slam heavy"`. Before featurising, it goes through
`ttucs.text.preprocess`:

```python
def preprocess(text):
    text = normalize_transcript(text)          # same normaliser as the parser
    text = re.sub(r"[^\w\s'&-]", " ", text)    # drop punctuation
    return re.sub(r"\s+", " ", text).strip()   # single spaces
```

So the model input is lower-case, digit-normalised, punctuation-free, single-spaced.

## Three families of n-grams

An **n-gram** is a contiguous run of n items. For the text `metal gate slam heavy`:

| family | what it is | examples | count here |
|---|---|---|---|
| word unigrams | each word | `metal`, `gate`, `slam`, `heavy` | 4 |
| word bigrams | each adjacent word pair | `metal gate`, `gate slam`, `slam heavy` | 3 |
| char n-grams | every 3–5 character substring of each word, with `<` `>` boundary markers | `<me`, `met`, `eta`, `tal`, `al>`, ... | 42 |
| **total** | | | **49** |

The counts are exact: `HashedNgramFeaturizer(num_buckets=131072).features("metal gate slam heavy")`
returns 49 ids. `"waves crashing on the rocks"` returns 63.

Why both word and character n-grams?

- **Word n-grams** capture meaning and word order in a small window: "gate slam" is a different
  thing from "gate" and "slam" apart.
- **Character n-grams** capture morphology and survive misspellings: "creak", "creaks", "creaking",
  "creakng" share most of their 3–5-grams, so an STT error costs a little, not everything. This is
  the fastText trick, and it is why the model tolerates "sennheiser" / "senheiser" without a
  vocabulary.

## Hashing: turning strings into bucket ids

A normal bag-of-words model keeps a vocabulary dictionary (`"gate" -> 1234`). That dictionary grows
with the corpus and must be shipped with the model. ttucs instead **hashes** each n-gram into a
fixed number of buckets:

```python
def _h(self, s):
    return zlib.crc32(s.encode("utf-8")) % self.num_buckets
```

- `"w:metal"` -> bucket `5828`
- `"n:metal gate"` -> bucket `89692`
- `"c:<me"` -> some bucket in `[0, 131072)`

The prefixes `w:`, `n:`, `c:` keep the three families from colliding on the same string.

Consequences:

| property | effect |
|---|---|
| fixed size | the model has exactly `num_buckets` embedding rows, whatever the corpus |
| no vocabulary file | nothing to ship, nothing to keep in sync with the tokeniser |
| deterministic | `crc32` is stable across machines and Python versions |
| collisions | two different n-grams can land in the same bucket and share an embedding; with 131,072 buckets and a few hundred thousand distinct n-grams this happens, and the model learns to live with it |
| no OOV | an unseen word still produces char n-grams that hit known buckets |

This is the "hashing trick" (Weinberger et al., 2009), and it is why the published weights are a
single 131,072 x 256 matrix rather than a vocabulary plus a matrix.

## Batching: `collate`

`EmbeddingBag` wants one flat tensor of ids plus offsets marking where each example starts:

```
feats = [[5828, 97141, 18120, ...], [75888, 87721, ...]]     # two descriptions
flat   = [5828, 97141, 18120, ..., 75888, 87721, ...]
offsets= [0, 49]                                             # example 0 starts at 0, 1 at 49
```

`collate` builds both with `np.cumsum` and `np.fromiter`. An empty description becomes the single
id `0`, so the bag is never empty.

## Clause weighting (`head_repeat`)

`head_repeat` (default 1, off in the release config) changes featurisation:

- with `head_repeat == 1`, the whole text is featurised as one run, so n-grams may span a comma
  ("traffic passing, children" gives the bigram `passing children`);
- with `head_repeat > 1`, each comma clause is featurised independently (no n-gram crosses a
  comma) and the first clause's features are repeated into the bag mean, so the headline clause
  weighs more than the context clauses.

Because `EmbeddingBag(mode="mean")` has no per-feature weights, emphasis can only be expressed by
repetition. Round 4 of the retrain experiments found this netted flat, so the release config keeps
`head_repeat: 1` and the published behaviour is bit-identical to before the knob existed.

## What the featuriser does not do

- No stemming, lemmatisation or stop-word removal: "the" is a feature like any other, and the
  model learns that it carries little signal.
- No word order beyond bigrams: "gate metal" and "metal gate" share the unigrams but not the bigram.
- No negation handling: "no wind" and "wind" are nearly the same bag.
- No sentence structure: the bag is order-free once the n-grams are extracted.

Those limits are acceptable for short, telegraphic descriptions ("metal gate slam heavy"), which is
exactly what the tool is fed.

Next: [about_bag-of-words.md](about_bag-of-words.md) explains the idea these features implement.
