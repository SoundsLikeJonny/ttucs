# What a bag-of-words model is

This document assumes no machine-learning background. It builds from "count the words" to the
exact model ttucs ships, one idea at a time.

## Idea 1: a document is a bag of words

Take the description `"metal gate slam heavy"`. Throw away order and keep the words:

```
{metal, gate, slam, heavy}
```

That is a **bag of words**. It is called a bag because order is gone: `"gate metal heavy slam"`
produces the same bag. For short, telegraphic text this is a surprisingly good representation,
because the *presence* of "gate" and "slam" says most of what there is to say.

## Idea 2: turn the bag into numbers

A computer needs numbers. The classic encoding is a **count vector**: one slot per vocabulary word,
holding how often it appears.

| | metal | gate | slam | heavy | creak | glass |
|---|---:|---:|---:|---:|---:|---:|
| "metal gate slam heavy" | 1 | 1 | 1 | 1 | 0 | 0 |
| "wooden door creak slow" | 0 | 0 | 0 | 0 | 1 | 0 |

The vector is mostly zeros (sparse), and its length is the vocabulary size. A linear classifier on
top of these vectors (multinomial naive Bayes, logistic regression) is the textbook bag-of-words
classifier. It is fast, interpretable, and it works well when the classes are distinguished by
words.

## Idea 3: the problems with plain counts

1. **Vocabulary size.** Real vocabularies are large, and every new word needs a new slot. The
   vocabulary has to be stored with the model and kept in sync with the tokeniser.
2. **Unseen words.** "creakng" (an STT error) is a different slot from "creaking", so the model
   learns nothing from it.
3. **No similarity between words.** "gate" and "door" are orthogonal slots; the model cannot know
   they are related unless it has seen both with the same label.
4. **Order is lost.** "gate metal" and "metal gate" are identical.

## Idea 4: n-grams restore a little order

Instead of single words, use **word n-grams**: unigrams plus bigrams.

```
"metal gate slam heavy" -> {metal, gate, slam, heavy, metal gate, gate slam, slam heavy}
```

Now "gate slam" is its own feature, so the model can learn that this pair means a gate being
struck rather than a gate being opened. ttucs uses unigrams plus bigrams (`word_ngrams: 2`). Going
to trigrams adds features faster than it adds signal on 2–5 word descriptions.

## Idea 5: character n-grams fix spelling

Split each word into overlapping character runs of 3 to 5 characters, with boundary markers:

```
<metal> -> <me, met, eta, tal, al>
```

"creak", "creaks", "creaking" and "creakng" all share `<cr`, `cre`, `rea`, `eak`. So a misspelling
still activates most of the same features. This is the technique fastText popularised for text
classification, and it is why ttucs does not need a spelling corrector.

## Idea 6: hashing removes the vocabulary

Instead of a dictionary, send each n-gram through a hash function into a fixed number of buckets:

```
bucket = crc32("n:metal gate") % 131072   ->  89692
```

Now there is no vocabulary to store, no out-of-vocabulary problem, and the model size is fixed in
advance. The price is **collisions**: two different n-grams can share a bucket. With 131,072
buckets the model tolerates this; the training data teaches it which buckets matter.

## Idea 7: embeddings turn buckets into vectors

Counting is one option; **learning a vector per bucket** is better. Give every bucket a row of
`dim` numbers (256 in the release model), initialised small and random, and learn them by gradient
descent. Then:

- "gate" and "door" can end up with similar vectors if they behave similarly across classes;
- a bucket that only ever appears in door descriptions drifts towards "door-ness".

This is a **bag of embeddings**: the bag of n-grams, each replaced by a learned vector.

## Idea 8: pooling turns many vectors into one

A classifier needs one vector per example. **Mean pooling** averages the vectors of all the
n-grams in the bag:

```
v = (e_metal + e_gate + e_slam + e_heavy + e_metal_gate + ... ) / N
```

`torch.nn.EmbeddingBag(mode="mean")` does exactly this in one fused, fast operation: it takes the
flat id list and the offsets, looks up every row, and averages per example. That is the whole
"bag" in ttucs: `EmbeddingBag(131072, 256, mode="mean")`.

Because it is a mean, a long description is not automatically louder than a short one, and
repeating a feature (as `head_repeat` does) increases its weight.

## Idea 9: a small network on top

The pooled vector is passed through a residual block and two linear heads:

```
v (256) -> dropout -> Linear(256,256) -> GELU -> LayerNorm -> + v -> Linear(256, 738)  CatID logits
                                                              -> Linear(256, 81)   Category logits
```

That is the whole model. It is a bag-of-embeddings classifier with a two-layer head, which is why
it trains in minutes on a CPU and why the weights are 63 MB rather than hundreds.

## What the bag loses, and why that is acceptable here

| lost | consequence | why it is fine for ttucs |
|---|---|---|
| word order beyond bigrams | "gate metal" ~ "metal gate" | descriptions are short and telegraphic |
| negation | "no wind" ~ "wind" | slates rarely negate |
| syntax | "the door that the gate hit" is a bag | slates do not use complex syntax |
| long-range context | a word at the end weighs the same as one at the start | descriptions are 2–8 words |
| compositionality | "small gate" and "large gate" share "gate" | the CatID is usually about the source, not the size |

The alternative — a transformer encoder — is available as `models/encoder.py` (frozen MiniLM plus
an MLP head) but is optional, needs a download, and is not trained or verified. The bag model wins
on the project's real constraint: `pip install`, offline, CPU, small.

## Where to see it in code

| concept | code |
|---|---|
| n-grams and hashing | `HashedNgramFeaturizer._legacy_features`, `_h` |
| batching | `HashedNgramFeaturizer.collate` |
| bag of embeddings | `BowNet.emb = nn.EmbeddingBag(num_buckets, dim, mode="mean", sparse=True)` |
| pooling + head | `BowNet.forward` |
| training | `BowClassifier.fit` -> `train_loop` |

Next: [about_11_model-architecture.md](about_11_model-architecture.md) walks through every layer with
shapes and parameter counts.
