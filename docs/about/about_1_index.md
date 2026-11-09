# About ttucs — start here

This folder is a guided tour of the repository, written for someone who has just cloned it and
wants to understand **what it does, how it works, and where each piece of code lives**. It assumes
you know what sound effects and UCS are (you are a recordist or sound designer), but it does not
assume any machine-learning background: the model sections start from "what is a bag of words" and
build up.

The other documents in the repo are the *contract*: [README.md](../../README.md) is the pitch,
[MODEL_CARD.md](../../MODEL_CARD.md) and [DATA_CARD.md](../../DATA_CARD.md) are the honest
statements of what the model was trained on and how well it scores, [CHANGELOG.md](../../CHANGELOG.md)
is the history, [docs/RELEASING.md](../RELEASING.md) is the release runbook and
[AGENTS.md](../../AGENTS.md) is the rulebook for contributors (human or AI). This folder is the
*explanation*: it connects those documents to the actual code.

## The one-paragraph version

A recordist speaks a slate: *"Project Nightfall, location Tofino beach, waves crashing on the rocks,
MKH 8040 ORTF into a MixPre-6, 96k 24 bit, take one."* Speech-to-text turns that into a string.
`ttucs` normalises the string, parses the technical fields out of it, keeps only the *description*
("waves crashing on the rocks"), runs that description through a small neural network that scores
all 738 trainable UCS CatIDs, calibrates the scores, applies a decision rule, and returns the CatID,
a confidence, the top-k alternatives, the parsed slate fields, a UCS filename and keywords. It runs
on a laptop CPU, fully offline, in milliseconds.

## The 20 documents

Read them in order the first time; afterwards they work as reference.

### Part 1 — Orientation

| # | document | what it answers |
|---|---|---|
| 1 | [about_1_index.md](about_1_index.md) | this page: the map |
| 2 | [about_2_what-is-ttucs.md](about_2_what-is-ttucs.md) | the problem, the user, the goals, what "done" means |
| 3 | [about_3_repo-map.md](about_3_repo-map.md) | every directory and file, and which doc explains it |

### Part 2 — The domain

| # | document | what it answers |
|---|---|---|
| 4 | [about_4_ucs-taxonomy.md](about_4_ucs-taxonomy.md) | what UCS is, the 753 CatIDs, why 738 are used, how the CSV is loaded |
| 5 | [about_5_slate-anatomy.md](about_5_slate-anatomy.md) | what a slate contains, what STT does to it, what ttucs extracts |

### Part 3 — Inference, step by step

| # | document | what it answers |
|---|---|---|
| 6 | [about_6_normalisation.md](about_6_normalisation.md) | fillers, spoken numbers, homophones, spelled letters |
| 7 | [about_7_slate-parser.md](about_7_slate-parser.md) | the three-pass parser, spans, cues, leftovers, description assembly |
| 8 | [about_8_gear-lexicon.md](about_8_gear-lexicon.md) | how microphones, recorders and stereo setups are recognised |
| 9 | [about_9_text-features.md](about_9_text-features.md) | the model's input: tokenisation, word n-grams, char n-grams, hashing |
| 10 | [about_10_bag-of-words.md](about_10_bag-of-words.md) | what a bag of words is, why it works here, what it loses |
| 11 | [about_11_model-architecture.md](about_11_model-architecture.md) | every layer of `BowNet`, tensor shapes, parameter counts |
| 12 | [about_12_training-loop.md](about_12_training-loop.md) | batches, losses, optimisers, schedule, early stopping |
| 13 | [about_13_calibration-and-decisions.md](about_13_calibration-and-decisions.md) | temperature scaling, ECE, confident / fallback / unknown |
| 14 | [about_14_pipeline-and-outputs.md](about_14_pipeline-and-outputs.md) | `TTUCS`, `Prediction`, filenames, keywords, the CLI |

### Part 4 — Data and training

| # | document | what it answers |
|---|---|---|
| 15 | [about_15_synthetic-data.md](about_15_synthetic-data.md) | the description generator, banks, object routing, slates |
| 16 | [about_16_datasets-and-splits.md](about_16_datasets-and-splits.md) | label documents, augmentation, grouped splits, caching |
| 17 | [about_17_training-and-config.md](about_17_training-and-config.md) | `ttucs train`, configs, MLflow, reports, provenance |
| 18 | [about_18_evaluation.md](about_18_evaluation.md) | metrics, the three test sets, leakage, the autotrain gate |

### Part 5 — Shipping and maintaining

| # | document | what it answers |
|---|---|---|
| 19 | [about_19_distribution-and-hub.md](about_19_distribution-and-hub.md) | two PyPI packages, model resolution, download, integrity |
| 20 | [about_20_development-and-release.md](about_20_development-and-release.md) | tests, CI, versioning, the release runbook, where to change things |

## The pipeline in one diagram

```
 recorded voice slate text  (speech-to-text output, a string)
          │
          ▼
 ttucs.slate.normalize      "um okay ... mkh eighty forty ... ninety six k"
          │                 -> "mkh 8040 ... 96 k"
          ▼
 ttucs.slate.parser         three passes: hard spans, cue spans, leftovers
          │                 -> SlateFields(project, location, mics, ..., description)
          ▼
 description only           "waves crashing on the rocks"
          │
          ▼
 ttucs.models.bow           hashed word 1-2-gram + char 3-5-gram EmbeddingBag
          │                 -> residual MLP -> CatID head + Category head -> logits
          ▼
 ttucs.pipeline             temperature scaling -> softmax -> decision rule
          │                 -> Prediction(cat_id, confidence, alternatives, filename, ...)
          ▼
 ttucs.taxonomy             UCS filename + keywords
```

The rule that shapes everything: **only the description is classified.** Project, location, gear,
sample rate, take and so on are returned to the user but never reach the model (AGENTS.md rule 5).
That is why you can add a new microphone to the lexicon without retraining anything.

## How to run it while you read

From a checkout with the dev environment installed (see [about_20_development-and-release.md](about_20_development-and-release.md)):

```bash
ttucs predict "um okay so project nightfall uh location tofino beach waves crashing on the rocks mkh eighty forty ortf into a mix pre six ninety six k twenty four bit take one"
ttucs parse "project nightfall location tofino beach mkh 8040 96 k 24 bit take 1"
ttucs info --verify
```

The first command prints, on the published 0.2.0 weights:

```
  description : waves crashing on the rocks
  prediction  : WATRSurf  (WATER-SURF)  conf 0.755  [confident]
  filename    : WATRSurf_Waves Crashing Rocks_NONE_Nightfall_MKH8040-ORTF
```

Every example in these documents was produced by running the code in this repository; where a
number comes from a document rather than a run, the document says so.
