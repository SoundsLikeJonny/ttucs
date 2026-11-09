# Normalisation: making speech-to-text text predictable

`ttucs.slate.normalize.normalize_transcript` is the first transformation applied to every input.
It turns the messy string an STT engine produced into a canonical, lower-case, single-spaced form
that the parser and the model can rely on. It is pure regex and lookup: no model, no network, and
it is deterministic.

## Why a separate step

The parser needs to recognise "96 k" as a sample rate, "mkh 8040" as a microphone and "take 1" as a
take number. If it had to handle "ninety six kay", "m k h eighty forty" and "take one" as well, every
rule in the parser would be three times longer. Normalisation centralises that variability in one
place, and the model benefits too: `ttucs.text.preprocess` calls the same function, so the model
sees "96 k" whether the recordist said "ninety six k" or "96k".

## The steps, in order

`normalize_transcript` applies, in this order:

1. **Unicode and whitespace cleanup** — normalise quotes and dashes, collapse whitespace.
2. **Fillers** — remove "um", "uh", "erm", "hmm", "okay", "so", "like" when they are standalone
   words (never inside a word).
3. **Spelled single letters** — a run of separated single letters is joined:
   `"m k h"` -> `"mkh"`, `"o r t f"` -> `"ortf"`, `"d p a"` -> `"dpa"`, `"x y"` -> `"xy"`.
   The `_SINGLE_LETTERS` regex matches runs of 2+ single letters that are not "a" or "i".
4. **Homophones and brand variants** (`REPLACEMENTS`, an ordered list of regex pairs) — for
   example `"mix pre"` / `"mix pray"` -> `"mixpre"`, `"senheiser"` / `"senhizer"` -> `"sennheiser"`,
   `"shoeps"` -> `"schoeps"`, `"mikes"` -> `"mics"`, `"kilohertz"` / `"k hz"` / `"kay"` -> `"k"`,
   `"bits"` -> `"bit"`, `"sound device"` -> `"sound devices"`, `"task cam"` -> `"tascam"`.
5. **Spoken numbers** — number words become digits ("ninety six" -> `96`, "twenty four" -> `24`,
   "take one" -> `take 1`), and digit pairs spoken as two-digit groups are joined for gear models
   ("eighty forty" -> `8040`, "four sixteen" -> `416`).
6. **Decimal sample rates** — "forty four point one" -> `44.1`, "eighty eight point two" -> `88.2`.
7. **Final whitespace collapse** and lower-casing.

## What it produces

Running the normaliser on the example slate:

```
in : um okay so project nightfall uh location tofino beach waves crashing on the rocks mkh eighty forty ortf into a mix pre six ninety six k twenty four bit take one
out: project nightfall location tofino beach waves crashing on the rocks mkh 8040 ortf into a mixpre 6 96 k 24 bit take 1
```

and on the cases that PR #12 fixed:

```
in : m k h four sixteen            out: mkh 416
in : dual a b pair                 out: dual ab pair
in : senheiser mkh four sixteen    out: sennheiser mkh 416
in : forty four point one k        out: 44.1 k
in : i used a shoeps cmc six mk four  out: i used a schoeps cmc 6 mk 4
```

## Design rules

- **Idempotent.** Running the normaliser twice gives the same string; the parser and the model
  both call it, sometimes on already-normalised text.
- **Conservative.** It only rewrites patterns it is sure about. "kay" becomes "k" only in the
  sample-rate context; "ab" is only produced from "a b" when followed by a stereo word.
- **Ordered.** `REPLACEMENTS` is a list, not a dict, because later rules depend on earlier ones
  ("mix pre six" must become "mixpre 6" before the number rule runs).
- **No domain knowledge beyond gear names.** It does not know what a door is; that is the model's
  job.

## Where it is called

| caller | call |
|---|---|
| `SlateParser.parse` | first line: `normalized = normalize_transcript(text)` |
| `ttucs.text.preprocess` | `normalize_transcript` then strip punctuation, lower-case, collapse spaces |
| `ttucs.datasets.build.slate_augment` | indirectly, via `parse_slate` on generated slates |

## Tests

`tests/test_normalize.py` (4 tests) covers fillers, spelled letters, homophones and numbers. When
you add a replacement, add a test with the exact STT spelling you saw in the wild, and run
`python scripts/check_leakage.py` afterwards if you touched a bank (normalisation itself cannot
leak, but it is cheap to check).

Next: [about_7_slate-parser.md](about_7_slate-parser.md) consumes this output.
