# The slate parser

`ttucs.slate.parser` turns a normalised transcript into a `SlateFields` record: the technical
fields plus the **description**, which is the only part the classifier sees. It is rule-based
(regex and small tables), deterministic, and has no model of its own.

## Why rules, not a model

- The fields are a closed, well-defined set with unambiguous markers ("take", "96 k", "24 bit",
  "project", a known microphone name). Regex is exact, explainable and free.
- A learned parser would need labelled slates, which do not exist in redistributable form.
- Rules can be fixed the same day a user reports a miss, without retraining anything.

The cost is that unusual phrasing can be missed; the parser's job is to be predictable and to fail
by leaving text in `unparsed` rather than by inventing a field.

## The three passes

`SlateParser.parse(text)`:

```python
norm  = normalize_transcript(text)          # pass 0: normalisation
spans = self._hard_spans(norm, out)         # pass 1: technical tokens
spans += self._cue_spans(norm, spans, out)  # pass 2: cue-introduced free text
out.perspective = self._perspectives(norm)
... description assembly from leftovers ... # pass 3
```

A `_Span` is `(start, end, field, value, priority)` — a character range of the normalised string
that belongs to a field.

### Pass 1 — hard spans

Unambiguous technical tokens, matched by regex and by the gear lexicon
([about_gear-lexicon.md](about_gear-lexicon.md)):

| field | pattern (abridged) | priority |
|---|---|---:|
| `sample_rate` | `44.1 / 48 / 88.2 / 96 / 176.4 / 192 / 384` followed by `k` or `000` | 9 |
| `bit_depth` | `16 / 24 / 32` + `bit`, optional `float` | 9 |
| `take` | `take (number \| no. \| #)? <digits>` | 9 |
| `scene`, `slate` | same shape | 9 |
| `date` | ISO, `d/m/y`, "November 9th 2026", "9th of November" | 8 |
| gear (`mic`, `mic_type`, `setup`, `recorder`) | the lexicon | 10 |
| `perspective` | "close up", "medium perspective", "distant", "onboard" | 7 |

Overlaps are resolved by `_resolve_overlaps`: highest priority first, then longest, then earliest;
a span is kept only if it does not overlap an already-kept one. That is why "MKH 8040" wins over a
generic number match inside it.

Values are written into `SlateFields` as they are found: microphones and setups append (deduped),
recorder/sample rate/bit depth/take/date take the first hit.

### Pass 2 — cue spans

Cues introduce free text whose value runs until the next span, cue or clause boundary:

| cue field | cue words (abridged) |
|---|---|
| `project` | "project", "project name", "production", "for the project/show/game/film", "library" |
| `location` | "location", "loc", "on location at/in", "recorded at/in", "we're at/in", "place", "venue" |
| `creator` | "recordist", "recorded by", "my name is", "slated by", "engineer" |
| `notes` | "notes", "comment", "fyi" |
| `description` | "recording", "this is", "sound of", "description", "subject", "capturing", "fx", "source" |

Details that matter:

- A cue that starts inside a hard span is ignored (so "recording at 96 k" does not become a
  location).
- Nested cues are dropped; when two cues start at the same character the higher priority wins
  (`project`/`location`/`creator` 6 > `notes` 5 > `description` 4).
- The value ends at the next boundary: any hard span start, any cue start, any clause end
  (`, ; ! ? .` or ` - `), or the end of the string.
- `_trim_value` shortens run-on values in unpunctuated STT output: a location ends after its head
  noun (`beach`, `forest`, `street`, ...) when at least two words follow; a project is capped at
  four words and a creator at two when the transcript has no clause punctuation at all.
- Values are stripped of leading/trailing glue words (`a`, `the`, `into`, `using`, `i'm`, ...) and
  title-cased (`_title`, which keeps small words lower-case after the first word).

### Pass 3 — description assembly

The spans are masked out of the string, and what remains is split into clauses at the same
boundaries. Each leftover clause is stripped of glue; clauses with no content words are dropped.
Then:

- **If there is a description cue**, the description is the cue's value plus every content-bearing
  clause that *follows* it (and is not perspective-only or a bare gear-brand fragment). This is the
  multi-clause behaviour: "recording traffic passing by, children playing in a nearby park, wind in
  the trees" keeps all three.
- **If there is no description cue**, every content-bearing leftover clause is joined in spoken
  order: "ambience nearby traffic, crossing sign, children playing, wind" keeps all four.
- Perspective-only words ("exterior", "interior", "close up") and brand fragments ("sennheiser")
  are never joined into the description; they stay in `perspective` or `unparsed`.
- Perspective meta phrases are removed from the description text, glue is stripped again, and if
  nothing is left the whole normalised string becomes the description (better to classify
  something than nothing).

Absorbed clauses are removed from `unparsed`; the rest are reported there.

## Worked examples

| input | description | fields |
|---|---|---|
| `recording traffic passing by, children playing in a nearby park, wind in the trees` | `traffic passing, children playing in a nearby park, wind in the trees` | — |
| `ambience nearby traffic, crossing sign, children playing, wind` | `ambience nearby traffic, crossing sign, children playing, wind` | — |
| `project thick as thieves, location squamish forest, recording a wooden door creak slow, mkh four sixteen into a mix pre six, ninety six k twenty four bit, take three` | `wooden door creak slow` | project `Thick As Thieves`, location `Squamish Forest`, mic `Sennheiser MKH 416`, recorder `Sound Devices MixPre-6`, 96000, 24, take 3 |
| `location tofino beach waves crashing on the rocks` | `waves crashing on the rocks` | location `Tofino Beach` |
| `exterior a metal gate slam heavy` | `exterior a metal gate slam heavy` | perspective `EXT` |

(The last row is a known rough edge: "exterior" is only removed from the description when it is
matched as a perspective *meta* phrase such as "close up"; a bare "exterior" at the start of a
cue-less slate stays in the description. It is recorded here rather than hidden.)

## Where the parser is used

| caller | use |
|---|---|
| `TTUCS.predict_batch` | parse every input, classify `slate.description` |
| `ttucs parse` (CLI) | print the fields as JSON |
| `datasets.build.slate_augment` | parse generated slates so training text matches inference text |
| `evaluate.parser_field_accuracy` | score fields against `SlateGenerator` ground truth |

## Tests and how to change it

`tests/test_parser.py` (13 tests) covers cues, trimming, multi-clause joining, gear fragments,
perspectives and dates. When you change the parser:

1. add a test with the exact transcript that failed;
2. run `pytest -q tests/test_parser.py`;
3. run `python scripts/check_leakage.py` if you touched any bank;
4. remember that parser changes ship with `ttucs` and need no retrain — but if the parser now feeds
   the classifier *more* text (as multi-clause did), the training data should match, which is why
   `data.synthetic.multi_clause_prob` exists ([about_synthetic-data.md](about_synthetic-data.md)).

Next: [about_gear-lexicon.md](about_gear-lexicon.md) explains the largest table the parser uses.
