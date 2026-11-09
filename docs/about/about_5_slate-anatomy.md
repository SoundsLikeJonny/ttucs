# Anatomy of a slate

A slate is the spoken note a recordist makes before or after a take. This document describes what
one contains, what speech-to-text does to it, and exactly which fields ttucs extracts from it. It
is the input contract for everything else in the tool.

## What a recordist says

A typical slate, spoken in one breath:

> "Um, okay, so project Nightfall, uh, location Tofino beach, waves crashing on the rocks, MKH
> eighty forty ORTF into a MixPre six, ninety six k twenty four bit, take one."

It mixes five kinds of information:

| kind | example | who cares |
|---|---|---|
| session identity | project Nightfall, location Tofino beach, recordist, date | the library's SourceID, CreatorID, folders |
| **what was recorded** | waves crashing on the rocks | **the classifier** |
| gear | MKH 8040, ORTF, MixPre-6 | the library's notes, the filename's UserData |
| format | 96 k, 24 bit | the library's notes |
| take / perspective | take one, exterior | the library's notes, the filename's UserCategory |

Only the third row is a description of the sound. The rest is metadata *about* the recording.

## What speech-to-text does to it

Any STT engine (Whisper, Deepgram, a phone dictation app) produces a string with characteristic
damage:

- **fillers**: "um", "uh", "okay", "so", "like";
- **spoken numbers**: "ninety six k", "twenty four bit", "take one", "eighty forty";
- **spelled letters**: "m k h", "o r t f", "d p a", "x y";
- **homophones and brand misspellings**: "mix pray" for MixPre, "senheiser" for Sennheiser,
  "shoeps" for Schoeps, "mikes" for mics;
- **no punctuation or inconsistent punctuation**;
- **run-together words**: "mixpre", "sounddevice".

ttucs is designed for this input. It never assumes clean text, and it never assumes the fields
arrive in a fixed order.

## The `SlateFields` record

`ttucs.slate.parser.SlateFields` is a dataclass with these fields:

| field | type | example | notes |
|---|---|---|---|
| `raw_text` | str | the original string | kept for traceability |
| `normalized_text` | str | `"project nightfall location tofino beach ..."` | after `normalize_transcript` |
| `description` | str | `"waves crashing on the rocks"` | **the only field the classifier sees** |
| `project` | str | `"Nightfall"` | title-cased |
| `location` | str | `"Tofino Beach"` | title-cased |
| `creator` | str | `"Jon Evans"` | recordist name |
| `date` | str | `"2026-11-09"` | ISO, from spoken dates |
| `microphones` | list[str] | `["Sennheiser MKH 8040"]` | canonical gear names |
| `mic_types` | list[str] | `["Condenser"]` | when no model is named |
| `mic_setup` | list[str] | `["ORTF"]` | stereo technique |
| `recorder` | str | `"Sound Devices MixPre-6"` | canonical |
| `sample_rate` | int | `96000` | Hz |
| `bit_depth` | str | `"24"` | `"24"`, `"32 float"`, ... |
| `take` | str | `"1"` | digits |
| `perspective` | list[str] | `["EXT"]` | `INT`/`EXT` only |
| `user_data` | str | `"MKH8040-ORTF"` | free-form tail |
| `unparsed` | list[str] | `["some leftover words"]` | what the parser could not place |

`to_dict()` returns all of them; the CLI hides empty ones unless `--verbose` is passed.

## A worked example

Input (one line, as STT would emit it):

```
um okay so project nightfall uh location tofino beach waves crashing on the rocks mkh eighty forty ortf into a mix pre six ninety six k twenty four bit take one
```

`ttucs parse` on this string returns (abridged):

```json
{
  "normalized_text": "project nightfall location tofino beach waves crashing on the rocks mkh 8040 ortf into a mixpre 6 96 k 24 bit take 1",
  "description": "waves crashing on the rocks",
  "project": "Nightfall",
  "location": "Tofino Beach",
  "microphones": ["Sennheiser MKH 8040"],
  "mic_setup": ["ORTF"],
  "recorder": "Sound Devices MixPre-6",
  "sample_rate": 96000,
  "bit_depth": "24",
  "take": "1"
}
```

Note what happened: fillers vanished, "eighty forty" became `8040`, "mix pre six" became
`mixpre 6` and then the canonical `Sound Devices MixPre-6`, and the description kept exactly the
content clause. The classifier then sees only `"waves crashing on the rocks"` and returns
`WATRSurf` with confidence 0.755.

## Slates without fields

Not every input is a full slate. A user may paste a library description ("metal gate slam heavy")
or a bare phrase ("a quiet room hum"). The parser then finds no fields and treats the whole string
as the description. This is why `ttucs predict` works on both slates and plain descriptions, and
why the evaluation sets contain both (`examples/realistic_slates.jsonl` are full slates,
`examples/dev_descriptions.jsonl` are descriptions).

## Multi-clause slates

Real ambience slates often list several things:

> "recording traffic passing by, children playing in a nearby park, wind in the trees"

The parser joins the content clauses that follow a description cue into one description, so the
classifier sees all of them. This is the multi-clause behaviour added in PR #12
([about_7_slate-parser.md](about_7_slate-parser.md)).

## Why the split matters

Keeping the description separate from the fields is the central design decision of the tool:

- the classifier only ever sees short, content-bearing text, which is what the synthetic training
  data looks like;
- the parser can be improved (new gear, new cues) without retraining the model;
- the model cannot cheat by learning "MKH 416 means DOORCreak" from a handful of co-occurrences;
- a wrong gear parse cannot corrupt the classification, and a wrong classification cannot corrupt
  the gear fields.

Next: [about_6_normalisation.md](about_6_normalisation.md) shows the first transformation applied to
the raw string.
