# The gear lexicon

`ttucs.slate.gear` is a table of regular expressions that recognise microphones, microphone types,
recorders and stereo techniques in a normalised transcript, and map each match to a **canonical
display name**. It is the largest single table in the parser and the easiest part of the tool to
extend.

## The data structure

```python
@dataclass(frozen=True)
class GearPattern:
    kind: str                      # "mic" | "mic_type" | "recorder" | "setup"
    regex: re.Pattern[str]
    fmt: Callable[[re.Match[str]], str]
```

`_p(kind, pattern, fmt)` compiles a pattern; `fmt` is either a constant string or a lambda that
builds the canonical name from the match groups. `ALL_GEAR = MICS + MIC_TYPES + RECORDERS + SETUPS`
is what the parser iterates.

Current size: **57 patterns** — 20 microphones, 7 mic types, 16 recorders, 14 setups.

## Why canonical names

The same microphone is spoken a dozen ways: "mkh 416", "m k h four sixteen", "sennheiser 416",
"the 416". All of them should produce one string, `Sennheiser MKH 416`, so that library metadata
and filenames are consistent. The formatter does that:

```python
_p("mic", r"\b(?:sennheiser\s+)?mkh\s?(\d{2,4})\b", lambda m: f"Sennheiser MKH {m.group(1)}")
```

`"mkh 8040"` and `"sennheiser mkh 8040"` both give `Sennheiser MKH 8040`.

## What each group covers

### Microphones (`kind="mic"`)

| brand | patterns cover |
|---|---|
| Sennheiser | MKH 2–4 digit capsules (`mkh 416`, `mkh 8040`), `mk 3xx/5xx` with the brand, ME/MKE, Ambeo VR |
| Sennheiser (STT damage) | `mk 8040` — the "h" dropped; 3–4 digit numbers are unambiguous because Schoeps capsules are 1–2 digits |
| Schoeps | CMC bodies, MK capsules, CMIT, CCM; bare `mk 4 / 8 / 21 / 22 / 41` and `cmc 6 / cmit 5u` |
| Rode | NTG 1–5, NT-SF1, NT4, NT5, NT1 |
| DPA | 4-digit models (`dpa 4060`, `dpa 4006`) |
| Neumann | KMR 81/82, U 87, KM 183/184, RSM 191, TLM, KM 1xx |
| Shure | SM57/58, SM7, VP88, Beta |
| Audio-Technica | AT / BP 4xxx, AT875R, AT8022 |
| Sanken | CS-3e, CO-100K, CUB-01, COS-11D |
| LOM | Usi / Usi Pro, Geofon, Elektrosluch, MikroUsi |
| others | Aquarian H2a, JrF series/hydrophone/contact, Sony ECM, Zoom SSH-6/MSH-6/XYH-6 |

### Microphone types (`kind="mic_type"`)

When no model is named but the *kind* is: hydrophone, contact mic / piezo, geophone, lavalier,
shotgun, binaural, coil pickup. These land in `SlateFields.mic_types`, not `microphones`.

### Recorders (`kind="recorder"`)

Sound Devices MixPre-3/6/10 (and II), 6xx/7xx/8xx, Scorpio, A20; Zoom F/H/M/UAC; Tascam DR/HD-P2/
Portacapture/X/Frontier; Nagra; Sony PCM; Roland R-05/07/26/88; Zaxcom; Aaton Cantar; iPhone/iPad/
phone. Some patterns use a lookahead (`(?=.*\brecorder\b)`) so a bare "633" is only a recorder when
the word "recorder" appears somewhere in the slate.

### Setups (`kind="setup"`)

Double MS, MS, XY, ORTF, NOS, Blumlein, Decca Tree, Spaced Pair (AB) — including the spoken
"dual a b" / "a b stereo" forms — Quad, Ambisonic/FOA, Dual Mono, Surround/5.1, Stereo, Mono.

## How the parser uses it

In `_hard_spans`, every `ALL_GEAR` pattern is run over the normalised string and each match becomes
a span with **priority 10**, the highest. That means a gear match beats a generic number match
inside it ("MKH 416" is not a take number) and beats a cue value ("recording at 96 k" does not
become a location). The matched text is masked out before the description is assembled, so gear
never leaks into the description.

## Extending it

Adding a microphone is a one-line change and needs **no retraining**, because slate fields are not
model inputs:

```python
_p("mic", r"\b(?:sonorous\s+)?obj\s?(\d{3})\b", lambda m: f"Sonorous Objects SO.{m.group(1)}"),
```

Rules to follow:

1. Write the pattern against **normalised** text: lower-case, digits, joined letters. Test with
   `ttucs parse "..."` on the raw spoken form, not on your pattern.
2. Anchor with `\b` and make the brand optional (`(?:brand\s+)?`) so both "mkh 416" and
   "sennheiser mkh 416" match.
3. Use a lookahead when a bare number is ambiguous (see the Sound Devices 6xx/7xx patterns).
4. Put more specific patterns before more general ones in the list; the parser resolves overlaps by
   priority then length, but a specific pattern that never matches is useless.
5. Add a test to `tests/test_parser.py` with the spoken form and the expected canonical name.
6. Keep it to gear a recordist would actually slate. The lexicon is not a catalogue.

## Known limits

- Brand-only mentions ("i'm using a sennheiser") are recognised as a brand fragment and kept out of
  the description, but they do not produce a microphone model.
- The bare `mk <1-2 digits>` form is only Schoeps when the brand is present; without it, "mk 4" is
  ambiguous and is left alone.
- Recorder patterns with lookaheads can miss when the word "recorder" comes after a clause break
  that the normaliser preserved.

Next: [about_9_text-features.md](about_9_text-features.md) leaves the parser and enters the model.
