# Synthetic training data

There is no redistributable corpus of labelled field-recording descriptions. The project's answer
is to **generate** one: `ttucs/synth/generator.py` writes short, library-style descriptions for
every CatID from the UCS text plus hand-written vocabulary banks, and a second generator wraps them
into spoken slates. Everything is deterministic, offline and LLM-free by default.

## Why synthetic

- **Redistributable.** The UCS list is public; the banks are written for this project. Nothing
  comes from a commercial library (AGENTS.md rule 2).
- **Complete.** Every one of the 738 trainable CatIDs gets the same number of examples, so rare
  classes are not starved.
- **Controllable.** Phrase coverage, multi-clause joins and noise levels are config knobs.
- **Honest about its limits.** Synthetic text comes from the same generator as the training data,
  so synthetic test scores overstate real accuracy. That is why the headline number is the 80
  hand-written slates ([about_18_evaluation.md](about_18_evaluation.md)).

## The vocabulary banks

`resources/synth_banks.yaml` holds:

| key | contents | used for |
|---|---|---|
| `modifiers` | groups of adjectives ("heavy", "slow", "creaky", "distant", ...) | prefixed to a head noun 55% of the time |
| `contexts` | prepositional phrases ("in a garage", "on concrete", "with wind") | appended 15% of the time |
| `subcategory_actions` | verbs per UCS subcategory ("slam", "creak", "shatter") | the action slot for generic subcategories |
| `categories` | per top-level category: `objects` and `actions` | head nouns and verbs when the subcategory has none |

`resources/catid_phrases.yaml` holds **hand-written everyday head phrases for 494 CatIDs**
("weed whacker start up", "creaky door slow", "metal gate slam"). They are used with
`phrase_prob` (0.5) and are the single biggest accuracy lever for real wording: on the 29
realistic slates with no bank-phrase overlap, top-1 went from 0.552 to 0.724 when the bank was
introduced.

## `DescriptionGenerator`

```python
gen = DescriptionGenerator(taxonomy, seed=42, phrase_prob=0.5,
                           multi_clause_prob=0.2, verbose_prob=0.0)
texts = gen.generate(entry, n=300)      # up to n unique descriptions
```

### Synonyms, weighted by distinctiveness

Each CatID's UCS synonyms are filtered to those appearing in at most `max_synonym_df` (25) entries
and weighted `1 / sqrt(df)`. So "Carbon"/"CO2", shared by every AIR subcategory, rarely appears,
while a synonym unique to one class is used often.

### Object routing

Category-level bank objects ("lawn mower", "gate") are assigned to the specific sibling
subcategories whose UCS synonyms or name mention them (`_route_objects`). Routed objects become
extra head nouns for that subcategory; `MISC`/`GENERAL` only get the leftovers. Without this, the
MISC classes absorbed specific sounds ("lawn mower" -> `MACHMisc` instead of `MACHGrdn`).

### Templates

For a specific subcategory, the head noun is drawn from the subcategory word, a distinctive
synonym, or a routed object; the action from the category's action list; and one of seven templates
is filled:

```
{mod} {head} {act}          {head} {act}          {head} {other} {act}
{mod} {head} {other}        {sub} {head} {act} {ctx}
{head} {act} {mod} {ctx}    {other} and {head} {act}
```

Generic subcategories (`BREAK`, `IMPACT`, `CREAK`, `FRICTION`, ...) use the category's objects and
the subcategory's own verbs instead.

### Phrases

With probability `phrase_prob`, the head noun is replaced by a hand-written phrase from
`catid_phrases.yaml` and a shorter template is used, which is what makes output read like library
metadata rather than like a template.

### Explanation fragments

12% of the time a fragment of the UCS Explanation is used (split on punctuation and "or"/"from",
2–9 words, cross-references such as "See also TOOLS-GARDEN" removed). This injects vocabulary that
the banks do not cover.

### Verbose narration (`verbose_prob`, default 0)

Longer sentence-style descriptions: for sources, "the sound of a heavy gate slamming in a garage";
for ambiences, "standing in a parking lot, traffic close by, children farther away". It is
implemented and tested but **off in the release config**: round 3 showed it cost multi-clause dev
accuracy (0.673 -> 0.582).

### Multi-clause joins (`multi_clause_prob`, default 0.2 in the release config)

Pairs of same-class descriptions are joined with a comma ("metal gate slam, creaky hinges") with
probability `multi_clause_prob`. This exists because the parser now feeds the classifier the full
multi-clause description, so the model must see multi-clause text in training. It is the only
retrain change that durably helped (dev multi-clause slice 0.600 -> 0.673, n=55).

### Noise

A word is dropped from 15% of descriptions longer than three words, to stop the model memorising
templates. Verbose sentences are exempt because dropping a connective breaks them.

### Example output

With `seed=42` and `multi_clause_prob=0.2`, `DOORGate` produces:

```
'garden gate swing weak'
'gate cattle open'
'gate cattle gate, farm barrier'
'rhythmic gate barricade'
'farm and cattle squeak'
```

Note the last two: synthetic text is sometimes odd ("farm and cattle squeak" for a gate). That is
the honest cost of template generation, and it is why real-slate accuracy is the number that
matters.

## `SlateGenerator`

`SlateGenerator(seed, p_field=0.55, p_fillers=0.35, p_no_punct=0.5).wrap(description)` returns
`(transcript, truth)`:

```python
txt, truth = SlateGenerator(seed=1).wrap("metal gate slam heavy")
# "metal gate slam heavy, for the project dragon game, we're at whistler village,
#  on mkh four sixteen mix pre six, 48 khz twenty four bit"
# {'description': 'metal gate slam heavy', 'project': 'dragon game',
#  'location': 'whistler village', 'microphones': ['Sennheiser MKH 416'],
#  'recorder': 'Sound Devices MixPre-6', 'sample_rate': 48000, 'bit_depth': '24'}
```

It draws from fixed lists (`_PROJ_A`, `_LOC`, `_NAMES`, `_MICS`, `_RECORDERS`, `_SETUPS`,
`_RATES`, `_BITS`, `_DATES`, `_FILLERS`, `_NUM`) and deliberately reproduces STT damage: spoken
numbers, "mix pre six", "sennheiser mk eighty forty" (the dropped h), fillers, and 50% of slates
with no punctuation. The ground-truth dict is what `evaluate.parser_field_accuracy` scores the
parser against.

## Leakage

Because the banks are hand-written, a phrase can accidentally appear in the dev or realistic sets.
`scripts/check_leakage.py` enforces:

- no bank phrase of 3+ words appears verbatim in `examples/dev_descriptions.jsonl` or
  `examples/realistic_slates.jsonl`;
- no dev text shares a 4-gram or substring with a realistic slate.

Known pre-existing violations are recorded as a baseline in `configs/champion.json` (the dev set
contains "sci fi door" and "sci fi computer" verbatim from `catid_phrases.yaml`, plus 10 generic
3-word bank entries). The gate blocks only **new** violations.

## Changing the generator

- Editing a bank changes the training data, so it changes the weights: retrain, rebuild, bump
  `ttucs-model-bow`.
- Run `python scripts/check_leakage.py` after any bank edit.
- Never copy wording from `examples/realistic_slates.jsonl` into a bank (AGENTS.md rule 6).
- Keep phrases short, lowercase and library-style; the model learns exactly what you write.

Next: [about_16_datasets-and-splits.md](about_16_datasets-and-splits.md) turns these texts into splits.
