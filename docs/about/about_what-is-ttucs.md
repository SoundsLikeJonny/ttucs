# What ttucs is, and why it exists

## The problem

Field recordings arrive with a spoken slate: the recordist talks into the recorder before or after
the take, naming the project, the location, what was recorded, the gear, the format and the take
number. That audio slate is the most reliable metadata a recording will ever have, and it is
almost always thrown away, because turning it into library metadata is manual work: listen, type
the CatID, type the description, type the gear, rename the file.

`ttucs` automates the last mile. It takes the **text** of the slate (from any speech-to-text
engine, or typed by hand) and returns the metadata a UCS-aware library expects:

- a CatID (`WATRSurf`) and its Category/SubCategory (`WATER` / `SURF`);
- a calibrated confidence and the top-k alternatives;
- the parsed slate fields (project, location, microphones, recorder, sample rate, bit depth, take,
  date, perspective, ...);
- a suggested UCS filename and a keyword list.

It does not listen to audio. It does not touch the recording. It is a text-to-metadata tool, and
that is deliberate: speech-to-text is a solved, cheap, local problem, and keeping ttucs text-only
means the whole tool is a few megabytes of Python plus one small weights file.

## Who it is for

Sound designers and recordists who keep a UCS-organised library (Soundminer, BaseHead, or plain
folders) and who already slate their takes. The owner of the project is a senior technical sound
designer; the tool is built to fit that workflow rather than to be a general audio-classification
research project.

## The goals, in priority order

These come from AGENTS.md section 1 and they decide most trade-offs in the code.

1. **Simple and genuinely useful.** Small, clear code, small install, few features. A new
   contributor should be able to read the whole inference path in an afternoon (this folder exists
   to make that true).
2. **`pip install ttucs` works out of the box, and inference is fully offline.** No API keys, no
   model download at first use, no network at import time or during prediction. `TTUCS_OFFLINE=1`
   must guarantee zero network access. This is why the weights ship as their own PyPI package
   ([about_distribution-and-hub.md](about_distribution-and-hub.md)).
3. **Everything published is redistributable.** MIT code, MIT weights, the public UCS list. No
   third-party sound-library metadata anywhere: training data is synthetic text plus the UCS list.
4. **Accuracy on real slate wording**, not on synthetic test sets. The headline number is the
   80 hand-written realistic slates, not the 22,384 synthetic descriptions.
5. **Next lever: fine-tuning on a user's own library metadata.** Not built yet; the design keeps
   the door open (the model is small, the training loop is generic, the config is a plain dict).

## What "done" means for a change

Because the goals are about honesty as much as accuracy, the project has an unusually explicit
definition of finished work (AGENTS.md section 2, rule 7):

- `ruff check src tests scripts` and `pytest -q` pass;
- the real user path was exercised: a clean venv, the real wheel, the CLI, with no network when
  offline behaviour matters;
- accuracy claims name the test set and its n, and are compared on dev plus synthetic validation
  only, with the realistic slates scored once at the end;
- the CHANGELOG has an entry and the versions follow SemVer.

## What ttucs is not

- **Not an audio classifier.** It never sees the waveform. A slate that says "door creak" but
  records a bird will be classified as a door creak.
- **Not a transcriber.** Speech-to-text is upstream; ttucs starts from text. It is tolerant of STT
  noise (that is what the normaliser and character n-grams are for) but it does not do ASR.
- **Not a metadata database.** It returns suggestions; filing them into a library is the user's
  step.
- **Not a general-purpose UCS oracle.** The model knows 738 CatIDs and was trained on synthetic
  text; it is good on the wording recordists actually use and it is over-confident on phrasing it
  has never seen ([about_calibration-and-decisions.md](about_calibration-and-decisions.md)).

## The shape of the solution

Three independent pieces, joined by one rule:

| piece | module | job | needs training? |
|---|---|---|---|
| slate parsing | `ttucs.slate` | split the transcript into fields and a description | no, rules |
| classification | `ttucs.models` | score the description against 738 CatIDs | yes, once |
| presentation | `ttucs.pipeline`, `ttucs.taxonomy` | calibrate, decide, build filename/keywords | no |

The rule that joins them: **only the parsed description is classified.** Slate fields are returned
but never fed to the model. This keeps the model's job small (short descriptions, not rambling
transcripts), keeps the parser free to change without retraining, and keeps the training data
honest (the generator produces descriptions, not slates, for the classifier).

## Where to go next

- The file-by-file map: [about_repo-map.md](about_repo-map.md)
- The label space: [about_ucs-taxonomy.md](about_ucs-taxonomy.md)
- The full inference path: [about_normalisation.md](about_normalisation.md) onwards
