"""ttucs - Text to UCS.

Classify spoken field-recording slates (speech-to-text output) into Universal Category
System (UCS) CatIDs::

    import ttucs as t

    pred = t.predict("project nightfall, location tofino beach, waves crashing on rocks, take 3")
    print(pred.cat_id, pred.confidence, pred.filename)

The first :func:`predict` call loads the default pretrained model (the installed
``ttucs-model-bow`` weights, offline) and reuses it afterwards; nothing is loaded at import
time. For a specific model directory or inference overrides, use :meth:`TTUCS.from_pretrained`
directly.
"""

from collections.abc import Sequence

from ttucs._version import __version__
from ttucs.pipeline import TTUCS, Candidate, Prediction
from ttucs.slate import SlateFields, SlateParser, parse_slate
from ttucs.taxonomy import UCS_VERSION, UCSEntry, UCSFilename, UCSTaxonomy, parse_ucs_filename

__all__ = ["TTUCS", "UCS_VERSION", "Candidate", "Prediction", "SlateFields", "SlateParser",
           "UCSEntry", "UCSFilename", "UCSTaxonomy", "__version__", "parse_slate",
           "parse_ucs_filename", "predict", "predict_batch"]

_DEFAULT: TTUCS | None = None


def _default_model() -> TTUCS:
    """The shared default pipeline - loaded lazily on first use, never at import time."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TTUCS.from_pretrained()
    return _DEFAULT


def predict(text: str, top_k: int = 5, creator_id: str | None = None,
            source_id: str | None = None) -> Prediction:
    """Classify one slate transcript (or plain description) with the default model."""
    return _default_model().predict(text, top_k=top_k, creator_id=creator_id,
                                    source_id=source_id)


def predict_batch(texts: Sequence[str], top_k: int = 5, creator_id: str | None = None,
                  source_id: str | None = None) -> list[Prediction]:
    """Classify several slate transcripts with the default model."""
    return _default_model().predict_batch(texts, top_k=top_k, creator_id=creator_id,
                                          source_id=source_id)
