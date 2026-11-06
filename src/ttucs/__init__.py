"""ttucs - Text to UCS.

Classify spoken field-recording slates (speech-to-text output) into Universal Category
System (UCS) CatIDs::

    from ttucs import TTUCS

    model = TTUCS.from_pretrained("models/ttucs-bow")
    pred = model.predict("project nightfall, location tofino beach, waves crashing on rocks, take 3")
    print(pred.cat_id, pred.confidence, pred.filename)
"""

from ttucs._version import __version__
from ttucs.pipeline import TTUCS, Candidate, Prediction
from ttucs.slate import SlateFields, SlateParser, parse_slate
from ttucs.taxonomy import UCS_VERSION, UCSEntry, UCSFilename, UCSTaxonomy, parse_ucs_filename

__all__ = ["TTUCS", "UCS_VERSION", "Candidate", "Prediction", "SlateFields", "SlateParser",
           "UCSEntry", "UCSFilename", "UCSTaxonomy", "__version__", "parse_slate",
           "parse_ucs_filename"]
