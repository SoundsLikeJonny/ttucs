"""BOOM ONE demo dataset ingestion.

BOOM Library links a public "Full XLS Content List" (a Google Sheet, "BOOM ONE Metadata")
from https://www.boomlibrary.com/sound-effects/boom-one/. It contains ~40k UCS-labelled
rows (Filename, Description, CatID, FXName, Keywords, ...). It is the best openly visible
example of professional UCS metadata, so ttucs uses it to

* study description style and category distribution (see ``docs/DATA_CARD.md``),
* provide a *real-text* evaluation set (and optional training data) for the classifier.

The metadata is (c) BOOM Library. ttucs never redistributes it: ``ttucs fetch-boom``
downloads it to your local ``data/raw`` folder for personal research use. Check BOOM
Library's terms before using models trained on it commercially.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import urllib.request
from pathlib import Path

from ttucs.synth.generator import Example
from ttucs.taxonomy import UCSTaxonomy
from ttucs.text import split_description, strip_take_numbers

log = logging.getLogger(__name__)

BOOM_ONE_SHEET_ID = "1GkJTKbUIha3FU92grSZjmIe-KHIxziXQbatOehI2p6c"
BOOM_ONE_CSV_URL = (f"https://docs.google.com/spreadsheets/d/{BOOM_ONE_SHEET_ID}"
                    "/gviz/tq?tqx=out:csv")


def fetch_boom_one(dest: str | Path, url: str = BOOM_ONE_CSV_URL, timeout: int = 120) -> Path:
    """Download the BOOM ONE metadata sheet as CSV to ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not url.startswith("https://"):
        raise ValueError("only https URLs are supported")
    req = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, headers={"User-Agent": "ttucs/0.1 (+https://github.com/SoundsLikeJonny/ttucs)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        data = resp.read()
    if data[:200].lstrip().startswith(b"<"):
        raise RuntimeError("BOOM ONE sheet returned HTML instead of CSV (access changed?)")
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()[:16]
    log.info("downloaded %s bytes to %s (sha256 %s...)", len(data), dest, sha)
    return dest


def _group_key(fx_name: str) -> str:
    """Near-duplicate group: FXName without take numbers / trailing variation words."""
    stem = strip_take_numbers(fx_name).lower()
    return re.sub(r"\s+", " ", stem)


def load_boom_one(path: str | Path, taxonomy: UCSTaxonomy,
                  text_fields: tuple[str, ...] = ("description", "fxname"),
                  exclude_categories: set[str] | frozenset[str] = frozenset()) -> list[Example]:
    """Turn BOOM ONE rows into :class:`Example` objects.

    ``text_fields`` selects which views of a row become examples:
    ``description`` (prose part of the Description, keyword tail removed),
    ``fxname`` (FXName without take numbers) and ``keywords`` (the keyword tail).
    Rows whose CatID is not in the taxonomy (after alias resolution) are skipped.
    """
    examples: list[Example] = []
    skipped = 0
    seen: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            entry = taxonomy.resolve(row.get("CatID", "") or "")
            if entry is None:
                skipped += 1
                continue
            if entry.category in exclude_categories:
                continue
            prose, keywords = split_description(row.get("Description", ""))
            fx = strip_take_numbers(row.get("FXName", "") or "")
            group = f"boom:{entry.cat_id}:{_group_key(fx)}"
            views = {"description": prose, "fxname": fx, "keywords": ", ".join(keywords)}
            for f in text_fields:
                text = (views.get(f) or "").strip()
                if not text or (text.lower(), entry.cat_id) in seen:
                    continue
                seen.add((text.lower(), entry.cat_id))
                examples.append(Example(text=text, label=entry.cat_id, source="boom", group=group,
                                        meta={"view": f, "library": row.get("Library", "")}))
    if skipped:
        log.warning("skipped %d BOOM rows with CatIDs not in UCS v%s", skipped, taxonomy.version)
    return examples
