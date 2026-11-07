"""UCS (Universal Category System) taxonomy: loading, lookup and filename handling.

The UCS list is the label space of every ttucs model. The official list is shipped as
package data (``ttucs/resources/ucs_v<version>.csv``) so inference has no network or
spreadsheet dependency.

Filename structure (from the UCS Filenaming Convention document)::

    CatID(-UserCategory)_(VendorCategory-)FXName_CreatorID_SourceID(_UserData)
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources

UCS_VERSION = "8.2.1"

# Known CatID spellings found in the wild that are not in the official list.
# Some library metadata uses "MACHOffi" for MACHINES-OFFICE (official CatID: MACHOffc).
CATID_ALIASES: dict[str, str] = {"MACHOffi": "MACHOffc", "WOODHandl": "WOODHndl"}

# Categories that describe library *workflow* state rather than sound content
# (stems, ADR, WIP...). They stay in the taxonomy but are excluded from training by default.
WORKFLOW_CATEGORIES = frozenset({"ARCHIVED"})


@dataclass(frozen=True)
class UCSEntry:
    """A single Category/SubCategory pair of the UCS list."""

    cat_id: str
    category: str
    subcategory: str
    cat_short: str
    explanation: str
    synonyms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def category_full(self) -> str:
        """``CATEGORY-SUBCATEGORY`` as used by UCS-aware databases (e.g. ``DOORS-WOOD``)."""
        return f"{self.category}-{self.subcategory}"

    def label_document(self) -> str:
        """Natural language description of the class, used for zero-shot / cold start."""
        syn = ", ".join(self.synonyms[:40])
        return f"{self.category.lower()} {self.subcategory.lower()}. {self.explanation} {syn}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["synonyms"] = list(self.synonyms)
        d["category_full"] = self.category_full
        return d


class UCSTaxonomy:
    """In-memory view of a UCS list with fast CatID lookup."""

    def __init__(self, entries: list[UCSEntry], version: str = UCS_VERSION,
                 top_level: dict[str, str] | None = None) -> None:
        if not entries:
            raise ValueError("UCS taxonomy is empty")
        self.version = version
        self.entries = entries
        self.top_level = top_level or {}
        self._by_id = {e.cat_id: e for e in entries}
        self._by_id_lower = {e.cat_id.lower(): e for e in entries}
        self._by_full = {e.category_full.upper(): e for e in entries}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, version: str = UCS_VERSION) -> UCSTaxonomy:
        """Load the UCS list shipped with the package."""
        return _load_cached(version)

    @classmethod
    def from_csv(cls, path: str, version: str = UCS_VERSION) -> UCSTaxonomy:
        with open(path, newline="", encoding="utf-8") as fh:
            return cls(_rows_to_entries(csv.DictReader(fh)), version=version)

    # ------------------------------------------------------------------ lookup
    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, cat_id: object) -> bool:
        return isinstance(cat_id, str) and self.resolve(cat_id) is not None

    def __getitem__(self, cat_id: str) -> UCSEntry:
        entry = self.resolve(cat_id)
        if entry is None:
            raise KeyError(f"Unknown UCS CatID: {cat_id!r}")
        return entry

    def resolve(self, cat_id: str) -> UCSEntry | None:
        """Resolve a CatID (exact, alias, case-insensitive, or ``CATEGORY-SUBCATEGORY``)."""
        cat_id = cat_id.strip()
        cat_id = CATID_ALIASES.get(cat_id, cat_id)
        return (self._by_id.get(cat_id) or self._by_id_lower.get(cat_id.lower())
                or self._by_full.get(cat_id.upper()))

    @property
    def cat_ids(self) -> list[str]:
        return [e.cat_id for e in self.entries]

    @property
    def categories(self) -> list[str]:
        return sorted({e.category for e in self.entries})

    def subcategories(self, category: str) -> list[UCSEntry]:
        return [e for e in self.entries if e.category == category.upper()]

    def trainable(self, exclude_categories: frozenset[str] | set[str] = WORKFLOW_CATEGORIES
                  ) -> list[UCSEntry]:
        return [e for e in self.entries if e.category not in exclude_categories]


def _rows_to_entries(rows) -> list[UCSEntry]:
    entries = []
    for row in rows:
        syn = tuple(s.strip() for s in (row.get("Synonyms") or "").split(",") if s.strip())
        entries.append(UCSEntry(
            cat_id=row["CatID"].strip(), category=row["Category"].strip(),
            subcategory=row["SubCategory"].strip(), cat_short=row["CatShort"].strip(),
            explanation=(row.get("Explanations") or "").strip(), synonyms=syn))
    return entries


@lru_cache(maxsize=4)
def _load_cached(version: str) -> UCSTaxonomy:
    pkg = resources.files("ttucs.resources")
    with pkg.joinpath(f"ucs_v{version}.csv").open(encoding="utf-8") as fh:
        entries = _rows_to_entries(csv.DictReader(fh))
    top: dict[str, str] = {}
    top_file = pkg.joinpath(f"ucs_v{version}_top_level.csv")
    if top_file.is_file():
        with top_file.open(encoding="utf-8") as fh:
            top = {r["Category"]: r["Explanation"] for r in csv.DictReader(fh)}
    return UCSTaxonomy(entries, version=version, top_level=top)


# ---------------------------------------------------------------------- filenames
_FORBIDDEN = re.compile(r"[\\/:*?\"<>|_\x00-\x1f]+")


def sanitize_block(text: str, max_len: int | None = None) -> str:
    """Make a string safe for a UCS filename block (no ``_`` or path characters)."""
    text = _FORBIDDEN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_len and len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] or text[:max_len]
    return text


@dataclass
class UCSFilename:
    """Parsed (or to-be-built) UCS filename."""

    cat_id: str
    fx_name: str
    creator_id: str = ""
    source_id: str = ""
    user_category: str | None = None
    vendor_category: str | None = None
    user_data: str | None = None
    extension: str | None = None

    def build(self) -> str:
        head = self.cat_id + (f"-{sanitize_block(self.user_category)}" if self.user_category else "")
        fx = sanitize_block(self.fx_name)
        if self.vendor_category:
            fx = f"{sanitize_block(self.vendor_category)}-{fx}"
        blocks = [head, fx, sanitize_block(self.creator_id) or "NONE",
                  sanitize_block(self.source_id) or "NONE"]
        if self.user_data:
            blocks.append(sanitize_block(self.user_data))
        name = "_".join(blocks)
        return f"{name}.{self.extension}" if self.extension else name

    def to_dict(self) -> dict:
        return asdict(self)


def parse_ucs_filename(filename: str, taxonomy: UCSTaxonomy | None = None) -> UCSFilename:
    """Parse a UCS filename into its blocks.

    Raises ``ValueError`` if fewer than the 4 required blocks are present or, when a
    taxonomy is given, if the CatID is not part of the UCS list.
    """
    ext = None
    m = re.match(r"^(.*)\.([A-Za-z0-9]{2,5})$", filename)
    if m:
        filename, ext = m.group(1), m.group(2)
    parts = filename.split("_")
    if len(parts) < 4:
        raise ValueError(f"UCS filenames need at least 4 '_' separated blocks: {filename!r}")
    head, fx, creator, source = parts[:4]
    user_data = "_".join(parts[4:]) or None
    cat_id, _, user_cat = head.partition("-")
    vendor_cat = None
    if "-" in fx:
        vendor_cat, fx = fx.split("-", 1)
    if taxonomy is not None:
        entry = taxonomy.resolve(cat_id)
        if entry is None:
            raise ValueError(f"Unknown UCS CatID {cat_id!r} in {filename!r}")
        cat_id = entry.cat_id
    return UCSFilename(cat_id=cat_id, fx_name=fx, creator_id=creator, source_id=source,
                       user_category=user_cat or None, vendor_category=vendor_cat,
                       user_data=user_data, extension=ext)
