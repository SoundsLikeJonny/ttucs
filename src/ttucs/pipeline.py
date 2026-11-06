"""High-level inference API: slate text in, UCS metadata out."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ttucs.models import MODEL_REGISTRY, UCSModel
from ttucs.slate.parser import SlateFields, SlateParser
from ttucs.taxonomy import UCSFilename, UCSTaxonomy
from ttucs.text import abbreviate, initials, make_fx_name, make_keywords

DEFAULT_UNKNOWN_THRESHOLD = 0.12
DEFAULT_CATEGORY_THRESHOLD = 0.5


@dataclass
class Candidate:
    """One ranked CatID hypothesis."""

    cat_id: str
    category: str
    subcategory: str
    cat_short: str
    confidence: float

    @property
    def category_full(self) -> str:
        return f"{self.category}-{self.subcategory}"


@dataclass
class Prediction:
    """Result of :meth:`TTUCS.predict` - JSON serialisable via :meth:`to_dict`/:meth:`to_json`.

    ``decision`` is one of

    * ``"confident"`` - top CatID confidence >= ``unknown_threshold``;
    * ``"category_fallback"`` - the subcategory is uncertain but the Category is not, so
      ``cat_id`` is that Category's MISC CatID (UCS's own convention for "this category,
      unspecified"); the model's best guess stays in ``alternatives[0]``;
    * ``"unknown"`` - neither is confident; ``cat_id`` holds the best guess, ``filename`` is
      still suggested but should be reviewed.
    """

    text: str
    description: str
    cat_id: str
    category: str
    subcategory: str
    cat_short: str
    confidence: float
    category_confidence: float
    decision: str
    alternatives: list[Candidate]
    categories: list[dict[str, float | str]]
    slate: dict[str, Any]
    fx_name: str
    keywords: list[str]
    filename: str
    filename_parts: dict[str, Any] = field(default_factory=dict)

    @property
    def is_unknown(self) -> bool:
        return self.decision == "unknown"

    @property
    def category_full(self) -> str:
        return f"{self.category}-{self.subcategory}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category_full"] = self.category_full
        d["is_unknown"] = self.is_unknown
        return d

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


class TTUCS:
    """Slate parser + UCS classifier + filename/keyword suggestion.

    Parameters
    ----------
    model:
        Any :class:`~ttucs.models.base.UCSModel`.
    temperature:
        Softmax temperature fitted on validation data (calibration).
    unknown_threshold:
        Minimum top-1 probability for a ``"confident"`` decision.
    category_threshold:
        Minimum aggregated Category probability for a ``"category_fallback"`` decision.
    creator_id, source_id:
        Defaults for the UCS filename when the slate does not name a recordist/project.
    """

    def __init__(self, model: UCSModel, taxonomy: UCSTaxonomy | None = None,
                 parser: SlateParser | None = None, temperature: float = 1.0,
                 unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
                 category_threshold: float = DEFAULT_CATEGORY_THRESHOLD,
                 creator_id: str = "", source_id: str = "", meta: dict | None = None) -> None:
        self.model = model
        self.taxonomy = taxonomy or UCSTaxonomy.load()
        self.parser = parser or SlateParser()
        self.temperature = float(temperature)
        self.unknown_threshold = float(unknown_threshold)
        self.category_threshold = float(category_threshold)
        self.creator_id = creator_id
        self.source_id = source_id
        self.meta = meta or {}
        self._entries = [self.taxonomy[c] for c in model.labels]
        self._misc = {e.category: e.cat_id for e in self.taxonomy.entries if e.subcategory == "MISC"}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path: str | Path, **overrides: Any) -> TTUCS:
        """Load a model directory written by ``ttucs train`` (or :meth:`save_pretrained`)."""
        model, meta = UCSModel.load(path)
        inf = meta.get("inference", {})
        kwargs = {"temperature": meta.get("temperature", 1.0),
                  "unknown_threshold": inf.get("unknown_threshold", DEFAULT_UNKNOWN_THRESHOLD),
                  "category_threshold": inf.get("category_threshold", DEFAULT_CATEGORY_THRESHOLD)}
        kwargs.update(overrides)
        tax = UCSTaxonomy.load(meta.get("ucs_version", "8.2.1"))
        return cls(model, taxonomy=tax, meta=meta, **kwargs)

    @classmethod
    def from_mlflow(cls, model_uri: str, **overrides: Any) -> TTUCS:
        """Load from an MLflow model URI, e.g. ``models:/ttucs/1`` (needs ``ttucs[mlflow]``)."""
        import mlflow

        local = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        return cls.from_pretrained(Path(local) / "artifacts" / "model_dir", **overrides)

    def save_pretrained(self, path: str | Path) -> Path:
        extra = {k: v for k, v in self.meta.items()
                 if k not in {"model_name", "labels", "label_categories", "model"}}
        extra.update({"temperature": self.temperature, "ucs_version": self.taxonomy.version,
                      "inference": {**extra.get("inference", {}),
                                    "unknown_threshold": self.unknown_threshold,
                                    "category_threshold": self.category_threshold}})
        return self.model.save(path, extra=extra)

    @property
    def labels(self) -> list[str]:
        return self.model.labels

    # ------------------------------------------------------------------ inference
    def probabilities(self, descriptions: Sequence[str]) -> np.ndarray:
        """Calibrated CatID probabilities for already-extracted descriptions."""
        logits = self.model.logits(list(descriptions)) / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def parse(self, text: str) -> SlateFields:
        return self.parser.parse(text)

    def predict(self, text: str, top_k: int = 5, creator_id: str | None = None,
                source_id: str | None = None) -> Prediction:
        """Parse a slate transcript (or plain description) and classify it."""
        return self.predict_batch([text], top_k=top_k, creator_id=creator_id, source_id=source_id)[0]

    def predict_batch(self, texts: Sequence[str], top_k: int = 5, creator_id: str | None = None,
                      source_id: str | None = None) -> list[Prediction]:
        slates = [self.parser.parse(t) for t in texts]
        probs = self.probabilities([s.description or s.normalized_text or " " for s in slates])
        return [self._build(t, s, p, top_k, creator_id, source_id)
                for t, s, p in zip(texts, slates, probs, strict=True)]

    # ------------------------------------------------------------------ helpers
    def _build(self, text: str, slate: SlateFields, p: np.ndarray, top_k: int,
               creator_id: str | None, source_id: str | None) -> Prediction:
        order = np.argsort(-p)[: max(1, top_k)]
        alts = [Candidate(e.cat_id, e.category, e.subcategory, e.cat_short, round(float(p[i]), 6))
                for i, e in ((i, self._entries[i]) for i in order)]
        cat_p = np.bincount(self.model.label_to_cat, weights=p, minlength=len(self.model.categories))
        cat_order = np.argsort(-cat_p)
        categories = [{"category": self.model.categories[i], "confidence": round(float(cat_p[i]), 6)}
                      for i in cat_order[: max(1, top_k)]]
        best = alts[0]
        best_cat_p = float(cat_p[self.model.categories.index(best.category)])
        chosen = self.taxonomy[best.cat_id]
        if best.confidence >= self.unknown_threshold:
            decision = "confident"
        else:
            top_cat = self.model.categories[cat_order[0]]
            misc = self._misc.get(top_cat)
            if float(cat_p[cat_order[0]]) >= self.category_threshold and misc:
                decision, chosen = "category_fallback", self.taxonomy[misc]
                best_cat_p = float(cat_p[cat_order[0]])
            else:
                decision = "unknown"
        conf = float(p[self.model.label_index[chosen.cat_id]]) if chosen.cat_id in self.model.label_index \
            else 0.0
        fx = make_fx_name(slate.description)
        fn = self._filename(chosen.cat_id, fx, slate, creator_id, source_id)
        return Prediction(
            text=text, description=slate.description, cat_id=chosen.cat_id, category=chosen.category,
            subcategory=chosen.subcategory, cat_short=chosen.cat_short, confidence=round(conf, 6),
            category_confidence=round(best_cat_p, 6), decision=decision, alternatives=alts,
            categories=categories, slate=slate.to_dict(), fx_name=fx,
            keywords=make_keywords(slate.description, [chosen.subcategory.title(), chosen.category.title()]),
            filename=fn.build(), filename_parts=fn.to_dict())

    def _filename(self, cat_id: str, fx_name: str, slate: SlateFields, creator_id: str | None,
                  source_id: str | None) -> UCSFilename:
        persp = [x for x in slate.perspective if x in {"INT", "EXT"}]
        return UCSFilename(
            cat_id=cat_id, fx_name=fx_name,
            creator_id=creator_id or initials(slate.creator) or self.creator_id or "NONE",
            source_id=source_id or abbreviate(slate.project) or self.source_id or "NONE",
            user_category=persp[0] if persp else None, user_data=slate.user_data)


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
