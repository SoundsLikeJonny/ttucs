"""Zero-shot / cold-start scorer: similarity between the input and UCS label documents.

Every CatID is represented by its UCS Explanation + Synonyms (see
:meth:`ttucs.taxonomy.UCSEntry.label_document`). No labelled data is needed, which makes
this a useful floor for rare or brand-new categories and a sanity baseline for the
supervised models.

Backends
--------
``tfidf`` (default)
    word + character n-gram TF-IDF, cosine similarity. Offline, instant.
``sbert``
    sentence-transformers embeddings (needs ``pip install ttucs[transformers]`` and a
    one-time model download).
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ttucs.models.base import UCSModel, register
from ttucs.taxonomy import UCSTaxonomy
from ttucs.text import preprocess


def label_documents_for(labels: Sequence[str], version: str | None = None) -> list[str]:
    tax = UCSTaxonomy.load(version) if version else UCSTaxonomy.load()
    return [preprocess(tax[c].label_document()) for c in labels]


@register("zeroshot")
class ZeroShotClassifier(UCSModel):
    """Cosine similarity between text and label documents, scaled into logits."""

    def __init__(self, labels, label_categories, config=None) -> None:
        super().__init__(labels, label_categories, config)
        self.backend = self.config.get("backend", "tfidf")
        self.scale = float(self.config.get("similarity_scale", 20.0))
        self._vec = None
        self._label_matrix: np.ndarray | None = None
        self._encoder = None

    # ------------------------------------------------------------------ index
    def _build(self) -> None:
        docs = label_documents_for(self.labels, self.config.get("ucs_version"))
        if self.backend == "sbert":
            self._label_matrix = self._embed(docs)
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self._label_matrix = self._vec.fit_transform(docs)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.config.get("encoder", "sentence-transformers/all-MiniLM-L6-v2"), device="cpu")
        return np.asarray(self._encoder.encode(list(texts), batch_size=256, normalize_embeddings=True,
                                               show_progress_bar=False))

    def fit(self, train, val, train_cfg, on_epoch=None) -> dict:
        self._build()
        return {}

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        if self._label_matrix is None:
            self._build()
        clean = [preprocess(t) for t in texts]
        if self.backend == "sbert":
            sims = self._embed(clean) @ self._label_matrix.T
        else:
            from sklearn.preprocessing import normalize

            q = normalize(self._vec.transform(clean))
            sims = (q @ normalize(self._label_matrix).T).toarray() / 2.0  # two blocks -> [0, 1]
        return (sims * self.scale).astype(np.float32)

    def _save_weights(self, path: Path) -> None:
        if self.backend == "tfidf" and self._vec is not None:
            with open(path / "tfidf.pkl", "wb") as fh:
                pickle.dump((self._vec, self._label_matrix), fh)

    def _load_weights(self, path: Path) -> None:
        f = path / "tfidf.pkl"
        if self.backend == "tfidf" and f.is_file():
            with open(f, "rb") as fh:
                self._vec, self._label_matrix = pickle.load(fh)  # noqa: S301 - own artifact
        else:
            self._build()
