"""Fast CPU baseline: hashed word + character n-gram bag of embeddings (fastText-style).

Robust to STT misspellings thanks to character n-grams, trains in minutes on a laptop CPU
and has no download dependency - the default ttucs model.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ttucs.models.base import UCSModel, load_state_dict, register, train_loop
from ttucs.text import preprocess


class HashedNgramFeaturizer:
    def __init__(self, num_buckets: int = 262144, word_ngrams: int = 2,
                 char_ngrams: Sequence[int] = (3, 5), head_repeat: int = 1) -> None:
        self.num_buckets = int(num_buckets)
        self.word_ngrams = int(word_ngrams)
        self.cmin, self.cmax = int(char_ngrams[0]), int(char_ngrams[1])
        # How many times the first clause's features count in the bag mean vs context
        # clauses (EmbeddingBag mean has no per-feature weights, so emphasis is expressed
        # by repeating features). 1 = every clause equal (the published behaviour).
        self.head_repeat = max(1, int(head_repeat))

    def _h(self, s: str) -> int:
        return zlib.crc32(s.encode("utf-8")) % self.num_buckets

    def features_and_weights(self, text: str) -> tuple[list[int], list[float]]:
        """Bucket ids and matching per-feature weights (all 1.0; weights kept for symmetry).

        With ``head_repeat == 1`` (the published behaviour) the whole text is featurised
        exactly as before - n-grams may span comma boundaries. With ``head_repeat > 1``,
        clauses (comma-separated) are featurised independently - no n-gram spans a clause
        boundary - and the first clause (the label driver) enters the bag ``head_repeat``
        times while later clauses enter once.
        """
        feats: list[int] = []
        clauses = [c for c in text.split(",") if c.strip()] or [text]

        def featurise_clause(words: list[str]) -> None:
            for w in words:
                feats.append(self._h("w:" + w))
            for n in range(2, self.word_ngrams + 1):
                for i in range(len(words) - n + 1):
                    feats.append(self._h("n:" + " ".join(words[i:i + n])))
            for w in words:
                ww = f"<{w}>"
                for n in range(self.cmin, self.cmax + 1):
                    for i in range(max(0, len(ww) - n + 1)):
                        feats.append(self._h("c:" + ww[i:i + n]))

        if self.head_repeat == 1:
            feats = self._legacy_features(text)
        else:
            for k, clause in enumerate(clauses):
                words = preprocess(clause).split()
                for _ in range(self.head_repeat if k == 0 else 1):
                    featurise_clause(words)
        if not feats:
            return [0], [1.0]
        return feats, [1.0] * len(feats)

    def _legacy_features(self, text: str) -> list[int]:
        words = preprocess(text).split()
        feats = [self._h("w:" + w) for w in words]
        for n in range(2, self.word_ngrams + 1):
            feats += [self._h("n:" + " ".join(words[i:i + n])) for i in range(len(words) - n + 1)]
        for w in words:
            ww = f"<{w}>"
            for n in range(self.cmin, self.cmax + 1):
                feats += [self._h("c:" + ww[i:i + n]) for i in range(max(0, len(ww) - n + 1))]
        return feats or [0]

    def features(self, text: str) -> list[int]:
        return self.features_and_weights(text)[0]

    def batch(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.collate([self.features(t) for t in texts])

    @staticmethod
    def collate(feats: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = np.cumsum([0] + [len(f) for f in feats[:-1]])
        flat = np.fromiter((i for f in feats for i in f), dtype=np.int64)
        return torch.from_numpy(flat), torch.from_numpy(offsets.astype(np.int64))


class BowNet(nn.Module):
    def __init__(self, num_buckets: int, dim: int, n_labels: int, n_cats: int,
                 dropout: float) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(num_buckets, dim, mode="mean", sparse=True)
        self.drop = nn.Dropout(dropout)
        self.hidden = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = nn.Linear(dim, n_labels)
        self.cat_head = nn.Linear(dim, n_cats)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.drop(self.emb(ids, offsets))
        h = h + self.hidden(h)
        return self.head(h), self.cat_head(h)


@register("bow")
class BowClassifier(UCSModel):
    """Hashed n-gram EmbeddingBag classifier with an auxiliary UCS Category head."""

    def __init__(self, labels, label_categories, config=None) -> None:
        super().__init__(labels, label_categories, config)
        c = self.config
        self.featurizer = HashedNgramFeaturizer(c.get("num_buckets", 262144),
                                                c.get("word_ngrams", 2), c.get("char_ngrams", (3, 5)),
                                                c.get("head_repeat", 1))
        self.net = BowNet(self.featurizer.num_buckets, int(c.get("embedding_dim", 128)),
                          len(self.labels), len(self.categories), float(c.get("dropout", 0.2)))
        self.net.eval()

    @torch.no_grad()
    def logits(self, texts: Sequence[str], batch_size: int = 1024) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            ids, offs = self.featurizer.batch(texts[i:i + batch_size])
            out.append(self.net(ids, offs)[0].numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def fit(self, train, val, train_cfg, on_epoch: Callable[[int, dict], None] | None = None) -> dict:
        texts = [t for t, _ in train]
        y = self.encode_labels(lab for _, lab in train)
        yc = self.label_to_cat[y]
        feats = [self.featurizer.features(t) for t in texts]
        val_texts = [t for t, _ in val]
        val_y = self.encode_labels(lab for _, lab in val)
        bs = int(train_cfg["batch_size"])
        rng = random.Random(int(train_cfg.get("seed", 0)))

        def batches(train_mode: bool = True):
            idx = list(range(len(feats)))
            rng.shuffle(idx)
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                ids, offs = HashedNgramFeaturizer.collate([feats[j] for j in b])
                yield ids, offs, torch.from_numpy(y[b]), torch.from_numpy(yc[b])

        batches.num_batches = lambda: (len(feats) + bs - 1) // bs  # type: ignore[attr-defined]

        def forward(batch):
            return self.net(batch[0], batch[1])

        def evaluate() -> dict:
            pred = self.logits(val_texts).argmax(1) if val_texts else np.array([])
            return {"val_top1": float((pred == val_y).mean()) if len(val_y) else 0.0}

        return train_loop(self.net, batches, forward, evaluate, train_cfg,
                          float(self.config.get("category_loss_weight", 0.3)), on_epoch)

    def _save_weights(self, path: Path) -> None:
        torch.save(self.net.state_dict(), path / "weights.pt")

    def _load_weights(self, path: Path) -> None:
        self.net.load_state_dict(load_state_dict(path))
        self.net.eval()
