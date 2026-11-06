"""Optional transformer path: frozen sentence-transformers encoder + MLP head.

The pretrained encoder (default ``all-MiniLM-L6-v2``, 22M parameters) embeds each text
once; only a small head is trained, so the model trains on a CPU in minutes. The output
layer is initialised from embeddings of the UCS label documents, giving every CatID a
sensible zero-shot prior before any training (cold start for rare classes).

Requires ``pip install "ttucs[transformers]"`` and a one-time model download from the
Hugging Face Hub.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ttucs.models.base import UCSModel, register, train_loop
from ttucs.models.zeroshot import label_documents_for
from ttucs.text import preprocess


class EncoderHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_labels: int, n_cats: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, in_dim))
        self.head = nn.Linear(in_dim, n_labels)
        self.cat_head = nn.Linear(in_dim, n_cats)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x + self.proj(x)
        return self.head(h), self.cat_head(h)


@register("encoder")
class EncoderClassifier(UCSModel):
    def __init__(self, labels, label_categories, config=None) -> None:
        super().__init__(labels, label_categories, config)
        self.encoder_name = self.config.get("encoder", "sentence-transformers/all-MiniLM-L6-v2")
        self._encoder = None
        self.dim = int(self.config.get("encoder_dim", 384))
        self.net = EncoderHead(self.dim, int(self.config.get("hidden_dim", 512)), len(self.labels),
                               len(self.categories), float(self.config.get("dropout", 0.2)))
        self.net.eval()

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name, device="cpu")
        return self._encoder

    def embed(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        return np.asarray(self.encoder.encode([preprocess(t) for t in texts], batch_size=batch_size,
                                              normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)

    @torch.no_grad()
    def logits(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        self.net.eval()
        return self.net(torch.from_numpy(self.embed(texts)))[0].numpy()

    def fit(self, train, val, train_cfg, on_epoch: Callable[[int, dict], None] | None = None) -> dict:
        torch.manual_seed(int(train_cfg.get("seed", 0)))
        docs = self.embed(label_documents_for(self.labels, self.config.get("ucs_version")))
        with torch.no_grad():  # zero-shot prior: logit_i = scale * cos(x, doc_i)
            self.net.head.weight.copy_(torch.from_numpy(docs) * float(self.config.get("init_scale", 10.0)))
            self.net.head.bias.zero_()
        x = torch.from_numpy(self.embed([t for t, _ in train]))
        y = self.encode_labels(lab for _, lab in train)
        yc = self.label_to_cat[y]
        vx = torch.from_numpy(self.embed([t for t, _ in val])) if val else None
        vy = self.encode_labels(lab for _, lab in val)
        bs = int(train_cfg["batch_size"])
        rng = random.Random(int(train_cfg.get("seed", 0)))

        def batches(train_mode: bool = True):
            idx = list(range(len(y)))
            rng.shuffle(idx)
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                yield x[b], torch.from_numpy(y[b]), torch.from_numpy(yc[b])

        batches.num_batches = lambda: (len(y) + bs - 1) // bs  # type: ignore[attr-defined]

        def evaluate() -> dict:
            if vx is None:
                return {"val_top1": 0.0}
            with torch.no_grad():
                pred = self.net(vx)[0].argmax(1).numpy()
            return {"val_top1": float((pred == vy).mean())}

        return train_loop(self.net, batches, lambda b: self.net(b[0]), evaluate, train_cfg,
                          float(self.config.get("category_loss_weight", 0.3)), on_epoch)

    def _save_weights(self, path: Path) -> None:
        torch.save(self.net.state_dict(), path / "head.pt")

    def _load_weights(self, path: Path) -> None:
        self.net.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
        self.net.eval()
