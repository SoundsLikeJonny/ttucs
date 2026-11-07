"""Common model interface and a reusable PyTorch training loop."""

from __future__ import annotations

import abc
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[UCSModel]] = {}


def register(name: str) -> Callable[[type[UCSModel]], type[UCSModel]]:
    def deco(cls: type[UCSModel]) -> type[UCSModel]:
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


class UCSModel(abc.ABC):
    """A text -> UCS CatID scorer.

    Sub-classes implement :meth:`logits`, :meth:`fit`, :meth:`_save_weights` and
    :meth:`_load_weights`. Labels are CatIDs; ``label_categories`` maps each label to its
    UCS Category for the hierarchical auxiliary head.
    """

    name: str = "base"

    def __init__(self, labels: Sequence[str], label_categories: Sequence[str],
                 config: dict[str, Any] | None = None) -> None:
        self.labels = list(labels)
        self.label_categories = list(label_categories)
        self.categories = sorted(set(self.label_categories))
        self._cat_index = {c: i for i, c in enumerate(self.categories)}
        self.label_to_cat = np.array([self._cat_index[c] for c in self.label_categories])
        self.label_index = {lab: i for i, lab in enumerate(self.labels)}
        self.config = dict(config or {})

    # ------------------------------------------------------------------ interface
    @abc.abstractmethod
    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Unnormalised scores, shape ``(len(texts), len(labels))``."""

    def fit(self, train: Sequence[tuple[str, str]], val: Sequence[tuple[str, str]],
            train_cfg: dict, on_epoch: Callable[[int, dict], None] | None = None) -> dict:
        """Train on (text, label) pairs. Rule-based models need not override this."""
        return {}

    def _save_weights(self, path: Path) -> None:  # pragma: no cover - optional
        pass

    def _load_weights(self, path: Path) -> None:  # pragma: no cover - optional
        pass

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path, extra: dict | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"model_name": self.name, "labels": self.labels,
                "label_categories": self.label_categories, "model": self.config, **(extra or {})}
        (d / "config.json").write_text(json.dumps(meta, indent=2))
        self._save_weights(d)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> tuple[UCSModel, dict]:
        d = Path(directory)
        meta = json.loads((d / "config.json").read_text())
        klass = MODEL_REGISTRY[meta["model_name"]]
        model = klass(meta["labels"], meta["label_categories"], meta.get("model", {}))
        model._load_weights(d)
        return model, meta

    def encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        return np.array([self.label_index[lab] for lab in labels], dtype=np.int64)


def load_state_dict(directory: str | Path) -> dict[str, torch.Tensor]:
    """Load ``weights.safetensors`` (preferred, may be fp16) or ``weights.pt`` as fp32 tensors."""
    d = Path(directory)
    st = d / "weights.safetensors"
    if st.is_file():
        from safetensors.torch import load_file

        state = load_file(str(st))
    elif (d / "weights.pt").is_file():
        state = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no weights.safetensors or weights.pt in {d}")
    return {k: v.float() if v.is_floating_point() else v for k, v in state.items()}


# ---------------------------------------------------------------------------- training
def train_loop(module: nn.Module, batches: Callable[[bool], Iterable[tuple]],
               forward: Callable[[tuple], tuple[torch.Tensor, torch.Tensor | None]],
               evaluate: Callable[[], dict], train_cfg: dict, cat_weight: float,
               on_epoch: Callable[[int, dict], None] | None = None) -> dict:
    """Generic loop: AdamW, label smoothing, aux category loss, early stopping on val top-1.

    ``batches(train=True)`` yields tuples whose last two items are ``y`` and ``y_cat``.
    """
    torch.set_num_threads(int(train_cfg.get("num_threads", 8)))
    lr = float(train_cfg["lr"])
    # Large sparse embedding tables (hashed n-grams) get SparseAdam: only rows touched by a
    # batch are updated, which is ~50x faster than a dense AdamW step over the whole table.
    sparse = [m.weight for m in module.modules()
              if isinstance(m, (nn.Embedding, nn.EmbeddingBag)) and m.sparse]
    sparse_ids = {id(p) for p in sparse}
    dense = [p for p in module.parameters() if id(p) not in sparse_ids]
    opts: list[torch.optim.Optimizer] = [torch.optim.AdamW(
        dense, lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.0)))]
    if sparse:
        opts.append(torch.optim.SparseAdam(sparse, lr=lr))
    epochs = int(train_cfg["epochs"])
    total_steps = max(1, epochs * _count(batches))
    scheds = [torch.optim.lr_scheduler.OneCycleLR(o, max_lr=lr, total_steps=total_steps, pct_start=0.1)
              for o in opts]
    ce = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    best, best_state, bad = -1.0, None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_top1": []}
    for epoch in range(epochs):
        module.train()
        t0, total, n = time.time(), 0.0, 0
        for batch in batches(True):
            y, y_cat = batch[-2], batch[-1]
            logits, cat_logits = forward(batch)
            loss = ce(logits, y)
            if cat_logits is not None and cat_weight > 0:
                loss = loss + cat_weight * ce(cat_logits, y_cat)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o, sc in zip(opts, scheds, strict=True):
                o.step()
                sc.step()
            total += loss.item() * len(y)
            n += len(y)
        module.eval()
        metrics = evaluate()
        metrics["train_loss"] = total / max(n, 1)
        metrics["epoch_seconds"] = time.time() - t0
        history["train_loss"].append(metrics["train_loss"])
        history["val_top1"].append(metrics["val_top1"])
        log.info("epoch %d  loss %.4f  val_top1 %.4f  (%.1fs)", epoch + 1, metrics["train_loss"],
                 metrics["val_top1"], metrics["epoch_seconds"])
        if on_epoch:
            on_epoch(epoch, metrics)
        if metrics["val_top1"] > best + 1e-4:
            best, bad = metrics["val_top1"], 0
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        else:
            bad += 1
            if bad >= int(train_cfg.get("patience", 3)):
                log.info("early stopping at epoch %d", epoch + 1)
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    history["best_val_top1"] = [best]
    return history


def _count(batches: Callable[[bool], Iterable[tuple]]) -> int:
    fn = getattr(batches, "num_batches", None)
    return int(fn()) if fn else sum(1 for _ in batches(True))


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> float:
    """Temperature scaling (Guo et al., 2017) fitted on validation logits."""
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    ce = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = ce(lt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))
