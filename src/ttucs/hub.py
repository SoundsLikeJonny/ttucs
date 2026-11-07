"""Offline-first model resolution, download and export.

``TTUCS.from_pretrained()`` resolves a model in this order and stops at the first hit:

1. an explicit local directory (``TTUCS.from_pretrained("path/to/model")``);
2. the installed weights package (``ttucs-model-bow`` -> ``ttucs_model_bow``), which
   ``pip install ttucs`` pulls in automatically;
3. the user cache (``$TTUCS_HOME`` or the platform cache dir, e.g. ``~/.cache/ttucs``),
   filled by ``ttucs download``;
4. a download of the GitHub Release asset with SHA-256 verification - **only** when
   offline mode is off (``TTUCS_OFFLINE=1`` or ``HF_HUB_OFFLINE=1`` disables all network).

Nothing in this module touches the network at import time or during prediction.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ttucs._version import __version__

log = logging.getLogger(__name__)

DEFAULT_MODEL = "bow"
#: Highest model directory layout this ttucs version understands.
MODEL_FORMAT_VERSION = 1
GITHUB_RELEASES = "https://github.com/SoundsLikeJonny/ttucs/releases/download"
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelSpec:
    """A named, published model: its weights package and its release asset."""

    name: str
    package: str          # import name of the data-only weights distribution
    distribution: str     # PyPI name
    version: str
    sha256: str           # of the release tarball
    archive: str

    @property
    def url(self) -> str:
        return f"{GITHUB_RELEASES}/model-{self.name}-v{self.version}/{self.archive}"


MODELS: dict[str, ModelSpec] = {
    "bow": ModelSpec(
        name="bow", package="ttucs_model_bow", distribution="ttucs-model-bow", version="0.1.1",
        sha256="b554dc771527b9a864aeb877ad7ea6741834ac55de78fad5a6a726e2f357954d",
        archive="model-bow-v0.1.1.tar.gz"),
}


class ModelNotFoundError(FileNotFoundError):
    """No usable model directory could be resolved."""


class OfflineError(RuntimeError):
    """A network operation was requested while offline mode is on."""


class ChecksumError(RuntimeError):
    """Downloaded or packaged files do not match their expected SHA-256."""


class IncompatibleModelError(RuntimeError):
    """The model directory needs a different ttucs version."""


class DownloadError(RuntimeError):
    """The weights could not be fetched (no network, HTTP error, timeout)."""


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    source: str           # "path" | "package" | "cache" | "download"
    name: str | None = None


# ----------------------------------------------------------------------------- settings
def is_offline() -> bool:
    """``TTUCS_OFFLINE`` (or the Hugging Face convention ``HF_HUB_OFFLINE``) disables network."""
    return any(os.environ.get(v, "").strip().lower() in _TRUTHY for v in ("TTUCS_OFFLINE", "HF_HUB_OFFLINE"))


def cache_root() -> Path:
    env = os.environ.get("TTUCS_HOME")
    if env:
        return Path(env).expanduser()
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("ttucs", appauthor=False))


def is_model_dir(path: str | Path) -> bool:
    return (Path(path) / "config.json").is_file()


def _spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise ModelNotFoundError(f"unknown model {name!r}: not a directory and not one of "
                                 f"{sorted(MODELS)}") from None


def packaged_model_dir(name: str = DEFAULT_MODEL) -> Path | None:
    """Model directory shipped inside the installed weights package, if any."""
    spec = _spec(name)
    if importlib.util.find_spec(spec.package) is None:
        return None
    path = Path(str(importlib.resources.files(spec.package) / "model"))
    return path if is_model_dir(path) else None


def cached_model_dir(name: str = DEFAULT_MODEL) -> Path:
    spec = _spec(name)
    return cache_root() / "models" / name / spec.version


# ----------------------------------------------------------------------------- resolution
def resolve_model(model: str | Path | None = None, allow_download: bool | None = None) -> ResolvedModel:
    """Find a model directory (see module docstring for the order)."""
    if model is not None and (isinstance(model, Path) or str(model) not in MODELS):
        path = Path(model).expanduser()
        if is_model_dir(path):
            return ResolvedModel(path, "path")
        if path.exists():
            raise ModelNotFoundError(f"{path} exists but has no config.json - not a ttucs model directory")
        raise ModelNotFoundError(f"model directory not found: {path}\n"
                                 f"  Known model names: {', '.join(sorted(MODELS))}")
    name = str(model or DEFAULT_MODEL)
    spec = _spec(name)
    pkg = packaged_model_dir(name)
    if pkg is not None:
        return ResolvedModel(pkg, "package", name)
    cached = cached_model_dir(name)
    if is_model_dir(cached):
        return ResolvedModel(cached, "cache", name)
    offline = is_offline()
    if allow_download is None:
        allow_download = not offline
    if allow_download and not offline:
        log.info("model %r not installed - downloading %s (one-time)", name, spec.url)
        return ResolvedModel(download_model(name), "download", name)
    raise ModelNotFoundError(
        f"no weights found for model {name!r}.\n"
        f"  Looked in: installed package {spec.package!r}, cache {cached}\n"
        f"  Fix one of:\n"
        f"    pip install {spec.distribution}=={spec.version}   (normally installed with ttucs)\n"
        f"    ttucs download --model {name}                    (needs network; unset TTUCS_OFFLINE)\n"
        f"    TTUCS.from_pretrained('path/to/model_dir')       (a directory written by `ttucs train`)"
        + ("\n  Offline mode is ON (TTUCS_OFFLINE / HF_HUB_OFFLINE)." if offline else ""))


# ----------------------------------------------------------------------------- integrity
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(directory: str | Path) -> Path:
    d = Path(directory)
    files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
             for p in sorted(d.iterdir()) if p.is_file() and p.name != "manifest.json"}
    out = d / "manifest.json"
    out.write_text(json.dumps({"files": files}, indent=2) + "\n")
    return out


def verify_manifest(directory: str | Path) -> dict[str, str]:
    """Check every file listed in ``manifest.json``; raises :class:`ChecksumError` on mismatch."""
    d = Path(directory)
    man = d / "manifest.json"
    if not man.is_file():
        raise ChecksumError(f"{d} has no manifest.json")
    files = json.loads(man.read_text())["files"]
    for name, info in files.items():
        p = d / name
        if not p.is_file():
            raise ChecksumError(f"{p} is listed in manifest.json but missing")
        got = sha256_file(p)
        if got != info["sha256"]:
            raise ChecksumError(f"{p}: sha256 {got} != expected {info['sha256']}")
    return {n: i["sha256"] for n, i in files.items()}


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def check_compatibility(meta: dict[str, Any], where: str | Path = "model") -> None:
    """Refuse models that need a newer ttucs or use an unknown directory layout."""
    fmt = int(meta.get("format_version", 1))
    if fmt > MODEL_FORMAT_VERSION:
        raise IncompatibleModelError(
            f"{where} uses model format {fmt}; ttucs {__version__} reads up to {MODEL_FORMAT_VERSION}. "
            "Upgrade ttucs:  pip install -U ttucs")
    need = meta.get("min_ttucs_version")
    if need and _version_tuple(__version__) < _version_tuple(str(need)):
        raise IncompatibleModelError(
            f"{where} needs ttucs>={need} (installed: {__version__}). Upgrade:  pip install -U ttucs")
    trained = meta.get("ttucs_version")
    if trained and _version_tuple(str(trained))[:1] != _version_tuple(__version__)[:1]:
        log.warning("%s was trained with ttucs %s; installed ttucs is %s", where, trained, __version__)


# ----------------------------------------------------------------------------- download
def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if not (m.isfile() or m.isdir()) or (target != root and root not in target.parents):
            raise ChecksumError(f"refusing unsafe archive member {m.name!r}")
    if hasattr(tarfile, "data_filter"):  # Python >= 3.10.12 / 3.11.4
        tar.extractall(dest, filter="data")
    else:  # pragma: no cover
        tar.extractall(dest)  # noqa: S202


def download_model(name: str = DEFAULT_MODEL, dest: str | Path | None = None, url: str | None = None,
                   sha256: str | None = None, force: bool = False, timeout: float = 60.0) -> Path:
    """Download + verify a release tarball into ``dest`` (default: the user cache).

    ``url``/``sha256`` default to the published release; ``TTUCS_MODEL_URL`` and
    ``TTUCS_MODEL_SHA256`` override them (mirrors, air-gapped artefact stores).
    """
    if is_offline():
        raise OfflineError("offline mode is on (TTUCS_OFFLINE / HF_HUB_OFFLINE) - refusing to download. "
                           "Unset it, or install the weights package / pass a local model directory.")
    spec = _spec(name)
    url = url or os.environ.get("TTUCS_MODEL_URL") or spec.url
    expected = (sha256 or os.environ.get("TTUCS_MODEL_SHA256") or spec.sha256).lower()
    if not url.startswith(("https://", "http://", "file://")):
        raise ValueError(f"unsupported URL scheme: {url}")
    if len(expected) != 64:
        raise ChecksumError(f"no valid sha256 pinned for {url}; pass sha256= or set TTUCS_MODEL_SHA256")
    target = Path(dest).expanduser() if dest else cached_model_dir(name)
    if is_model_dir(target) and not force:
        log.info("model already present at %s", target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".ttucs-dl-") as tmp:
        archive = Path(tmp) / "model.tar.gz"
        h = hashlib.sha256()
        req = urllib.request.Request(url, headers={"User-Agent": f"ttucs/{__version__}"})  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(archive, "wb") as fh:  # noqa: S310
                while block := resp.read(1 << 20):
                    h.update(block)
                    fh.write(block)
        except (urllib.error.URLError, OSError) as exc:
            raise DownloadError(
                f"could not download model {name!r} from {url}: {getattr(exc, 'reason', exc)}. "
                f"Install the weights with `pip install {spec.distribution}`, pass a local model "
                "directory, or retry with network access.") from exc
        got = h.hexdigest()
        if got != expected:
            raise ChecksumError(f"sha256 mismatch for {url}: got {got}, expected {expected} "
                                "(download discarded)")
        extract = Path(tmp) / "x"
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, extract)
        found = [p.parent for p in extract.rglob("config.json")]
        if len(found) != 1:
            raise ModelNotFoundError(f"{url} does not contain exactly one model directory")
        verify_manifest(found[0])
        check_compatibility(json.loads((found[0] / "config.json").read_text()), url)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(found[0]), str(target))
    log.info("model %r saved to %s", name, target)
    return target


# ----------------------------------------------------------------------------- export
def export_model(src: str | Path, dst: str | Path, dtype: str = "float16",
                 extra_meta: dict[str, Any] | None = None) -> Path:
    """Write a compact, safe release copy of a trained model directory.

    Floating-point tensors are cast to ``dtype`` and stored as ``weights.safetensors``
    (no pickle); ``config.json`` gains format/compatibility metadata and a ``manifest.json``
    with per-file SHA-256 is written.
    """
    import torch
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    meta = json.loads((src / "config.json").read_text())
    if (src / "weights.safetensors").is_file():
        from safetensors.torch import load_file

        state = load_file(str(src / "weights.safetensors"))
    else:
        state = torch.load(src / "weights.pt", map_location="cpu", weights_only=True)
    tdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    state = {k: (v.to(tdtype) if v.is_floating_point() else v).contiguous() for k, v in state.items()}
    dst.mkdir(parents=True, exist_ok=True)
    for stale in ("weights.pt", "weights.safetensors", "manifest.json"):
        (dst / stale).unlink(missing_ok=True)
    save_file(state, str(dst / "weights.safetensors"), metadata={"format": "pt", "dtype": dtype})
    meta.update({"format_version": MODEL_FORMAT_VERSION, "weights_dtype": dtype,
                 "min_ttucs_version": meta.get("min_ttucs_version", "0.2.0")})
    meta.update(extra_meta or {})
    (dst / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_manifest(dst)
    return dst


def describe(resolved: ResolvedModel) -> dict[str, Any]:
    """Summary used by ``ttucs info``."""
    meta = json.loads((resolved.path / "config.json").read_text())
    return {"path": str(resolved.path), "source": resolved.source, "model_name": meta.get("model_name"),
            "model_version": meta.get("model_version"), "trained_with_ttucs": meta.get("ttucs_version"),
            "ucs_version": meta.get("ucs_version"), "weights_dtype": meta.get("weights_dtype", "float32"),
            "labels": len(meta.get("labels", [])), "training_data": meta.get("training_data")}
