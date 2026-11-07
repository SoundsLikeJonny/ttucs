"""Offline-first model resolution, fp16 safetensors export, verified downloads."""

from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from ttucs import TTUCS, hub
from ttucs._extras import MissingExtraError, require_train_extras
from ttucs.cli import main
from ttucs.models import MODEL_REGISTRY
from ttucs.models.base import UCSModel

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["DOORWood", "GLASBrk", "WATRSurf", "ANMLDog"]


@pytest.fixture()
def model_dir(taxonomy, tmp_path):
    import torch

    torch.manual_seed(0)
    model = MODEL_REGISTRY["bow"](LABELS, [taxonomy[c].category for c in LABELS],
                                  {"embedding_dim": 16, "num_buckets": 4096})
    return model.save(tmp_path / "trained", extra={"temperature": 1.0, "ucs_version": "8.2.1",
                                                   "ttucs_version": "0.1.1"})


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    """No installed weights package, empty cache, network allowed unless a test sets offline."""
    monkeypatch.setattr(hub, "packaged_model_dir", lambda name="bow": None)
    monkeypatch.setenv("TTUCS_HOME", str(tmp_path / "home"))
    for var in ("TTUCS_OFFLINE", "HF_HUB_OFFLINE", "TTUCS_MODEL_URL", "TTUCS_MODEL_SHA256"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _tarball(model_dir: Path, tmp_path: Path) -> tuple[Path, str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("bmp", ROOT / "scripts" / "build_model_package.py")
    bmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bmp)
    release = hub.export_model(model_dir, tmp_path / "release", extra_meta={"model_version": "0.1.1"})
    tb = bmp.deterministic_tarball(release, tmp_path / "srv" / "model-bow-v0.1.1.tar.gz", "model-bow")
    again = bmp.deterministic_tarball(release, tmp_path / "again.tar.gz", "model-bow")
    assert hub.sha256_file(tb) == hub.sha256_file(again)  # reproducible archive
    return tb, hub.sha256_file(tb)


@pytest.fixture()
def server(tmp_path):
    (tmp_path / "srv").mkdir(exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "srv"))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_offline_env(monkeypatch):
    monkeypatch.delenv("TTUCS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert not hub.is_offline()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hub.is_offline()
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setenv("TTUCS_OFFLINE", "true")
    assert hub.is_offline()


def test_export_fp16_safetensors_matches_fp32(model_dir, tmp_path):
    out = hub.export_model(model_dir, tmp_path / "fp16")
    assert (out / "weights.safetensors").is_file() and not (out / "weights.pt").exists()
    meta = json.loads((out / "config.json").read_text())
    assert meta["weights_dtype"] == "float16" and meta["format_version"] == hub.MODEL_FORMAT_VERSION
    hub.verify_manifest(out)
    texts = ["wooden door slam", "glass bottle smash", "dog barking", "waves on rocks"]
    a, _ = UCSModel.load(model_dir)
    b, _ = UCSModel.load(out)
    np.testing.assert_allclose(a.logits(texts), b.logits(texts), atol=2e-3)


def test_manifest_detects_tampering(model_dir, tmp_path):
    out = hub.export_model(model_dir, tmp_path / "fp16")
    with open(out / "weights.safetensors", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(hub.ChecksumError):
        hub.verify_manifest(out)


def test_explicit_path_and_errors(model_dir, tmp_path, isolated):
    pipe = TTUCS.from_pretrained(model_dir)
    assert pipe.model_source == "path" and pipe.model_path == model_dir
    with pytest.raises(hub.ModelNotFoundError, match="not found"):
        TTUCS.from_pretrained(tmp_path / "nope")
    with pytest.raises(hub.ModelNotFoundError, match="unknown model"):
        hub._spec("bow2")


def test_offline_without_weights_gives_actionable_error(isolated, monkeypatch):
    monkeypatch.setenv("TTUCS_OFFLINE", "1")
    with pytest.raises(hub.ModelNotFoundError) as exc:
        TTUCS.from_pretrained()
    msg = str(exc.value)
    assert "pip install ttucs-model-bow" in msg and "ttucs download" in msg and "Offline mode is ON" in msg
    with pytest.raises(hub.OfflineError):
        hub.download_model("bow")


def test_resolution_order_package_then_cache(model_dir, isolated, monkeypatch):
    cache = hub.cached_model_dir("bow")
    hub.export_model(model_dir, cache)
    monkeypatch.setenv("TTUCS_OFFLINE", "1")
    assert hub.resolve_model().source == "cache"
    monkeypatch.setattr(hub, "packaged_model_dir", lambda name="bow": model_dir)
    assert hub.resolve_model("bow").source == "package"
    assert TTUCS.from_pretrained().model_source == "package"


def test_download_verifies_sha256(model_dir, isolated, server, tmp_path):
    tb, digest = _tarball(model_dir, tmp_path)
    url = f"{server}/{tb.name}"
    with pytest.raises(hub.ChecksumError, match="sha256 mismatch"):
        hub.download_model("bow", url=url, sha256="0" * 64)
    assert not hub.is_model_dir(hub.cached_model_dir("bow"))  # nothing left behind
    path = hub.download_model("bow", url=url, sha256=digest)
    assert path == hub.cached_model_dir("bow")
    hub.verify_manifest(path)
    pipe = TTUCS.from_pretrained("bow")
    assert pipe.model_source == "cache" and pipe.predict("glass smash").cat_id in set(LABELS) | {
        "GLASMisc", "DOORMisc", "WATRMisc", "ANMLMisc"}


def test_auto_download_when_online_and_cli_download(model_dir, isolated, server, tmp_path, monkeypatch,
                                                    capsys):
    tb, digest = _tarball(model_dir, tmp_path)
    monkeypatch.setenv("TTUCS_MODEL_URL", f"{server}/{tb.name}")
    monkeypatch.setenv("TTUCS_MODEL_SHA256", digest)
    assert TTUCS.from_pretrained().model_source == "download"
    dest = tmp_path / "elsewhere"
    assert main(["download", "--dest", str(dest), "--sha256", "f" * 64]) == 2
    assert "sha256 mismatch" in capsys.readouterr().err
    assert main(["download", "--dest", str(dest)]) == 0 and hub.is_model_dir(dest)


def test_unreachable_download_gives_clean_error(isolated, server, capsys):
    url = f"{server}/missing.tar.gz"  # HTTP 404 from the local server
    with pytest.raises(hub.DownloadError, match="pip install ttucs-model-bow"):
        hub.download_model("bow", url=url, sha256="0" * 64)
    assert main(["download", "--url", url, "--sha256", "0" * 64]) == 2
    assert "could not download" in capsys.readouterr().err


def test_compatibility_checks():
    with pytest.raises(hub.IncompatibleModelError, match="needs ttucs>="):
        hub.check_compatibility({"min_ttucs_version": "99.0.0"})
    with pytest.raises(hub.IncompatibleModelError, match="model format"):
        hub.check_compatibility({"format_version": hub.MODEL_FORMAT_VERSION + 1})
    hub.check_compatibility({"min_ttucs_version": "0.1.0", "ttucs_version": "0.1.1"})


def test_cli_info(model_dir, isolated, monkeypatch, capsys):
    monkeypatch.setenv("TTUCS_OFFLINE", "1")
    assert main(["info", "--json"]) == 1
    info = json.loads(capsys.readouterr().out)
    assert info["offline_mode"] is True and info["model"] is None and "no weights" in info["error"]
    rel = hub.export_model(model_dir, hub.cached_model_dir("bow"))
    assert main(["info", "--json", "--verify"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["model"]["source"] == "cache" and info["model"]["path"] == str(rel)
    assert info["model"]["manifest_verified"] is True


def test_missing_train_extras_message(monkeypatch, capsys):
    from ttucs import _extras

    monkeypatch.setattr(_extras, "missing", lambda mods: ["scikit-learn"])
    with pytest.raises(MissingExtraError, match=r'pip install "ttucs\[train\]"'):
        require_train_extras()
    assert main(["train", "-c", str(ROOT / "configs" / "smoke.yaml")]) == 2
    assert 'pip install "ttucs[train]"' in capsys.readouterr().err


def test_release_sha_pinned_consistently():
    pinned = (ROOT / "packages" / "ttucs-model-bow" / "RELEASE_SHA256").read_text().split()[0]
    assert pinned == hub.MODELS["bow"].sha256 and len(pinned) == 64
    version = (ROOT / "packages" / "ttucs-model-bow" / "pyproject.toml").read_text()
    assert f'version = "{hub.MODELS["bow"].version}"' in version
