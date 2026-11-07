"""Build the data-only ``ttucs-model-bow`` distribution (and its GitHub Release tarball).

Weights never go into git. Two ways to fill ``packages/ttucs-model-bow/src/ttucs_model_bow/model``:

1. From a trained model directory (local release build)::

       ttucs train -c configs/bow-release.yaml
       python scripts/build_model_package.py --model-dir models/ttucs-bow-release --tarball --wheel

   This exports fp16 ``weights.safetensors`` + ``config.json`` + ``manifest.json``, writes the
   deterministic ``dist/model-bow-v<version>.tar.gz`` (upload it to the GitHub Release
   ``model-bow-v<version>``) and builds the sdist + wheel into ``dist/``.

2. From the published Release tarball (what CI does)::

       python scripts/build_model_package.py --from-tarball URL_OR_PATH --sha256 HEX --wheel

The build refuses models whose ``config.json`` lists training sources other than the
redistributable ones (synthetic data and UCS label documents) unless ``--allow-other-sources`` is
given for a private, never-published build.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packages" / "ttucs-model-bow"
MODEL_DIR = PKG / "src" / "ttucs_model_bow" / "model"
MAX_WHEEL_MB = 90
PUBLISHABLE_SOURCES = {"synthetic", "ucs_doc"}  # PyPI's default per-file limit is 100 MB


def package_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (PKG / "pyproject.toml").read_text(), re.M)
    if not m:
        sys.exit("no version in packages/ttucs-model-bow/pyproject.toml")
    return m.group(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_tarball(model_dir: Path, out: Path, root_name: str) -> Path:
    """tar.gz with sorted members, zero mtimes and no owner info -> reproducible sha256."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in sorted(model_dir.iterdir()):
            if not p.is_file():
                continue
            info = tarfile.TarInfo(f"{root_name}/{p.name}")
            info.size, info.mtime, info.mode = p.stat().st_size, 0, 0o644
            with open(p, "rb") as fh:
                tar.addfile(info, fh)
    out.parent.mkdir(parents=True, exist_ok=True)
    # compress into memory: GzipFile would otherwise embed the output file's path in the header
    gz_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue())
    out.write_bytes(gz_buf.getvalue())
    return out


def from_trained(model_dir: Path, version: str, dtype: str, allow_other: bool) -> None:
    from ttucs.hub import export_model

    meta = json.loads((model_dir / "config.json").read_text())
    td = meta.get("training_data")
    if td is None and not allow_other:
        sys.exit(f"{model_dir}/config.json has no 'training_data' record - retrain with ttucs>=0.2.0 "
                 "so the data provenance can be checked")
    extra = set((td or {}).get("sources", [])) - PUBLISHABLE_SOURCES
    if extra and not allow_other:
        sys.exit(f"refusing to package a model trained on {sorted(extra)}: only "
                 f"{sorted(PUBLISHABLE_SOURCES)} are known to be redistributable. "
                 "Train with configs/bow-release.yaml, or pass --allow-other-sources for a private build.")
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    export_model(model_dir, MODEL_DIR, dtype=dtype, extra_meta={
        "model_version": version, "weights_license": "MIT",
        "published_as": f"ttucs-model-bow=={version}"})


def from_tarball(src: str, expected: str) -> None:
    from ttucs.hub import verify_manifest

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.tar.gz"
        if src.startswith(("https://", "http://")):
            with urllib.request.urlopen(src, timeout=120) as resp, open(archive, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        else:
            shutil.copy(src, archive)
        got = sha256(archive)
        if got != expected.lower():
            sys.exit(f"sha256 mismatch for {src}: got {got}, expected {expected}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(Path(tmp) / "x", filter="data")
        found = [p.parent for p in (Path(tmp) / "x").rglob("config.json")]
        if len(found) != 1:
            sys.exit("archive must contain exactly one model directory")
        verify_manifest(found[0])
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        shutil.copytree(found[0], MODEL_DIR)


def build_dists(outdir: Path) -> list[Path]:
    subprocess.run([sys.executable, "-m", "build", str(PKG), "--outdir", str(outdir)], check=True)
    built = sorted(outdir.glob("ttucs_model_bow-*"))
    for w in built:
        mb = w.stat().st_size / 1e6
        print(f"{w.name}: {mb:.1f} MB")
        if mb > MAX_WHEEL_MB:
            sys.exit(f"{w.name} is {mb:.1f} MB - too close to PyPI's 100 MB file limit")
    return built


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", type=Path, help="trained model directory (ttucs train output)")
    src.add_argument("--from-tarball", help="release tarball URL or path (needs --sha256)")
    ap.add_argument("--sha256", help="expected sha256 of --from-tarball")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tarball", action="store_true", help="also write dist/model-bow-v<version>.tar.gz")
    ap.add_argument("--wheel", action="store_true", help="build sdist + wheel with python -m build")
    ap.add_argument("--outdir", type=Path, default=ROOT / "dist")
    ap.add_argument("--allow-other-sources", action="store_true",
                    help="allow non-synthetic training data (private builds only - never publish)")
    args = ap.parse_args()

    version = package_version()
    if args.model_dir:
        from_trained(args.model_dir, version, args.dtype, args.allow_other_sources)
    else:
        if not args.sha256:
            ap.error("--from-tarball needs --sha256")
        from_tarball(args.from_tarball, args.sha256)
    meta = json.loads((MODEL_DIR / "config.json").read_text())
    if meta.get("model_version") != version:
        sys.exit(f"model_version {meta.get('model_version')} != package version {version}")
    print(f"model files in {MODEL_DIR}:")
    for p in sorted(MODEL_DIR.iterdir()):
        print(f"  {p.name:<22} {p.stat().st_size / 1e6:8.2f} MB")
    if args.tarball:
        tb = deterministic_tarball(MODEL_DIR, args.outdir / f"model-bow-v{version}.tar.gz", "model-bow")
        digest = sha256(tb)
        (tb.parent / (tb.name + ".sha256")).write_text(f"{digest}  {tb.name}\n")
        print(f"{tb.name}: {tb.stat().st_size / 1e6:.1f} MB  sha256 {digest}")
    if args.wheel:
        build_dists(args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
