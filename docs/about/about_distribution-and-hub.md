# Distribution and the model hub

`ttucs` ships as two PyPI distributions and resolves its weights through `ttucs/hub.py`. This
document explains the split, the resolution order, the integrity checks, and the build script that
produces the release artefacts.

## Why two packages

The code is small; the weights are 67.7 MB. Putting them in one wheel would make every code-only
upgrade re-download the weights, and would make the code wheel large for users who already have a
model. So:

| distribution | import name | contents | versioned by |
|---|---|---|---|
| `ttucs` | `ttucs` | code, UCS CSVs, banks, CLI | `src/ttucs/_version.py` |
| `ttucs-model-bow` | `ttucs_model_bow` | `config.json`, `weights.safetensors`, `manifest.json` | `packages/ttucs-model-bow/pyproject.toml` and `__init__.py` |

`ttucs` depends on `ttucs-model-bow` with a compatible range (`>=0.2.0,<0.3`), so a plain
`pip install ttucs` gets working weights and inference works offline immediately. The two versions
move independently: the weights version bumps only when the weights change.

`ttucs_model_bow.MODEL_DIR` is an `importlib.resources` traversable pointing at the `model/`
directory, so the package works from a wheel, a zip or an editable install without touching the
network.

## Resolution order

`hub.resolve_model(model=None, allow_download=None)` returns a `ResolvedModel(path, source, name)`:

1. **Explicit path** — if `model` is a `Path` or a string that is not a known model name. The
   directory must contain `config.json`; otherwise `ModelNotFoundError` explains what was wrong.
2. **Installed package** — `packaged_model_dir(name)` via `importlib.resources`; `source="package"`.
3. **Cache** — `cache_root()/models/<name>/<version>`; `source="cache"`.
4. **Download** — only when not offline and `allow_download` is true; `source="download"`.
5. Otherwise `ModelNotFoundError` listing every place it looked and the three ways to fix it.

`cache_root()` is `$TTUCS_HOME` if set, else `platformdirs.user_cache_dir("ttucs")`.

`TTUCS.from_pretrained()` calls this, then `check_compatibility`, then `UCSModel.load`.

## Offline mode

```python
def is_offline():
    return any(os.environ.get(v, "").strip().lower() in {"1", "true", "yes", "on"}
               for v in ("TTUCS_OFFLINE", "HF_HUB_OFFLINE"))
```

When offline:

- `resolve_model` never downloads; it raises `ModelNotFoundError` with an explicit "offline mode is
  ON" line;
- `download_model` raises `OfflineError` before any socket is opened;
- `train.resolve_tracking_uri` ignores `MLFLOW_TRACKING_URI` and forces the local SQLite store.

Nothing in the package touches the network at import time or during prediction. The only network
code in the library is `download_model`, and it is reachable only through an explicit call.

## Integrity

### `manifest.json`

`write_manifest(directory)` records the sha256 and byte size of every file in the model directory.
`verify_manifest(directory)` recomputes them and raises `ChecksumError` on a mismatch. It is checked:

- after a download, before the model is moved into place;
- by `ttucs info --verify`;
- by `tests/test_hub.py`.

Published manifest (0.2.0):

```json
{"files": {"config.json": {"sha256": "c122fe89...", "bytes": 23476},
           "weights.safetensors": {"sha256": "3834be1e...", "bytes": 67663222}}}
```

### Release tarball sha256

`hub.MODELS["bow"]` pins the release asset:

```python
ModelSpec(name="bow", package="ttucs_model_bow", distribution="ttucs-model-bow",
          version="0.2.0",
          sha256="22e8ac483fa97bbd8152d29bffad60cda668653c7786f8c20e92f90f52913e7b",
          archive="model-bow-v0.2.0.tar.gz")
```

`spec.url` is `https://github.com/SoundsLikeJonny/ttucs/releases/download/model-bow-v0.2.0/model-bow-v0.2.0.tar.gz`.
`packages/ttucs-model-bow/RELEASE_SHA256` holds the same hash in `sha256sum` format, and
`tests/test_hub.py` asserts that the two agree with the package version. That test is what stops a
version bump from shipping with a stale hash.

### Download safety

`download_model`:

- refuses non-`http(s)`/`file` schemes;
- requires a 64-character expected sha256 (`ChecksumError` otherwise);
- streams to a temporary directory next to the target, hashing as it writes;
- compares the hash and discards the file on mismatch;
- extracts with `_safe_extract` (rejects absolute paths and `..` traversal);
- requires exactly one `config.json` in the archive;
- runs `verify_manifest` and `check_compatibility` before moving it into place.

`TTUCS_MODEL_URL` and `TTUCS_MODEL_SHA256` override the pinned values for mirrors or air-gapped
artifact stores; `ttucs download --url ... --sha256 ...` does the same per call.

## Compatibility

`check_compatibility(meta, where)` reads `format_version` and `min_ttucs_version` from the model's
`config.json` and raises `IncompatibleModelError` when:

- `format_version` is newer than `MODEL_FORMAT_VERSION` (1);
- the running `ttucs` is older than `min_ttucs_version`.

This is what lets the directory layout evolve without silently loading a model the code cannot
interpret.

## Building the release artefacts

```bash
python scripts/build_model_package.py --model-dir models/ttucs-bow-release --tarball --wheel
```

1. `hub.export_model(src, dst, dtype="float16")` casts floating-point tensors to fp16, writes
   `weights.safetensors` (no pickle), adds `format_version`, `weights_dtype` and
   `min_ttucs_version` to `config.json`, and writes `manifest.json`.
2. `deterministic_tarball` builds `dist/model-bow-v<version>.tar.gz` with sorted members, zero
   mtimes, mode 0644, no owner info and `gzip mtime=0`, so the sha256 is reproducible.
3. `--wheel` builds the sdist and wheel with `python -m build` and checks the wheel size against
   `MAX_WHEEL_MB = 90` (PyPI's limit is 100 MB).
4. `--from-tarball URL_OR_PATH --sha256 HEX` is the CI path: it verifies a published tarball and
   builds the wheel from it, so CI never needs a trained model.

The current wheel is ~68 MB: 131,072 x 256 fp16 weights plus a 23 KB `config.json`.

## `ttucs info` and `ttucs download`

```bash
ttucs info --verify
ttucs download --model bow
ttucs download --dest /opt/models/bow --force
```

`info` prints the resolved path, source, model name and version, the `ttucs` version that trained
it, the UCS version, the weights dtype, the label count and the training data sources; `--verify`
adds the manifest check and exits 1 on failure. `download` fetches into the cache (or `--dest`) and
verifies.

## Rules that keep this honest

- The sha in the repo must match the exact uploaded file: build, hash, upload, never rebuild after
  pinning.
- The weights release tag must not start with `v` (`model-bow-v0.2.0`), so it cannot be confused
  with a code release tag.
- The weights release is never marked "Latest".
- A weights change updates, together: `hub.MODELS["bow"]`, `RELEASE_SHA256`, the dependency range
  in `pyproject.toml`, `MODEL_CARD.md`, the README results table and the weights package README
  changelog.

Next: [about_development-and-release.md](about_development-and-release.md) covers the process
around all of this.
