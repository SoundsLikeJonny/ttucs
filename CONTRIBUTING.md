# Contributing

1. Set up a virtualenv and install the local, weightless `ttucs-model-bow` first, then the dev
   extras. ttucs depends on the `ttucs-model-bow` weights distribution, so installing the local copy
   first keeps pip off PyPI for it. The local copy has no weights until you run
   `scripts/build_model_package.py`; for real predictions, point at a trained model dir
   (`TTUCS.from_pretrained("models/...")`) or `pip install ttucs-model-bow`. No test needs the real
   weights: they train tiny models in-test and bypass the installed package, so `pytest` runs fine
   without them.

   macOS / Linux (bash):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e packages/ttucs-model-bow -e ".[dev]"
   pre-commit install
   ```

   Windows (PowerShell 5.1, one command per line):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e packages/ttucs-model-bow -e ".[dev]"
   pre-commit install
   ```

   If activation fails with "running scripts is disabled on this system", run
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first; it only applies to that window.

   For an exactly pinned environment, use `requirements.txt` instead. It is a generated lock of the
   `[dev]` dependencies for Python 3.10+ on Windows, macOS and Linux, with CPU-only PyTorch:
   ```bash
   pip install -r requirements.txt
   pip install --no-deps -e packages/ttucs-model-bow -e .
   ```
   It is not used by the package itself (`pyproject.toml` keeps the loose ranges). After changing
   dependencies in `pyproject.toml`, regenerate it with the `uv pip compile` command in its header.
2. Branch from `main` and use [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).
3. `make lint test smoke` must pass. Add tests for new behaviour.
4. Add a line under `## [Unreleased]` in `CHANGELOG.md`.
5. Never commit data, models, weights (`*.pt`, `*.safetensors`), `mlruns/`, `mlflow.db` or credentials.
   Third-party sound library metadata must not be added to the repository or to published weights.

Adding a model: subclass `ttucs.models.base.UCSModel`, decorate it with `@register("name")`, import it in
`ttucs/models/__init__.py`, and add `configs/<name>.yaml`.

> CI and publishing run from `.github/workflows/` (`ci.yml`, `publish.yml`). Releases: see [docs/RELEASING.md](https://github.com/SoundsLikeJonny/ttucs/blob/main/docs/RELEASING.md).
