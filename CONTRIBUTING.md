# Contributing

1. `python -m venv .venv && source .venv/bin/activate`, then
   `pip install -e packages/ttucs-model-bow -e ".[dev]" && pre-commit install`.
   ttucs depends on the `ttucs-model-bow` weights distribution. Installing the local copy first keeps
   pip off PyPI for it. The local copy has no weights until you run `scripts/build_model_package.py`,
   and tests that need real weights skip.
2. Branch from `main` and use [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).
3. `make lint test smoke` must pass. Add tests for new behaviour.
4. Add a line under `## [Unreleased]` in `CHANGELOG.md`.
5. Never commit data, models, weights (`*.pt`, `*.safetensors`), `mlruns/`, `mlflow.db` or credentials.
   Third-party sound library metadata must not be added to the repository or to published weights.

Adding a model: subclass `ttucs.models.base.UCSModel`, decorate it with `@register("name")`, import it in
`ttucs/models/__init__.py`, and add `configs/<name>.yaml`.

> The CI and publish workflows live in `ci/` (`github-actions-ci.yml`, `publish.yml`). Move them to
> `.github/workflows/` to enable them, because the automation token was not allowed to push workflow
> files. Releases: see [docs/RELEASING.md](https://github.com/SoundsLikeJonny/ttucs/blob/main/docs/RELEASING.md).
