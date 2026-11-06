# Contributing

1. `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pre-commit install`
2. Branch from `main` and use [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).
3. `make lint test smoke` must pass. Add tests for new behaviour.
4. Add a line under `## [Unreleased]` in `CHANGELOG.md`.
5. Never commit data, models, `mlruns/`, `mlflow.db` or credentials. BOOM ONE metadata is
   (c) BOOM Library and must not be added to the repository.

Adding a model: subclass `ttucs.models.base.UCSModel`, decorate it with `@register("name")`, import it in
`ttucs/models/__init__.py`, and add `configs/<name>.yaml`.
