"""Helpful errors for optional dependency groups (``pip install "ttucs[train]"``)."""

from __future__ import annotations

import importlib.util

TRAIN_MODULES = {"sklearn": "scikit-learn", "matplotlib": "matplotlib", "pandas": "pandas"}


class MissingExtraError(ImportError):
    """Raised when a feature needs an optional dependency group that is not installed."""


def missing(modules: dict[str, str]) -> list[str]:
    return [dist for mod, dist in modules.items() if importlib.util.find_spec(mod) is None]


def require_train_extras(feature: str = "training") -> None:
    """Fail early (and clearly) when ``ttucs[train]`` is not installed."""
    absent = missing(TRAIN_MODULES)
    if absent:
        raise MissingExtraError(
            f"{feature} needs the optional training dependencies ({', '.join(absent)} not installed).\n"
            '  Install them with:  pip install "ttucs[train]"\n'
            "  Inference (ttucs predict / parse / info / download) works without them.")


def require_mlflow(feature: str = "MLflow tracking") -> None:
    if importlib.util.find_spec("mlflow") is None:
        raise MissingExtraError(
            f"{feature} needs MLflow, which is not installed.\n"
            '  Install it with:  pip install "ttucs[train]"   (or "ttucs[mlflow]")\n'
            "  or disable tracking for this run:  -o mlflow.enabled=false")
