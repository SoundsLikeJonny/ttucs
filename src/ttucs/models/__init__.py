"""ttucs models. Importing this package registers every built-in model in ``MODEL_REGISTRY``."""

from ttucs.models import bow, encoder, zeroshot  # noqa: F401  (registration side effects)
from ttucs.models.base import MODEL_REGISTRY, UCSModel, fit_temperature, register

__all__ = ["MODEL_REGISTRY", "UCSModel", "fit_temperature", "register"]
