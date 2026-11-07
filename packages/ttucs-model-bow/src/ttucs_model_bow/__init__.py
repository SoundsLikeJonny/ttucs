"""Data-only weights package for ttucs (model ``bow``).

Installed automatically by ``pip install ttucs``. ttucs finds it via ``importlib.resources``;
nothing here touches the network.
"""

from importlib.resources import files

__version__ = "0.2.0"
#: Traversable pointing at the model directory (config.json, weights.safetensors, manifest.json).
MODEL_DIR = files(__name__) / "model"

__all__ = ["MODEL_DIR", "__version__"]
