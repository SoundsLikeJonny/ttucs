"""MLflow ``pyfunc`` flavour for ttucs models.

Logged models accept a DataFrame with a ``text`` column (slate transcripts or plain
descriptions) and return one row per input::

    import mlflow
    m = mlflow.pyfunc.load_model("models:/ttucs/1")
    m.predict(pd.DataFrame({"text": ["glass bottle smash on concrete"]}))
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd

OUTPUT_COLUMNS = ["cat_id", "category", "subcategory", "cat_short", "confidence", "decision",
                  "description", "fx_name", "filename", "top_k"]


class TTUCSPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        from ttucs.pipeline import TTUCS

        self.pipe = TTUCS.from_pretrained(context.artifacts["model_dir"])

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        texts = model_input["text"].astype(str).tolist()
        top_k = int((params or {}).get("top_k", 5))
        rows = []
        for p in self.pipe.predict_batch(texts, top_k=top_k):
            rows.append({"cat_id": p.cat_id, "category": p.category, "subcategory": p.subcategory,
                         "cat_short": p.cat_short, "confidence": p.confidence, "decision": p.decision,
                         "description": p.description, "fx_name": p.fx_name, "filename": p.filename,
                         "top_k": json.dumps([[a.cat_id, a.confidence] for a in p.alternatives])})
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def log_pyfunc(model_dir: str | Path, registered_model_name: str | None = None,
               example_texts: list[str] | None = None):
    """Log ``model_dir`` as an MLflow pyfunc model with an inferred signature."""
    from mlflow.models import infer_signature

    import ttucs

    example = pd.DataFrame({"text": example_texts or ["wooden door creak slow, take 3"]})
    wrapper = TTUCSPyFunc()

    class _Ctx:  # signature inference without a round-trip through MLflow
        artifacts = {"model_dir": str(model_dir)}

    wrapper.load_context(_Ctx())
    output = wrapper.predict(None, example)
    signature = infer_signature(example, output, params={"top_k": 5})
    pkg_dir = Path(ttucs.__file__).parent
    return mlflow.pyfunc.log_model(
        name="model", python_model=TTUCSPyFunc(), artifacts={"model_dir": str(model_dir)},
        code_paths=[str(pkg_dir)], signature=signature, input_example=example,
        pip_requirements=["torch", "numpy", "pandas", "pyyaml", "scikit-learn", "mlflow"],
        registered_model_name=registered_model_name)
