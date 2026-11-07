import pytest

from ttucs.config import DEFAULTS, flatten, load_config


def test_defaults_and_overrides():
    cfg = load_config(None, ["train.epochs=3", "data.label_documents=false", "model.char_ngrams=[2, 4]"])
    assert cfg["train"]["epochs"] == 3
    assert cfg["data"]["label_documents"] is False
    assert cfg["model"]["char_ngrams"] == [2, 4]
    assert DEFAULTS["train"]["epochs"] != 3  # defaults untouched


def test_yaml_merge():
    cfg = load_config("configs/minilm.yaml")
    assert cfg["model"]["name"] == "encoder"
    assert cfg["data"]["synthetic"]["target_per_class"] == DEFAULTS["data"]["synthetic"]["target_per_class"]


def test_bad_override():
    with pytest.raises(ValueError):
        load_config(None, ["train.epochs"])


def test_flatten():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
