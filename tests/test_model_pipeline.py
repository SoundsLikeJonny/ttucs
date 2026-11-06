"""End-to-end: tiny BoW model trained on synthetic data, saved, reloaded and used for inference."""

import json

import numpy as np
import pytest

from ttucs import TTUCS, Prediction
from ttucs.datasets.build import generate_synthetic, label_documents
from ttucs.evaluate import classification_metrics, parser_field_accuracy, token_f1
from ttucs.models import MODEL_REGISTRY, fit_temperature
from ttucs.models.base import UCSModel

LABELS = ["DOORWood", "GLASBrk", "WATRSurf", "ANMLDog", "VEHCar", "FIRECrkl", "DOORMisc"]


@pytest.fixture(scope="module")
def tiny_model(taxonomy, tmp_path_factory):
    cats = [taxonomy[c].category for c in LABELS]
    data = generate_synthetic(taxonomy, {c: 60 for c in LABELS}, seed=0) + label_documents(taxonomy, LABELS)
    pairs = [(e.text, e.label) for e in data]
    import torch

    torch.manual_seed(0)
    model = MODEL_REGISTRY["bow"](LABELS, cats, {"embedding_dim": 32, "num_buckets": 32768})
    model.fit(pairs, pairs[:50], {"epochs": 8, "batch_size": 32, "lr": 1e-2, "seed": 0, "num_threads": 2})
    out = tmp_path_factory.mktemp("model")
    model.save(out, extra={"temperature": 1.0, "ucs_version": "8.2.1",
                           "inference": {"unknown_threshold": 0.3}})
    return out


@pytest.mark.slow
def test_bow_learns_and_reloads(tiny_model):
    pipe = TTUCS.from_pretrained(tiny_model, creator_id="JE")
    pred = pipe.predict("project thick as thieves, recording a dog barking loudly, take 3", top_k=3)
    assert isinstance(pred, Prediction)
    assert pred.alternatives[0].cat_id == "ANMLDog"
    assert len(pred.alternatives) == 3
    assert pred.slate["project"] == "Thick As Thieves"
    assert pred.filename.startswith("ANMLDog_") and pred.filename.endswith("_JE_ThickAsThieves")
    probs = [a.confidence for a in pred.alternatives]
    assert probs == sorted(probs, reverse=True)
    json.loads(pred.to_json())


@pytest.mark.slow
def test_batch_matches_single_and_save_roundtrip(tiny_model, tmp_path):
    pipe = TTUCS.from_pretrained(tiny_model)
    texts = ["glass bottle smash", "waves crashing on the beach", "car driving by"]
    batch = pipe.predict_batch(texts)
    single = [pipe.predict(t).alternatives[0].cat_id for t in texts]
    assert [p.alternatives[0].cat_id for p in batch] == single
    pipe.unknown_threshold = 0.99
    pipe.save_pretrained(tmp_path / "copy")
    again = TTUCS.from_pretrained(tmp_path / "copy")
    assert again.unknown_threshold == 0.99
    np.testing.assert_allclose(again.probabilities(texts), pipe.probabilities(texts), rtol=1e-5)


@pytest.mark.slow
def test_unknown_and_category_fallback(tiny_model):
    strict = TTUCS.from_pretrained(tiny_model, unknown_threshold=1.01, category_threshold=1.01)
    p = strict.predict("door")
    assert p.decision == "unknown" and p.is_unknown
    fallback = TTUCS.from_pretrained(tiny_model, unknown_threshold=1.01, category_threshold=0.0)
    p = fallback.predict("wooden door creak")
    assert p.decision == "category_fallback"
    assert p.cat_id.endswith("Misc") and p.category == p.categories[0]["category"]


def test_zeroshot_model(taxonomy, tmp_path):
    cats = [taxonomy[c].category for c in LABELS]
    m = MODEL_REGISTRY["zeroshot"](LABELS, cats, {})
    m.fit([], [], {})
    logits = m.logits(["dog barking", "fire crackling", "ocean surf breaking"])
    assert [LABELS[i] for i in logits.argmax(1)] == ["ANMLDog", "FIRECrkl", "WATRSurf"]
    m.save(tmp_path)
    m2, _ = UCSModel.load(tmp_path)
    np.testing.assert_allclose(m2.logits(["dog barking"]), m.logits(["dog barking"]), rtol=1e-5)


def test_metrics_and_temperature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    logits = rng.normal(size=(200, 5))
    logits[np.arange(200), y] += 3
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    m = classification_metrics(p, y, np.array([0, 0, 1, 1, 2]), threshold=0.5)
    assert 0.7 < m["top1"] <= m["top5"] == 1.0
    assert m["category_acc"] >= m["top1"]
    assert 0 < m["coverage"] <= 1
    t = fit_temperature(logits * 4, y)
    assert t > 1.5  # over-confident logits get softened


def test_parser_field_accuracy_and_token_f1():
    from ttucs.synth.generator import Example, SlateGenerator

    sg = SlateGenerator(seed=0)
    exs = []
    for d in ["metal gate slam", "dog barking", "rain on a tin roof"] * 10:
        text, truth = sg.wrap(d)
        exs.append(Example(text, "X", "slate", meta={"truth": truth}))
    acc = parser_field_accuracy(exs)
    assert acc["parser_description_token_f1"] > 0.7
    assert token_f1("the door creak", "door creak") == 1.0
