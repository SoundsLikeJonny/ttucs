"""``ttucs serve``: the loopback HTTP contract other tools (UI, DAW extension) speak."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from ttucs.pipeline import Candidate, Prediction
from ttucs.serve import (
    AUTH_HEADER,
    MAX_TEXTS,
    SCHEMA_VERSION,
    clear_endpoint,
    create_server,
    endpoint_url,
    read_endpoint,
    search_taxonomy,
    write_endpoint,
)
from ttucs.slate import SlateParser

TOKEN = "test-token"


class _StubPipe:
    """Cheap stand-in for ``TTUCS`` so the HTTP contract is tested without a trained model."""

    def __init__(self, taxonomy) -> None:
        self.taxonomy = taxonomy
        self.meta = {"model_name": "stub", "model_version": "0.0.0"}
        self.model_path = None
        self.model_source = "path"
        self.unknown_threshold = 0.12
        self.category_threshold = 0.5
        self.parser = SlateParser()

    def parse(self, text: str):
        return self.parser.parse(text)

    def predict_batch(self, texts, top_k: int = 5, creator_id=None, source_id=None):
        return [self._one(t, top_k) for t in texts]

    def _one(self, text: str, top_k: int) -> Prediction:
        alts = [Candidate("DOORCreak", "DOOR", "Creak", "Creak", 0.9),
                Candidate("DOORMisc", "DOOR", "MISC", "Misc", 0.05),
                Candidate("GLASBrk", "GLAS", "Break", "Break", 0.02)][:max(1, top_k)]
        return Prediction(
            text=text, description=text, cat_id="DOORCreak", category="DOOR", subcategory="Creak",
            cat_short="Creak", confidence=0.9, category_confidence=0.95, decision="confident",
            alternatives=alts, categories=[{"category": "DOOR", "confidence": 0.95}],
            slate=self.parser.parse(text).to_dict(), fx_name="Wooden Door Creak",
            keywords=["door", "creak"], filename="DOORCreak_Wooden Door Creak_NONE_NONE",
            filename_parts={"cat_id": "DOORCreak"})


@pytest.fixture
def server(taxonomy):
    srv = create_server(_StubPipe(taxonomy), port=0, token=TOKEN, top_k=3)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(server, method: str, path: str, body=None, token: str | None = TOKEN):
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header(AUTH_HEADER, token)
    try:
        with _opener().open(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --------------------------------------------------------------------------- health
def test_health_needs_no_token(server):
    status, body = call(server, "GET", "/health", token=None)
    assert status == 200
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["status"] == "ok" and body["warm"] is True
    assert body["model"]["name"] == "stub"
    assert body["model"]["manifest_verified"] is None
    assert body["device"] == "cpu"
    assert body["limits"]["max_texts"] == MAX_TEXTS
    assert isinstance(body["offline"], bool)


# --------------------------------------------------------------------------- auth
def test_predict_requires_token(server):
    status, body = call(server, "POST", "/predict", {"texts": ["door creak"]}, token=None)
    assert status == 401
    assert body["schema_version"] == SCHEMA_VERSION
    status, _ = call(server, "POST", "/predict", {"texts": ["door creak"]}, token="wrong")
    assert status == 401


def test_unauthorised_request_with_a_large_body_still_answers(server):
    # The body is drained before the 401, otherwise Windows resets the connection instead.
    status, body = call(server, "POST", "/predict", {"texts": ["x" * (256 << 10)]}, token=None)
    assert status == 401
    assert body["schema_version"] == SCHEMA_VERSION


def test_no_token_mode_allows_anonymous(taxonomy):
    srv = create_server(_StubPipe(taxonomy), port=0, token=None)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = call(srv, "POST", "/predict", {"texts": ["door creak"]}, token=None)
        assert status == 200 and len(body["predictions"]) == 1
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- predict
def test_predict_single_returns_a_list(server):
    status, body = call(server, "POST", "/predict", {"texts": ["door creak slow"]})
    assert status == 200
    assert body["schema_version"] == SCHEMA_VERSION
    assert isinstance(body["predictions"], list) and len(body["predictions"]) == 1
    pred = body["predictions"][0]
    for key in ("text", "description", "cat_id", "category", "subcategory", "cat_short",
                "category_full", "confidence", "category_confidence", "decision", "alternatives",
                "categories", "slate", "fx_name", "keywords", "filename", "filename_parts",
                "is_unknown"):
        assert key in pred, key
    assert pred["cat_id"] == "DOORCreak" and pred["is_unknown"] is False
    assert len(pred["alternatives"]) == 3  # server default top_k


def test_predict_batch_preserves_order_and_top_k(server):
    texts = ["glass smash", "door creak", "dog bark"]
    status, body = call(server, "POST", "/predict", {"texts": texts, "top_k": 2})
    assert status == 200
    assert [p["text"] for p in body["predictions"]] == texts
    assert all(len(p["alternatives"]) == 2 for p in body["predictions"])


def test_predict_accepts_a_bare_string(server):
    status, body = call(server, "POST", "/predict", {"texts": "door creak"})
    assert status == 200 and len(body["predictions"]) == 1


@pytest.mark.parametrize("payload", [
    {},
    {"texts": []},
    {"texts": [""]},
    {"texts": [1]},
    {"texts": ["ok"], "top_k": 0},
    {"texts": ["ok"], "top_k": 999},
    {"texts": ["ok"], "top_k": "3"},
    {"texts": ["ok"], "creator_id": 5},
])
def test_predict_rejects_bad_input(server, payload):
    status, body = call(server, "POST", "/predict", payload)
    assert status == 400, body
    assert body["schema_version"] == SCHEMA_VERSION


def test_predict_rejects_too_many_texts(server):
    status, body = call(server, "POST", "/predict", {"texts": ["door"] * (MAX_TEXTS + 1)})
    assert status == 413
    assert "texts" in body["detail"]


def test_predict_rejects_oversized_body(server):
    status, _body = call(server, "POST", "/predict", {"texts": ["x" * (1 << 20)]})
    assert status == 413


def test_unknown_route_is_404(server):
    status, _ = call(server, "GET", "/nope")
    assert status == 404
    status, _ = call(server, "POST", "/nope", {})
    assert status == 404


# --------------------------------------------------------------------------- parse + taxonomy
def test_parse_returns_slate_fields(server):
    status, body = call(server, "POST", "/parse",
                        {"texts": ["project nightfall, glass smash, take two"]})
    assert status == 200
    slate = body["slates"][0]
    assert slate["project"] == "Nightfall" and slate["take"] == "2"


def test_taxonomy_search(server):
    status, body = call(server, "GET", "/taxonomy/search?q=door%20creak&limit=5")
    assert status == 200
    assert 0 < len(body["entries"]) <= 5
    assert all("door" in json.dumps(e).lower() for e in body["entries"])


def test_taxonomy_entry_and_404(server):
    status, body = call(server, "GET", "/taxonomy/DOORCreak")
    assert status == 200 and body["entry"]["cat_id"] == "DOORCreak"
    status, body = call(server, "GET", "/taxonomy/NOTACAT")
    assert status == 404


def test_search_taxonomy_ranks_exact_catid_first(taxonomy):
    hits = search_taxonomy(taxonomy, "DOORCreak", limit=3)
    assert hits[0]["cat_id"] == "DOORCreak"
    assert search_taxonomy(taxonomy, "", limit=2) == [e.to_dict() for e in taxonomy.entries[:2]]


# --------------------------------------------------------------------------- lifecycle
def test_shutdown_stops_the_server(taxonomy):
    srv = create_server(_StubPipe(taxonomy), port=0, token=TOKEN)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    status, body = call(srv, "POST", "/shutdown", {})
    assert status == 200 and body["ok"] is True
    thread.join(timeout=10)
    assert not thread.is_alive()
    srv.server_close()


def test_create_server_refuses_non_loopback(taxonomy):
    with pytest.raises(ValueError, match="loopback"):
        create_server(_StubPipe(taxonomy), host="0.0.0.0", token=TOKEN)


def test_endpoint_file_roundtrip(tmp_path):
    path = tmp_path / "nested" / "serve.json"
    assert read_endpoint(path) is None
    write_endpoint(path, host="127.0.0.1", port=8731, token=TOKEN, pid=4242)
    data = read_endpoint(path)
    assert data["port"] == 8731 and data["token"] == TOKEN and data["pid"] == 4242
    assert data["schema_version"] == SCHEMA_VERSION
    assert endpoint_url(data) == "http://127.0.0.1:8731"
    assert endpoint_url({"host": "::1", "port": 1}) == "http://[::1]:1"
    clear_endpoint(path)
    assert read_endpoint(path) is None
    clear_endpoint(path)  # idempotent


def test_endpoint_file_survives_garbage(tmp_path):
    path = tmp_path / "serve.json"
    path.write_text("not json", encoding="utf-8")
    assert read_endpoint(path) is None
    path.write_text(json.dumps({"host": "127.0.0.1"}), encoding="utf-8")
    assert read_endpoint(path) is None


# --------------------------------------------------------------------------- real pipeline
@pytest.fixture(scope="module")
def tiny_model_dir(taxonomy, tmp_path_factory):
    import torch

    from ttucs.datasets.build import generate_synthetic, label_documents
    from ttucs.models import MODEL_REGISTRY

    labels = ["DOORWood", "GLASBrk", "ANMLDog"]
    cats = [taxonomy[c].category for c in labels]
    data = generate_synthetic(taxonomy, {c: 40 for c in labels}, seed=0) + label_documents(taxonomy, labels)
    pairs = [(e.text, e.label) for e in data]
    torch.manual_seed(0)
    model = MODEL_REGISTRY["bow"](labels, cats, {"embedding_dim": 32, "num_buckets": 32768})
    model.fit(pairs, pairs[:30], {"epochs": 4, "batch_size": 32, "lr": 1e-2, "seed": 0,
                                  "num_threads": 2})
    out = tmp_path_factory.mktemp("serve-model")
    model.save(out, extra={"temperature": 1.0, "ucs_version": "8.2.1",
                           "inference": {"unknown_threshold": 0.3}})
    return out


@pytest.mark.slow
def test_real_pipeline_over_http(tiny_model_dir):
    from ttucs.pipeline import TTUCS

    pipe = TTUCS.from_pretrained(tiny_model_dir)
    srv = create_server(pipe, port=0, token=TOKEN)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = call(srv, "POST", "/predict",
                            {"texts": ["project thick as thieves, a dog barking loudly, take 3"]})
        assert status == 200
        pred = body["predictions"][0]
        assert pred["cat_id"] == "ANMLDog"
        assert pred["filename"].startswith(pred["cat_id"] + "_")
        assert pred["slate"]["project"] == "Thick As Thieves"
        status, health = call(srv, "GET", "/health", token=None)
        assert health["model"]["name"] == "bow" and health["model"]["source"] == "path"
        assert health["model"]["manifest_verified"] is None  # no manifest.json on a trained dir
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


@pytest.mark.slow
def test_predict_json_is_always_a_list(tiny_model_dir, capsys):
    from ttucs.cli import main

    assert main(["predict", "a dog barking", "--json", "-m", str(tiny_model_dir)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out, list) and len(out) == 1 and out[0]["cat_id"] == "ANMLDog"

