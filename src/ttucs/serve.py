"""Local HTTP API for the ttucs engine: ``ttucs serve``.

One model is loaded once and shared by every client on the machine, so a desktop UI and a DAW
extension can both tag recordings without paying a process start per call. The contract is
documented in ``docs/serve-api.md``.

Properties this module guarantees:

* loopback only - a non-loopback ``host`` is refused;
* a bearer token (``X-Ttucs-Token``) on every route except ``/health``;
* no CORS headers, so a web page cannot reach the engine even with the token;
* no network access of its own - model resolution still honours ``TTUCS_OFFLINE=1``;
* one lock around inference, because a model instance is not safe to call concurrently.

Nothing here touches the network at import time.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse

from ttucs._version import __version__
from ttucs.hub import ChecksumError, cache_root, is_offline, verify_manifest
from ttucs.pipeline import TTUCS
from ttucs.taxonomy import UCSEntry, UCSTaxonomy

log = logging.getLogger(__name__)

#: Bumped only for a breaking change to a request or response shape.
SCHEMA_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
AUTH_HEADER = "X-Ttucs-Token"
ENDPOINT_FILENAME = "serve.json"
MAX_BODY_BYTES = 1 << 20
MAX_TEXTS = 512
MAX_TOP_K = 50
MAX_SEARCH_LIMIT = 200
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


# --------------------------------------------------------------------------- endpoint file
def endpoint_path() -> Path:
    """Default location of the endpoint file: ``$TTUCS_HOME/serve.json`` or the cache dir."""
    return cache_root() / ENDPOINT_FILENAME


def write_endpoint(path: str | Path, *, host: str, port: int, token: str | None,
                   pid: int | None = None) -> Path:
    """Write the endpoint file a client reads to find a running server (mode ``0600``)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "host": host,
        "port": int(port),
        "token": token,
        "pid": int(pid if pid is not None else os.getpid()),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ttucs_version": __version__,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - best effort on Windows
        log.debug("could not chmod %s", target)
    return target


def read_endpoint(path: str | Path | None = None) -> dict[str, Any] | None:
    """Read the endpoint file, or ``None`` when it is missing or unreadable.

    Liveness is decided by ``GET /health``, never by the recorded ``pid``: ``os.kill(pid, 0)``
    terminates the target process on Windows, so a pid probe is not portable.
    """
    target = Path(path).expanduser() if path is not None else endpoint_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "port" not in data:
        return None
    return data


def clear_endpoint(path: str | Path | None = None) -> None:
    """Remove the endpoint file if present."""
    target = Path(path).expanduser() if path is not None else endpoint_path()
    with contextlib.suppress(FileNotFoundError):
        target.unlink()


def endpoint_url(endpoint: dict[str, Any]) -> str:
    """``http://host:port`` for an endpoint dict, bracketing IPv6 literals."""
    host = str(endpoint.get("host") or DEFAULT_HOST)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(endpoint['port'])}"


# --------------------------------------------------------------------------- taxonomy search
def search_taxonomy(taxonomy: UCSTaxonomy, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Rank UCS entries against ``query``; every token must appear somewhere in the entry."""
    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    q = query.strip().lower()
    if not q:
        return [e.to_dict() for e in taxonomy.entries[:limit]]
    tokens = q.split()
    scored: list[tuple[int, str, UCSEntry]] = []
    for entry in taxonomy.entries:
        haystack = " ".join((entry.cat_id, entry.category, entry.subcategory, entry.cat_short,
                             entry.explanation, *entry.synonyms)).lower()
        if not all(t in haystack for t in tokens):
            continue
        score = 0
        if entry.cat_id.lower() == q:
            score += 100
        elif entry.cat_id.lower().startswith(q):
            score += 50
        if entry.cat_short.lower().startswith(q):
            score += 20
        if entry.category.lower().startswith(q):
            score += 10
        if entry.subcategory.lower().startswith(q):
            score += 10
        scored.append((score, entry.cat_id, entry))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [entry.to_dict() for _, _, entry in scored[:limit]]


# --------------------------------------------------------------------------- server state
@dataclass
class _App:
    pipe: TTUCS
    token: str | None
    top_k: int
    lock: threading.Lock
    warm: bool
    manifest_verified: bool | None = None
    manifest_error: str | None = None


class _HttpError(Exception):
    def __init__(self, status: HTTPStatus, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _manifest_status(pipe: TTUCS) -> tuple[bool | None, str | None]:
    """Verify the weights manifest once at startup; ``None`` when there is no manifest."""
    if pipe.model_path is None or not (Path(pipe.model_path) / "manifest.json").is_file():
        return None, None
    try:
        verify_manifest(pipe.model_path)
    except ChecksumError as exc:
        return False, str(exc)
    return True, None


def _health(app: _App) -> dict[str, Any]:
    pipe = app.pipe
    meta = pipe.meta or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "warm": app.warm,
        "ttucs_version": __version__,
        "ucs_version": pipe.taxonomy.version,
        "model": {
            "name": meta.get("model_name"),
            "version": meta.get("model_version"),
            "source": pipe.model_source,
            "path": str(pipe.model_path) if pipe.model_path else None,
            "manifest_verified": app.manifest_verified,
            "manifest_error": app.manifest_error,
        },
        "offline": is_offline(),
        "device": "cpu",
        "unknown_threshold": pipe.unknown_threshold,
        "category_threshold": pipe.category_threshold,
        "top_k": app.top_k,
        "limits": {"max_texts": MAX_TEXTS, "max_body_bytes": MAX_BODY_BYTES, "max_top_k": MAX_TOP_K},
    }


# --------------------------------------------------------------------------- request handler
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"ttucs/{__version__}"
    sys_version = ""

    @property
    def app(self) -> _App:
        return cast("TTUCSHTTPServer", self.server).app

    # ------------------------------------------------------------------ plumbing
    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _error(self, status: HTTPStatus, detail: str) -> None:
        self._send(status, {"schema_version": SCHEMA_VERSION, "error": status.phrase.lower(),
                            "detail": detail})

    def _authorised(self) -> bool:
        if self.app.token is None:
            return True
        got = self.headers.get(AUTH_HEADER, "")
        if hmac.compare_digest(got, self.app.token):
            return True
        self._error(HTTPStatus.UNAUTHORIZED,
                    f"missing or wrong {AUTH_HEADER} header")
        return False

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise _HttpError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "Content-Length is not an integer") from None
        if length < 0 or length > MAX_BODY_BYTES:
            raise _HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                             f"body must be at most {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "body is not valid JSON") from None
        if not isinstance(data, dict):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return data

    def _texts(self, data: dict[str, Any]) -> list[str]:
        texts = data.get("texts")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "'texts' must be a non-empty list of strings")
        if len(texts) > MAX_TEXTS:
            raise _HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                             f"at most {MAX_TEXTS} texts per request")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "every entry in 'texts' must be a non-empty string")
        return list(texts)

    def _top_k(self, data: dict[str, Any]) -> int:
        raw = data.get("top_k", self.app.top_k)
        if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= MAX_TOP_K:
            raise _HttpError(HTTPStatus.BAD_REQUEST, f"'top_k' must be an integer 1..{MAX_TOP_K}")
        return raw

    @staticmethod
    def _optional_str(data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise _HttpError(HTTPStatus.BAD_REQUEST, f"'{key}' must be a string or null")
        return value

    # ------------------------------------------------------------------ routes
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send(HTTPStatus.OK, _health(self.app))
                return
            if not self._authorised():
                return
            if parsed.path == "/taxonomy/search":
                self._taxonomy_search(parse_qs(parsed.query))
                return
            if parsed.path.startswith("/taxonomy/"):
                self._taxonomy_entry(unquote(parsed.path[len("/taxonomy/"):]))
                return
            self._error(HTTPStatus.NOT_FOUND, f"no route for GET {parsed.path}")
        except _HttpError as exc:
            self._error(exc.status, exc.detail)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if not self._authorised():
                return
            if parsed.path == "/predict":
                self._predict()
            elif parsed.path == "/parse":
                self._parse()
            elif parsed.path == "/shutdown":
                self._send(HTTPStatus.OK, {"schema_version": SCHEMA_VERSION, "ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._error(HTTPStatus.NOT_FOUND, f"no route for POST {parsed.path}")
        except _HttpError as exc:
            self._error(exc.status, exc.detail)

    def _predict(self) -> None:
        data = self._body()
        texts = self._texts(data)
        top_k = self._top_k(data)
        creator_id = self._optional_str(data, "creator_id")
        source_id = self._optional_str(data, "source_id")
        with self.app.lock:
            predictions = self.app.pipe.predict_batch(texts, top_k=top_k, creator_id=creator_id,
                                                      source_id=source_id)
        self._send(HTTPStatus.OK, {"schema_version": SCHEMA_VERSION,
                                   "predictions": [p.to_dict() for p in predictions]})

    def _parse(self) -> None:
        data = self._body()
        texts = self._texts(data)
        self._send(HTTPStatus.OK, {"schema_version": SCHEMA_VERSION,
                                   "slates": [self.app.pipe.parse(t).to_dict() for t in texts]})

    def _taxonomy_search(self, query: dict[str, list[str]]) -> None:
        q = (query.get("q") or [""])[0]
        raw_limit = (query.get("limit") or ["20"])[0]
        try:
            limit = int(raw_limit)
        except ValueError:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "'limit' must be an integer") from None
        entries = search_taxonomy(self.app.pipe.taxonomy, q, limit)
        self._send(HTTPStatus.OK, {"schema_version": SCHEMA_VERSION, "query": q, "entries": entries})

    def _taxonomy_entry(self, cat_id: str) -> None:
        entry = self.app.pipe.taxonomy.resolve(cat_id)
        if entry is None:
            raise _HttpError(HTTPStatus.NOT_FOUND, f"unknown UCS CatID {cat_id!r}")
        self._send(HTTPStatus.OK, {"schema_version": SCHEMA_VERSION, "entry": entry.to_dict()})


class TTUCSHTTPServer(ThreadingHTTPServer):
    """Threaded loopback server; inference itself is serialised by ``app.lock``."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: _App) -> None:
        super().__init__(address, _Handler)
        self.app = app


# --------------------------------------------------------------------------- entry points
def create_server(pipe: TTUCS, *, host: str = DEFAULT_HOST, port: int = 0, token: str | None = None,
                  top_k: int = 5, warm: bool = True,
                  manifest_verified: bool | None = None,
                  manifest_error: str | None = None) -> TTUCSHTTPServer:
    """Build (but do not start) a server bound to ``host:port``; ``port=0`` picks a free port."""
    if host not in _LOOPBACK:
        raise ValueError(f"ttucs serve binds loopback only, refusing host {host!r}")
    app = _App(pipe=pipe, token=token, top_k=int(top_k), lock=threading.Lock(), warm=warm,
               manifest_verified=manifest_verified, manifest_error=manifest_error)
    return TTUCSHTTPServer((host, int(port)), app)


def run_serve(model: str | Path | None = None, *, host: str = DEFAULT_HOST, port: int = 0,
              token_file: str | Path | None = None, top_k: int = 5, warm: bool = True,
              no_token: bool = False) -> int:
    """Load the model, serve until interrupted, and clean up the endpoint file."""
    if host not in _LOOPBACK:
        raise ValueError(f"ttucs serve binds loopback only, refusing host {host!r}")
    pipe = TTUCS.from_pretrained(model)
    if warm:
        pipe.predict("test tone", top_k=1)
    verified, error = _manifest_status(pipe)
    token = None if no_token else secrets.token_urlsafe(32)
    if no_token:
        log.warning("running without a token: any local process can drive the engine")
    server = create_server(pipe, host=host, port=port, token=token, top_k=top_k, warm=warm,
                           manifest_verified=verified, manifest_error=error)
    path = write_endpoint(token_file if token_file is not None else endpoint_path(),
                          host=host, port=server.server_address[1], token=token)
    log.info("ttucs serve listening on %s (endpoint file %s)",
             endpoint_url({"host": host, "port": server.server_address[1]}), path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("ttucs serve interrupted")
    finally:
        server.server_close()
        clear_endpoint(path)
    return 0
