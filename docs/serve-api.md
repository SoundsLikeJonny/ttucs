# The `ttucs serve` HTTP contract

`ttucs serve` exposes the engine over loopback HTTP so other tools can tag recordings without
starting a Python process per call. One model is loaded once and shared by every client on the
machine. This page is the contract: a desktop UI (TypeScript) and a DAW extension (C++) both speak
it, and neither needs Python bindings.

```bash
ttucs serve -m bow --port 0 --token-file "$TTUCS_HOME/serve.json"
```

```powershell
ttucs serve -m bow --port 0 --token-file "$env:TTUCS_HOME\serve.json"
```

Nothing here touches the network: model resolution still honours `TTUCS_OFFLINE=1`, and the server
binds loopback only.

## 1. Lifecycle

1. Read the endpoint file (`$TTUCS_HOME/serve.json`, else the platform cache dir).
2. `GET /health` with no token. If it answers, a server is already running - use it.
3. Otherwise spawn `ttucs serve --port 0 --token-file <path>` detached and poll the file for up to
   30 s, then `/health` until `warm` is `true` (allow 60 s on a cold first run).
4. On exit, `POST /shutdown` **only** if this process started the server.

The endpoint file is written mode `0600` and removed on clean shutdown:

```json
{
  "schema_version": 1,
  "host": "127.0.0.1",
  "port": 55917,
  "token": "<generated, 43 chars>",
  "pid": 5692,
  "started": "2026-11-10T06:42:09Z",
  "ttucs_version": "0.3.0"
}
```

Do **not** decide liveness from `pid`: `os.kill(pid, 0)` terminates the target process on Windows.
Use `/health`.

## 2. Common rules

- Every response is JSON with `schema_version` (integer, currently `1`). Refuse an unknown value
  and say so in the UI instead of guessing.
- Every route except `GET /health` requires `X-Ttucs-Token: <token>`; a wrong or missing token is
  `401`. Compare in constant time.
- No CORS headers are sent, so a web page cannot reach the engine even with the token.
- Connections are closed after each response (`Connection: close`).
- Bodies must be JSON objects, at most 1 MiB, with `Content-Length` set.
- Errors are `{"schema_version": 1, "error": "<phrase>", "detail": "<human readable>"}`.
- Unknown keys in a response are forward-compatible additions: ignore them, never fail.

Limits: 512 texts per request, `top_k` 1-50, taxonomy search limit 1-200.

## 3. Routes

### `GET /health` (no token)

```json
{
  "schema_version": 1,
  "status": "ok",
  "warm": true,
  "ttucs_version": "0.3.0",
  "ucs_version": "8.2.1",
  "model": {
    "name": "bow",
    "version": "0.2.0",
    "source": "package",
    "path": "...",
    "manifest_verified": true,
    "manifest_error": null
  },
  "offline": true,
  "device": "cpu",
  "unknown_threshold": 0.12,
  "category_threshold": 0.5,
  "top_k": 5,
  "limits": {"max_texts": 512, "max_body_bytes": 1048576, "max_top_k": 50}
}
```

`model.version` is `null` for a locally trained directory with no `model_version` in its
`config.json`. `manifest_verified` is `null` when the directory has no `manifest.json`, `false`
with `manifest_error` set when a checksum does not match.

### `POST /predict`

```json
{"texts": ["...", "..."], "top_k": 5, "creator_id": "JE", "source_id": "ThickAsThieves"}
```

`texts` is required; a bare string is accepted as a one-element list. `top_k`, `creator_id` and
`source_id` are optional. The response **always** contains a list, even for one input:

```json
{"schema_version": 1, "predictions": [ /* Prediction objects, in request order */ ]}
```

A `Prediction` is `Prediction.to_dict()` from `src/ttucs/pipeline.py`:

| field | meaning |
|---|---|
| `text`, `description` | the input, and the parsed description the classifier actually saw |
| `cat_id`, `category`, `subcategory`, `cat_short`, `category_full` | the chosen UCS class |
| `confidence`, `category_confidence` | calibrated CatID and aggregated Category probability |
| `decision` | `confident`, `category_fallback` or `unknown` |
| `alternatives` | top-k `{cat_id, category, subcategory, cat_short, confidence}` |
| `categories` | top-k `{category, confidence}` |
| `slate` | every `SlateFields` field (see below) |
| `fx_name`, `keywords` | suggested FX name and keyword list |
| `filename`, `filename_parts` | the suggested UCS filename and its blocks |
| `is_unknown` | `decision == "unknown"` |

`slate` keys: `raw_text`, `normalized_text`, `description`, `project`, `location`, `microphones`,
`mic_types`, `mic_setup`, `recorder`, `sample_rate`, `bit_depth`, `take`, `scene`, `slate`,
`perspective`, `date`, `creator`, `notes`, `unparsed`.

Only `description` is classified. Slate fields are returned for the UI and for the filename; they
never feed the model.

`category_fallback` means the subcategory was unsure but the Category was not, so `cat_id` is that
Category's MISC CatID and the model's best guess is `alternatives[0]`. `unknown` means neither
threshold was met: `cat_id` is still the best guess and the filename should be reviewed.

### `POST /parse`

```json
{"texts": ["project nightfall, glass smash, take two"]}
```

Returns `{"schema_version": 1, "slates": [ /* SlateFields */ ]}`. Needs no model, but the server
has one loaded anyway.

### `GET /taxonomy/search?q=door%20creak&limit=20`

Returns `{"schema_version": 1, "query": "...", "entries": [ /* UCSEntry */ ]}`. Every query token
must appear in the entry (CatID, category, subcategory, CatShort, explanation or synonyms). Exact
CatID ranks first, then CatID prefix, CatShort prefix, category and subcategory prefix. An empty
`q` returns the first `limit` entries.

### `GET /taxonomy/{cat_id}`

Returns `{"schema_version": 1, "entry": { /* UCSEntry */ }}`, or `404` for an unknown CatID.

### `POST /shutdown`

Requires the token. Returns `{"schema_version": 1, "ok": true}` and stops the server, which removes
the endpoint file.

## 4. Status codes

| code | when |
|---|---|
| 200 | success |
| 400 | bad JSON, missing/empty `texts`, non-string entry, `top_k` out of range, bad `limit` |
| 401 | missing or wrong token |
| 404 | unknown route or CatID |
| 411 | no `Content-Length` |
| 413 | body over 1 MiB, or more than 512 texts |

## 5. Client notes

**TypeScript** - `fetch` with `X-Ttucs-Token`; validate every response against a schema (zod) and
reject an unexpected `schema_version`. Batch 32 texts per `/predict` call.

**C++ (REAPER extension)** - never call HTTP on the host's main thread. Use a worker thread and
marshal results back with `defer` / `plugin_register("timer", ...)`. `WinHTTP` on Windows,
`libcurl` elsewhere. Wrap a batch of item writes in `Undo_BeginBlock` / `Undo_EndBlock` so one
undo step covers the batch.

## 6. Versioning

`schema_version` changes only for a breaking request or response change; additive fields do not
bump it. `ttucs_version` in `/health` is the engine build, `model.version` the weights version -
they move independently.
