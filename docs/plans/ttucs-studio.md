# Plan: ttucs Studio (desktop UI) and the REAPER extension

Status: draft for implementation. Owner: Jon Evans. Written 2026-11-10.

This document is a handover brief for an agent building two consumers of `ttucs`:

1. **ttucs Studio** - a desktop app for reviewing and exporting UCS metadata for a folder of
   recordings.
2. **ttucs REAPER extension** - a C++ REAPER extension that tags media items in a project.

Both talk to the same engine over the same contract. Phase 0 lands that contract in this repo; the
two consumers live in their own repos.

---

## 1. Decisions already made (do not re-litigate without the owner)

| Decision | Choice | Why |
|---|---|---|
| Inference engine | The existing Python `ttucs` package, unchanged | The `bow` model is PyTorch; reimplementing it is not on the table |
| Engine surface | `ttucs serve` - JSON over loopback HTTP | Reachable from TypeScript *and* from C++ with no bindings, one loaded model shared by every consumer |
| Desktop shell | Tauri 2 + React + TypeScript + Vite | ~10-20 MB installer, native drag-drop and file dialogs, real web UI quality. **Confirmed by the owner on 2026-11-10; not open for substitution.** If a Tauri limitation blocks a phase, raise it with the owner rather than switching shells |
| Speech-to-text | `whisper.cpp` (native binary, C API) | C/C++ so the REAPER extension can use the same binary; no Python STT dependency |
| Python packaging | PyInstaller/Nuitka **onedir** sidecar, not onefile | Torch unpacks slowly from onefile; onedir starts in ~1 s |
| Repos | `ttucs` (this repo) + `ttucs-studio` + `ttucs-reaper` | Keeps the core install small; the REAPER SDK must never be vendored here |
| Offline | **User data and model input never leave the machine.** Telemetry and auto-update are opt-in and carry no user content | AGENTS.md rule 4 binds the engine (no network at import or prediction). The app shell may talk to a telemetry endpoint and an update feed, but never with transcripts, filenames, paths or audio |

Non-goals for v1: cloud sync, multi-user, model training from the UI, Windows-only shortcuts that
break on macOS, any third-party sound-library metadata format beyond a plain CSV.

---

## 2. Architecture

```
                    ┌──────────────────────────────┐
                    │  ttucs serve  (Python)       │
   ttucs Studio ───▶│  127.0.0.1:<ephemeral>       │
   (Tauri/React)    │  token in $TTUCS_HOME/       │
                    │  serve.json                  │
   ttucs-reaper ───▶│  one TTUCS instance,         │
   (C++ DLL)        │  one lock around predict     │
                    └──────────────┬───────────────┘
                                   │ TTUCS.from_pretrained()
                                   ▼
                    ttucs-model-bow (installed wheel, fp16)
```

- The server is the **only** place a model is loaded. Loading costs ~1-2 s and a few hundred MB;
  doing it once and sharing it is what makes per-item tagging in REAPER interactive.
- `ttucs predict --json` stays supported for scripting and one-shot use, but is not the integration
  path (2-4 s process start per call).
- Both consumers must be able to **start the server themselves** if it is not running (see 3.4).

---

## 3. Phase 0 - the contract, in this repo

Ship as one PR to `main` (owner merges). Branch `feat/serve-api`.

### 3.1 New CLI subcommand

```
ttucs serve [-m MODEL] [--host 127.0.0.1] [--port 0] [--token-file PATH]
            [--no-token] [--top-k 5] [--warm/--no-warm] [--log-level info]
```

- `--port 0` picks a free port; the bound port is written to the token file.
- `--host` must default to `127.0.0.1` and refuse a non-loopback address unless `--allow-remote`
  is passed explicitly. Do not add `--allow-remote` in v1.
- Implement with `http.server.ThreadingHTTPServer` from the standard library. **No new core
  dependency**; a `[serve]` extra is not needed and would violate the small-install goal.
- Import `ttucs.pipeline` lazily inside the command, matching the existing CLI style.
- Warm up with one `predict("test tone")` at startup so the first real request is fast; report
  `warm` in `/health`.

### 3.2 Endpoints

Every response is an envelope with `schema_version` (integer, starts at `1`). Consumers must refuse
an unknown major version and say so in the UI rather than guessing.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | - | engine + model identity, see below |
| POST | `/predict` | `{texts: [str], top_k?: int, creator_id?: str, source_id?: str}` | `{predictions: [Prediction]}` |
| POST | `/parse` | `{texts: [str]}` | `{slates: [SlateFields]}` |
| GET | `/taxonomy/search?q=&limit=` | - | `{entries: [{cat_id, category, subcategory, cat_short, explanation}]}` |
| GET | `/taxonomy/{cat_id}` | - | `{entry: {...}}` |
| POST | `/shutdown` | - | `{ok: true}` (token required) |

`/health` returns:

```json
{
  "schema_version": 1,
  "status": "ok",
  "warm": true,
  "ttucs_version": "0.3.0",
  "ucs_version": "8.2.1",
  "model": {"name": "bow", "version": "0.2.0", "source": "package",
            "manifest_verified": true, "path": "..."},
  "offline": true,
  "device": "cpu",
  "unknown_threshold": 0.12,
  "category_threshold": 0.5
}
```

`/predict` **always returns an array**, even for one input. This deliberately differs from
`ttucs predict --json`, which returns a bare object for a single text and an array for many - a wart
that has already bitten scripting. Fix that in the same PR: make `cmd_predict` always emit a list,
and note it in `CHANGELOG.md` under **Changed** as a breaking change for anyone parsing single-input
output.

`Prediction` is exactly `Prediction.to_dict()` from `src/ttucs/pipeline.py:76`:

```json
{
  "text": "...", "description": "...",
  "cat_id": "DOORCreak", "category": "DOOR", "subcategory": "Creak",
  "cat_short": "Creak", "category_full": "DOOR-Creak",
  "confidence": 0.71, "category_confidence": 0.88,
  "decision": "confident",
  "alternatives": [{"cat_id": "...", "category": "...", "subcategory": "...",
                    "cat_short": "...", "confidence": 0.71}],
  "categories": [{"category": "DOOR", "confidence": 0.88}],
  "slate": {"raw_text": "...", "description": "...", "project": null, "location": "...",
            "microphones": [], "mic_types": [], "mic_setup": [], "recorder": null,
            "sample_rate": 96000, "bit_depth": "24", "take": "3", "scene": null,
            "slate": null, "perspective": [], "date": null, "creator": null,
            "notes": null, "unparsed": []},
  "fx_name": "Wooden Door Creak Slow",
  "keywords": ["door", "creak"],
  "filename": "DOORCreak_Wooden Door Creak Slow_NONE_ThickAsThieves_MKH416",
  "filename_parts": {"cat_id": "...", "fx_name": "...", "creator_id": "...",
                     "source_id": "...", "user_category": null, "user_data": null},
  "is_unknown": false
}
```

`slate` keys are the `SlateFields` dataclass fields (`src/ttucs/slate/parser.py:34`). Treat unknown
keys as forward-compatible additions; never fail on them.

### 3.3 Security and lifecycle

- Bind loopback only. Require `X-TTucs-Token` on every route except `/health`; compare with
  `hmac.compare_digest`. `--no-token` exists for tests and must log a warning.
- Write the endpoint file to `$TTUCS_HOME/serve.json` (fall back to the platform cache dir, same
  resolution as `hub.cache_root()`), mode `0600`:

  ```json
  {"schema_version": 1, "host": "127.0.0.1", "port": 8731, "token": "...",
   "pid": 12345, "started": "2026-11-10T12:00:00Z", "ttucs_version": "0.3.0"}
  ```

  Delete it on clean shutdown; a stale file whose `pid` is dead is treated as "not running".
- No CORS headers. A browser page must not be able to reach the engine even with the token.
- One `threading.Lock` around `predict_batch`; the model is not safe to call concurrently.
  `ThreadingHTTPServer` handles accept concurrency, the lock serialises inference.
- Cap request size (e.g. 1 MB) and `len(texts)` (e.g. 512) so a runaway client cannot OOM the app.
- Log to stderr only; never log tokens or full transcripts at info level.

### 3.4 Launcher contract (used by both consumers)

1. Read `$TTUCS_HOME/serve.json`. If present and `GET /health` succeeds with the token, use it.
2. Otherwise spawn `ttucs serve --port 0 --token-file <path>` detached, poll the file for up to 30 s,
   then `GET /health` until `warm` is true (60 s budget on first run).
3. On app exit, call `/shutdown` only if this process started the server.

Document this in `docs/serve-api.md` so the C++ side implements the same steps.

### 3.5 Tests and acceptance for Phase 0

- `tests/test_serve.py`: start the server on port 0 in a fixture, assert `/health` shape, token
  rejection (401), `/predict` array shape for one and many texts, `/parse` without a model,
  taxonomy search, request-size cap, `/shutdown`.
- `tests/test_cli.py`: `predict --json` now always emits a list.
- `ruff check src tests scripts` and `pytest -q` green; CI green on 3.10-3.12.
- Manual: `ttucs serve` then `curl` `/health` and `/predict` on Windows PowerShell, with
  `TTUCS_OFFLINE=1` and an empty `TTUCS_HOME`.
- `CHANGELOG.md` **Added** entry for `ttucs serve`, **Changed** for the `--json` shape, and a
  `docs/serve-api.md` page. No weights change, so no `ttucs-model-bow` bump.

---

## 4. Phase 1-4 - ttucs Studio

Separate repo `ttucs-studio`. Layout:

```
ttucs-studio/
  src-tauri/            Rust shell: window, drag-drop, file dialogs, sidecar spawn, asset protocol
  src/                  React + TS
    app/                routing, layout, theme
    features/queue/     file list, transcription progress, sorting, filtering
    features/inspector/ slate fields, alternatives, filename editor, keywords
    features/export/    CSV / JSONL / rename
    features/settings/  model, thresholds, defaults, STT model
    lib/engine.ts       typed client for the Phase 0 contract (single source of truth)
    lib/schema.ts       zod schemas mirroring the contract, validated on every response
  sidecar/              PyInstaller spec + whisper.cpp binary fetch script
```

### 4.1 Screens

1. **Queue** (default). Drop audio files or paste transcripts. Columns: file, duration, transcript
   (truncated), CatID, confidence, decision badge, suggested filename. Sort by confidence ascending
   by default - the review order is "worst first". Filter chips: all / confident / fallback /
   unknown / overridden.
2. **Inspector** (right pane, always visible). Transcript with parsed slate spans highlighted;
   CatID with top-5 alternatives as clickable bars; category confidence; decision badge with a
   one-line explanation of what `category_fallback` and `unknown` mean; editable UCS filename with
   live validation against `UCSFilename` rules; keyword chips; copy filename / copy JSON.
3. **Override**. CatID search (`/taxonomy/search`), keyboard-first, `/` focuses it. Overriding
   marks the row `overridden` and keeps the model's original guess visible for later analysis.
4. **Export**. CSV (columns: file, cat_id, category, subcategory, confidence, decision, filename,
   keywords, plus the slate fields), JSONL (the raw `Prediction` objects), and **rename on disk**.
5. **Settings**. Model path or name, `unknown_threshold`, `category_threshold`, default creator and
   source IDs, STT model size, server port override. The engine's offline mode is displayed as
   locked on; telemetry and update checks are separate switches with their own state.
6. **Transcript editor**. Fix STT errors and re-predict that row instantly. This is the highest-value
   accuracy lever in the whole app: real slates are mis-transcribed far more often than they are
   mis-classified.

### 4.2 Workflow

Drop folder -> transcribe (whisper.cpp, per-file progress, cancellable) -> predict in batches of 32
-> review worst-first -> accept or override -> export.

Keyboard: `j`/`k` next/prev, `Enter` accept, `e` edit transcript, `c` copy filename, `o` override,
`Space` play the file, `Cmd/Ctrl+E` export. Every action reachable by mouse too.

### 4.3 Rename safety (the only destructive feature)

- Dry run by default: show a two-column diff of old -> new names, with collisions and invalid names
  flagged before anything moves.
- Never overwrite an existing file; skip and report.
- Write `ttucs-undo-<timestamp>.json` next to the folder and offer a one-click undo.
- Refuse to run on a folder containing subfolders unless the user opts in.

### 4.4 Packaging

- Tauri bundle per platform; sidecar is a PyInstaller onedir tree in `resources/`.
- Bundle the `ttucs-model-bow` wheel contents so the app works with no network and no separate pip
  install; set `TTUCS_OFFLINE=1` in the sidecar environment.
- Installer size expectation: 250-400 MB because of torch. State this in the README rather than
  pretending otherwise. If that is unacceptable, the alternative is exporting the model to ONNX and
  running it in Rust - a separate project, not a v1 task.

### 4.5 Acceptance per phase

- **P1**: app opens, `/health` shown in an About panel, single-text predict round-trips. Proves the
  pipe end to end.
- **P2**: 200 files dropped, transcribed, predicted, reviewed, exported to CSV; **no user content
  leaves the machine** (verify with a proxy log or a firewall rule: only the telemetry and update
  hosts may appear, and only when those switches are on).
- **P3**: transcript edit re-predicts in under 200 ms; STT cancel leaves no orphan processes.
- **P4**: rename dry run on a copy of a real folder, undo restores every name byte-for-byte.

---

## 5. Phase 5 - ttucs REAPER extension

Separate repo `ttucs-reaper`, C++17, built against the REAPER extension SDK fetched at build time.
**Never vendor SDK headers into the `ttucs` repo.**

### 5.1 What it does

- Menu under `Extensions > ttucs`.
- `Tag selected items`: for each selected media item, get a transcript from (in order) the item
  notes, an adjacent `.txt`, or whisper.cpp on the source file; POST to `/predict`; write back
  `P_NOTES` with the UCS filename plus CatID, category and confidence, and optionally set the take
  name.
- `Tag all items in project`: same, batched, with a progress dialog and cancel.
- `Apply render pattern`: set the project render filename pattern from the UCS fields.
- `Open ttucs Studio`: launch the app.

### 5.2 Gotchas that will otherwise cost a day each

- **Never call HTTP on the main thread.** REAPER's UI thread must not block. Run requests on a
  worker thread and marshal results back with `defer` / `plugin_register("timer", ...)`.
- **Start the server from the extension** using the launcher contract in 3.4, including the 30 s
  poll. Show a modal "starting engine" state; do not silently hang.
- Use `WinHTTP` on Windows and `libcurl` elsewhere, or `libcurl` everywhere for one code path.
- `GetSetMediaItemInfo_String(item, "P_NOTES", ...)` for notes; `GetSetMediaItemTakeInfo_String`
  for take names. Read `P_NOTES` first and preserve any existing text.
- Build per platform (win x64, macOS universal, linux x86_64); ship `.dll`/`.dylib`/`.so` into
  `UserPlugins`. There is no installer framework in REAPER - document the copy step.
- Undo: wrap writes in `Undo_BeginBlock` / `Undo_EndBlock` so a batch tag is one undo step.
- The extension must degrade gracefully when the engine is missing: show the error and a link to
  the install instructions, never crash the host.

### 5.3 Acceptance

- Tag 20 selected items in a scratch project; verify notes, undo in one step, and that REAPER stays
  responsive (no frame stalls over ~50 ms) during the batch.
- Engine absent: menu action reports a clear error and REAPER keeps running.

---

## 6. Constraints inherited from AGENTS.md

- Work on a feature branch, open a PR, **never merge or push to `main`**.
- No third-party sound-library metadata anywhere, including test fixtures and screenshots.
- Nothing may touch the network at import time or during prediction; `TTUCS_OFFLINE=1` must
  guarantee zero network access. That binds the engine and the sidecar, not the app shell: the
  shell may send telemetry and check for updates under the rules in section 7, and must never
  include user content in either.
- Only the parsed description is classified. The UI may show slate fields but must not send them
  into the classifier.
- `examples/realistic_slates.jsonl` stays held out. Do not use it to tune the UI's thresholds or to
  demo accuracy numbers.
- Changelog and SemVer for every user-visible change in this repo. The two consumer repos keep their
  own changelogs.
- Report honestly: no "works" or "verified" without a run behind it.

---

## 7. Telemetry and auto-update

Both are wanted: telemetry to improve the tool, auto-update so users never chase an installer. Both
are allowed to use the network. Neither may carry user content.

### 7.1 The line

| Stays on the machine, always | May leave the machine |
|---|---|
| audio files and their paths | app version, OS and arch, locale |
| transcripts and slate text | model name/version, `schema_version` |
| CatIDs, filenames, keywords, exports | counts and timings (files processed, ms per predict) |
| project, location, creator/source IDs | decision mix (`confident` / `fallback` / `unknown`) |
| the user's overrides and corrections | crash type and stack, with paths and text stripped |

Rule of thumb: **aggregate numbers and identifiers of the build may leave; anything a recordist
typed or recorded may not.** A transcript is the most sensitive field in the app - it carries
project names, locations and client names.

### 7.2 Telemetry

- **Opt-in on first run**, one clear screen, default off until the user chooses. A single switch in
  Settings, revocable at any time, with a "what is sent" panel showing the exact payload schema.
- Anonymous install ID: a random UUID stored in the app config, resettable from Settings. No
  hardware fingerprint, no account, no email.
- Events are a fixed, versioned allowlist (`telemetry_schema_version`). Adding a field is a schema
  bump and a changelog entry. No free-form strings, no exception messages that could embed a path
  or a transcript.
- Batch and send on a timer, not per action; drop the batch if the send fails. Never block the UI
  and never retry into a loop.
- Respect `DO_NOT_TRACK=1` and a `TTUCS_STUDIO_TELEMETRY=0` environment override.
- Document the endpoint and the retention period in the README, and give the user a "delete my
  data" mailto or endpoint.
- The sidecar never sends anything: telemetry lives in the shell, so `TTUCS_OFFLINE=1` still means
  zero network from the engine.

### 7.3 Auto-update

- Tauri's updater plugin against a signed manifest; **signature verification is mandatory** and a
  failed check aborts the update rather than installing.
- Check on launch and every 24 h, on a background thread. A failed check is silent and logged
  locally; it must never block startup or show a modal.
- Notify, do not force: a non-blocking banner with "Install and restart" / "Later". No silent
  install while a batch is running, and never mid-rename.
- Channels: `stable` by default, `beta` opt-in in Settings. The manifest points at GitHub Releases
  for the app repo, separate from the `ttucs` and `ttucs-model-bow` releases.
- Weights update independently of the app: a new `ttucs-model-bow` is a sidecar payload change, so
  the app version and the weights version are shown separately in About.
- Offline users must be unaffected: with no network the check fails quietly and the app runs as
  installed. Never gate a feature on a successful update check.

### 7.4 Acceptance

- With telemetry off and no network, the app is fully functional and makes zero outbound
  connections (verify with a proxy log).
- With telemetry on, capture one batch and assert against the allowlist that no field contains a
  transcript, filename, path or CatID from the test corpus.
- Auto-update: serve a manifest with a bad signature and confirm the update is refused with a
  clear message and the app keeps running.

## 8. Risks and open questions

| Risk | Mitigation |
|---|---|
| Torch sidecar makes the installer 300 MB+ | Accept for v1 and say so; ONNX export is the later lever |
| Tauri's webview differs across platforms (WebView2 / WKWebView / WebKitGTK) | Test on Windows and macOS before P2 sign-off; keep CSS conservative |
| STT quality dominates end-to-end accuracy | Make the transcript editable and show it prominently; never hide it behind a disclosure |
| REAPER SDK licensing for redistribution | Fetch at build time, do not vendor; the extension repo carries its own license note |
| Two repos drift from the contract | `schema_version` plus zod validation in TS and a compile-time struct in C++; a contract test in Phase 0 that both consumers run against a recorded fixture |
| Telemetry accidentally captures user content | Fixed allowlist schema, no free-form strings, a test that asserts the payload against a corpus of transcripts and paths, and a visible "what is sent" panel |
| Auto-update ships a broken build | Signed manifest, `beta` channel first, and a rollback path: keep the previous sidecar and app bundle so a bad release can be reverted without a reinstall |

Open for the owner: the app name (`ttucs Studio` is a placeholder), and whether the rename feature
ships in v1 or waits for v1.1.
