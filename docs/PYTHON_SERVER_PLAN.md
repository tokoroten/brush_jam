# Plan: single Python server (`apps/brushjam`)

Decision (2026-09-05, user): unify frontend, room server and inference into ONE Python process. The Node room server (`apps/server`) is kept in the repo as the reference implementation until parity is proven, then removed.

## 0. Non-negotiables

- **The browser client does not change.** `apps/web` keeps talking the exact same WebSocket/HTTP protocol (`packages/shared/src/protocol.ts` is the contract). Any deviation is a bug in the port.
- **One process, one port**: `uv run brushjam` (or `python -m brushjam`) serves the built client, REST, WebSocket and inference on `:8787`.
- **One model resident on the GPU** (8 GB rule); inference runs in the same process as today's `apps/stream-worker` (its pipeline code is moved, not rewritten).
- Full-canvas AI mode only (the default since 7ddb0b8). Patch mode is NOT ported (documented as dropped; the Node code stays in git history).

## 1. Layout

```
apps/brushjam/                 uv project, Python 3.10 (same as the worker: torch/diffusers pin)
  pyproject.toml               console script `brushjam`
  src/brushjam/
    main.py                    uvicorn entry, CLI flags/env (PORT, HOST, AI_BACKEND, ...)
    app.py                     FastAPI app: static, REST, /ws, lifespan (backend load/unload)
    config.py                  env parsing + validation (mirror apps/server/src/config.ts semantics)
    protocol.py                pydantic models for every client/server message (mirror protocol.ts)
    validate.py                structural validation of inbound messages (mirror validate.ts)
    room.py                    RoomState + reducer (strokes, undone, per-user undo, layers, offsets, settings, revisions, aiGeneration, sessions/tokens, limits)
    runtime.py                 RoomRegistry/RoomRuntime: sockets, presence, broadcast, heartbeat, eviction, image store + quotas, capability refresh
    scheduler.py               full-mode AI scheduler (debounce, single in-flight, pending, promptEpoch, stale discard, watchdog, permanent vs transient errors, backoff floors)
    raster.py                  canvas rendering with Pillow: layers (order/visibility/opacity/offset), pen/noise/eraser strokes with per-stroke alpha, reference images, downsample/upsample to aiResolution; noise = same FNV/xorshift hash as packages/shared/src/noise.ts
    ai/backends/{base,mock,inproc,comfyui,runpod}.py   inproc = today's stream_worker pipeline (moved), with LoRA attach/detach for fast/quality profiles
    static/                    built client copied by `scripts/build_web.py` (runs `pnpm --filter @brushjam/web build`)
  tests/                       pytest; see §4
```

## 2. Behavioural parity (port these, in this order)

1. Protocol + validation (all message types, limits: MAX_POINTS, stroke caps, alpha 0.05–1, denoise grid, resolution grid, profile enum).
2. Room reducer: join/leave/resume token (64-session table rules), strokes (qualified ids, pending caps/expiry, cancel on layer delete/clear/offset change/leave), per-user undo, clear_layer dirty semantics, layers (max 8, kinds, includeInAI, offsets, locked transform refusal), settings (prompt, denoise, negative, resolution ceiling vs start, profile, promptChanged → re-run), snapshot fields incl. `aiGeneration`, `canvasSize`, `aiProfiles`, `maxDenoise`, `aiResolutionMax`, `negativeActive`, `aiResolutionAdjustable`.
3. Runtime: ws handshake `/ws/rooms/{id}?name&token`, snapshot on join, presence, cursor relay (no queueing), chunk relay, `stroke_committed`, `undo_applied`, `clear_applied`, `layer_*`, `prompt_changed`, `ai_settings_changed`, `ai_capabilities`, `ai_status`, `ai_result {rect,url,aiRevision,crop,apply,latencyMs,profile}`, `error`, `stroke_cancel`; ping/pong heartbeat (30 s, 2 misses); maxPayload 1 MiB; slow client cutoff 8 MiB; room cap 64; idle eviction 30 min; image upload (`POST /rooms/{id}/images`, header preflight + decode verify, 32 images / 64 MiB per room, decoded LRU budget), `GET /rooms/{id}/images/{imageId}`, `GET /rooms/{id}/ai.png`, `GET /rooms/{id}/patches/{n}.png` (retain 2 in full mode), `POST /api/rooms`, `GET /healthz`.
4. Scheduler (full mode) + raster + backends. Inference runs in a worker thread with the existing GPU lock; the event loop never blocks. Render snapshot captured synchronously before any await.
5. Static serving of the built client with SPA fallback for `/r/{id}`.

## 3. Rendering rules (Pillow)

- Canvas is `canvasSize`² RGBA. Each draw layer is rendered to its own RGBA image at alpha 1, then composited with layer opacity. A stroke = polyline with round joins/caps at its width (pressure-scaled per segment, like the TS renderer), rasterised into its own bbox-sized RGBA at full strength and composited ONCE with the stroke alpha (no self-overlap accumulation). Eraser = destination-out on the layer. Noise = per-pixel `noiseRGB(seed, worldX, worldY)` — port `fnv1a`/`noiseHash`/`noiseRGB` bit-exactly; verify with a fixture generated from the TS implementation.
- Antialiasing parity with the browser is NOT required (AI input only) but must be deterministic across runs. Use supersampling ×2 for lines if Pillow's antialiasing is insufficient.
- Reference layers with `includeInAI` are drawn with position/scale; downsample to `aiResolution` with LANCZOS, upsample results with BICUBIC.

## 4. Tests (pytest, GPU-free by default)

- Cross-language fixtures: a TS script `apps/server/scripts/export-fixtures.ts` writes `fixtures/protocol-samples.json` (one valid + invalid sample per message type) and `fixtures/noise-samples.json` (seed, x, y → rgb) and `fixtures/reducer-trace.json` (a scripted sequence of messages and the expected snapshot after each). The Python tests replay these; they are the parity proof.
- Port the intent of the Node suites: reducer/undo (Alice/Bob), layers/offsets, settings, session table, scheduler (fake backend + fake clock), validation, ws integration (httpx + websockets against the ASGI app: two clients, strokes, undo, presence, reconnect with token, heartbeat), image upload guards, raster determinism, backend selection/capabilities, config validation.
- `playtest-sim` equivalent: reuse the Node script (it is protocol-level) against the Python server — it must PASS unchanged.

## 5. Delivery

- `pnpm build` (web) → `python scripts/build_web.py` → `uv run brushjam` serves everything. README section "Single-process Python server". `apps/stream-worker` becomes a thin shim or is removed once `inproc` works (keep `/healthz` shape for tooling).
- Milestones: M1 protocol+reducer+ws with mock backend and static client (browser-verifiable, no GPU); M2 raster + scheduler + inproc backend (GPU window needed); M3 quality profile via LoRA detach; M4 Codex review + fixes + remove Node server.
- GPU: ask the coordinator for a window before any real model load (one model at a time).


---

## Status

| milestone | state |
| --- | --- |
| M1 - protocol, reducer, rooms, sockets, fixtures | done |
| M2 - the in-process pipeline as a backend | done |
| M3 - the HTTP backends and backend selection | done |
| M4 - review, fixes, retire the Node server | review done (6 Codex rounds, GO); retirement prepared, deletion pending approval |

M4 in detail:

- Six rounds of adversarial review, 20 findings, all fixed with tests. The
  table is in [`PYTHON_SERVER.md`](PYTHON_SERVER.md#review).
- The tooling that had to survive `apps/server` now lives in `tools/`
  (`@brushjam/tools`): `latency`, `playtest-sim`, `smoke`. They speak only the
  public HTTP and WebSocket surface.
- `export-fixtures` stays in `apps/server` and dies with it; the fixtures it
  produced are committed and keep being replayed.
- The deletion itself is one reviewable commit, listed in
  [`RETIRE_NODE_CHECKLIST.md`](RETIRE_NODE_CHECKLIST.md), pending approval.
