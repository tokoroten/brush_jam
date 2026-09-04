# Brush Jam — MVP vertical slice

Several people draw on one big shared canvas while an AI continuously
reinterprets what they make. This repository implements the MVP described in
[`docs/MVP_PLAN.md`](docs/MVP_PLAN.md) (sections 1–9), which in turn implements
phases 1–3 of [`docs/BRUSHJAM_CONTEXT.md`](docs/BRUSHJAM_CONTEXT.md). Nothing
marked Post-MVP in the context document is implemented.

## Quick start

```bash
pnpm install
pnpm dev          # server on :8787, web on :5173
```

Open <http://localhost:5173>, pick a name, press **Create room**, and send the
resulting `/r/<id>` URL to someone else (or open a second tab). Draw on the left;
AI patches appear on the right.

With ComfyUI running locally the ComfyUI backend is auto-detected. Without it the
server falls back to a GPU-free mock backend and logs:

```
[ai] backend: mock - ComfyUI was not reachable at http://127.0.0.1:8188
```

Other commands:

```bash
pnpm test         # 227 tests across shared / server / web
pnpm typecheck
pnpm build        # server bundle + web dist
pnpm start        # production: node apps/server/dist/index.js, serves apps/web/dist
pnpm --filter @brushjam/server smoke   # real end-to-end generation, saves smoke-out/patch.png
```

`pnpm build` then `pnpm start` serves the built client from the room server on
`:8787`, so no Vite proxy is needed in production.

## Layout

```
packages/shared/       protocol types + pure logic, no DOM and no Node imports
  src/geometry.ts        rects, stroke bboxes, rect subtraction
  src/dirty.ts           dirty-region merging
  src/crop.ts            AI window selection + central apply rect
  src/revision.ts        latest-useful-wins rule
  src/undo.ts            per-user undo selection
  src/mask.ts            mask geometry (dilate / clip / apply area)
  src/camera.ts          pan / zoom / fit math
  src/render.ts          renderStrokes() — runs on both server and browser
  src/protocol.ts        WebSocket message types

apps/server/           Node 22 + ws + node:http, in-memory rooms
  src/room.ts            authoritative room reducer (strokes, undo, layers, prompt)
  src/validate.ts        runtime validation of every client message
  src/imageInfo.ts       PNG/JPEG/WebP header probe and upload limits
  src/runtime.ts         room runtime: sockets, AI canvas, patch store
  src/server.ts          HTTP routes + WebSocket upgrade + static client
  src/raster.ts          @napi-rs/canvas: AI input, soft mask, AI canvas compositing
  src/ai/scheduler.ts    debounce, single in-flight, crop choice, stale handling
  src/ai/backends/       comfyui | mock | runpod
  scripts/smoke.ts       end-to-end smoke test against the configured backend

apps/web/              Vite + React 19
  src/App.tsx            name gate, home page, /r/<id> routing
  src/Room.tsx           tools, camera, pointer handling, paste, prompt
  src/StageView.tsx      one viewport (human or AI), shared camera
  src/LayerPanel.tsx     layer list
  src/roomClient.ts      WebSocket client + all client-side rasters
  src/raster.ts          per-layer offscreen canvases
  src/paste.ts           clipboard downscale / placement helpers
  src/serialQueue.ts     keeps ordered websocket frames applied in order
  src/session.ts         per-room reconnect token
```

## How it works

- **World.** 4096×4096 logical canvas. One `Camera {centerX, centerY, zoom}`
  drives both the Human and AI views, so they can never drift apart.
- **Truth.** The room server owns an append-only stroke log plus a set of undone
  stroke ids. Rasters are derived, never authoritative — a client can reconnect
  and rebuild everything from the `snapshot` message.
- **Undo.** `undo` reverts the *sender's* latest not-yet-undone stroke
  (Alice #100, Bob #101, Alice #102 → Alice's undo removes #102). `clear_layer`
  is room-level and not undoable. No redo in MVP.
- **AI loop.** Every committed stroke / undo / clear adds its bbox to the room's
  dirty regions (merged when within 256 px). After `AI_DEBOUNCE_MS` of quiet the
  scheduler picks a square `AI_WINDOW` crop centred on the most recent region,
  renders the AI input server-side with the same `renderStrokes` the browser
  uses, builds a mask (dilate 48 px → feather 32 px → limited to the central
  `AI_APPLY` area), and calls the backend. Only one request is in flight per
  room; activity during a request queues exactly one more run. A result whose
  revision is older than the last accepted one is discarded. Accepted patches are
  composited into the room's persistent AI raster through the soft mask, and
  clients are sent the composited rect as a PNG URL.
- **Drawing never waits.** Strokes are drawn locally on pointer input and relayed
  as chunks (~40 ms); the AI is entirely out of that path.
- **Reference images.** Ctrl/Cmd+V uploads a downscaled (≤2048 px) PNG and adds a
  `reference` layer. Reference layers are visible but **excluded from AI input**
  until you tick "AI input" in the layer panel.

## Environment variables

Read from the process environment; a repo-root `.env` is loaded if present (its
values are never logged and it stays git-ignored).

Every value is validated at startup and the server refuses to boot with a clear
message if anything is out of range (`AI_APPLY` must not exceed `AI_WINDOW`, both
are multiples of 64, denoise is 0..1, and so on).

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind address; set `0.0.0.0` to expose on the LAN |
| `PORT` | `8787` | Room server port |
| `AI_BACKEND` | auto | `comfyui`, `mock`, `runpod`, or unset for auto-detect |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | Local ComfyUI |
| `COMFYUI_CHECKPOINT` | `waiNSFWIllustrious_v150.safetensors` | Must exist in ComfyUI |
| `AI_WINDOW` | `1024` | Square generation window in world px (8 GB VRAM friendly) |
| `AI_APPLY` | `768` | Central area the result is allowed to change |
| `AI_STEPS` | `14` | Sampler steps |
| `AI_DENOISE` | `0.55` | img2img strength |
| `AI_CFG` | `5.5` | CFG scale |
| `AI_DEBOUNCE_MS` | `400` | Quiet time before a generation starts |
| `AI_WATCHDOG_MS` | `180000` | A generation past this is abandoned, not left in flight |
| `ROOM_IDLE_MS` | `1800000` | Empty rooms are reclaimed after this long |
| `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY` | — | Only for `AI_BACKEND=runpod` |
| `WEB_DIST` | `apps/web/dist` | Static client directory |

## ComfyUI requirements

ComfyUI 0.28.0 with `waiNSFWIllustrious_v150.safetensors` and core nodes only.
The workflow is built in TypeScript (`apps/server/src/ai/backends/comfyui.ts`,
node ids fixed so tests can assert on it):

```
CheckpointLoaderSimple → 2× CLIPTextEncode
LoadImage(image) → VAEEncode ┐
LoadImage(mask) → ImageToMask ┴→ SetLatentNoiseMask → KSampler → VAEDecode → SaveImage
```

`SetLatentNoiseMask` (rather than `VAEEncodeForInpaint`) keeps the human drawing
as the img2img base, so the model reinterprets the strokes instead of filling
holes.

### Switching to RunPod later

Set `AI_BACKEND=runpod`, `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY`. The adapter
posts the **same workflow JSON** to `/v2/{id}/runsync` in the `worker-comfyui`
input format (`{ input: { workflow, images: [{name, image: base64}] } }`), so
moving to the cloud is a transport change, not a pipeline change. Nothing in this
repository deploys to RunPod, and the adapter has not been run against a live
endpoint — treat it as untested code.

## Measured behaviour

Smoke test on the local box (RTX 3070 8 GB, `AI_WINDOW=1024`, 14 steps,
denoise 0.55): first generation ~73 s including checkpoint load, warm generations
~26–31 s. That is slower than the plan's 8–15 s estimate; the machine had ~3 GB
free RAM during the run, so weight paging is the likely cause. The pipeline is
correct end to end: stroke → dirty region → crop → mask → ComfyUI → composited
patch broadcast to clients.

**Honest caveat on output quality.** With sparse line art on a white background,
the reinterpretation is subtle — at denoise 0.55 the model mostly smooths and
recolours the strokes rather than turning them into a scene. Raising
`AI_DENOISE` to ~0.85 changes colours and line weight but still reads as line
art, because the input latent is dominated by white. Making the AI feel like a
real collaborator is a tuning problem for the playtest phase (denoise, prompt,
maybe a coloured/filled base), not an architectural one — the plumbing described
above is what this slice was built to prove.

## Decisions and deviations from the plan

Judgement calls made while implementing, since the plan left them open:

1. **Rooms are created on demand.** `POST /api/rooms` mints an id, but joining
   `/ws/rooms/<any valid id>` also creates the room, so a shared URL always works
   after a server restart.
2. **`chooseCrop` returns an exact integer square.** The plan's clamp-then-round
   could produce a 1024×1025 rect (observed in the first smoke run), which broke
   the single-scale assumption in the mask builder.
3. **Dirty regions are rect-subtracted, not cleared.** After an accepted result
   the repainted apply area is subtracted from each region that was dirty *when
   the request was built*; regions dirtied *during* the request survive whole, so
   the queued follow-up run has something to do. This avoids both an infinite
   regeneration loop and losing edits made mid-generation.
4. **AI patches are served over HTTP, not pushed over the socket.** `ai_result`
   carries `{rect, url}`; the last 24 patches are kept in memory per room, and a
   late joiner pulls `GET /rooms/:id/ai.png` once.
5. **Prompt changes re-run the existing dirty regions** (`scheduler.nudge()`)
   rather than marking the whole canvas dirty. A prompt change with a clean
   canvas does nothing until someone draws.
6. **Layer reorder sends the full ordered id list**, which is simpler to validate
   than relative moves.
7. **Colours must be 6-digit hex.** The server rejects anything else and falls
   back to black; the UI uses `<input type="color">`, which always sends 6 digits.
8. **Shift+drag also pans** (in addition to space+drag and middle-drag), because
   space+drag is awkward while a pointer is captured.
9. **`packages/shared` is consumed as TypeScript source** (no build step) via a
   Vite alias and Node's TS loader, which keeps one source of truth for
   `renderStrokes` on both sides.
10. **The reference-layer transform is a drag plus a scale slider**, per the
    plan — no rotation and no transform handles.
11. **Stroke ids avoid `crypto.randomUUID`.** That API only exists in secure
    contexts, and browser testing over a plain-http LAN hostname threw on it, so
    ids come from `crypto.getRandomValues` with a `Math.random` fallback.
12. **Stroke ids are namespaced by author** (`<userId>:<clientId>`). Two clients
    choosing the same id would otherwise share an undo entry.
13. **The apply rect is centred on the dirty region, not on the crop.** With a
    fixed centred rect, a stroke against a canvas edge was never inside the
    repainted area, so the scheduler regenerated the same crop forever. A
    per-region no-progress guard is the backstop.
14. **Reconnects carry a `sessionStorage` token** so a dropped connection keeps
    the same server identity and undo stack. It is not authentication: anyone
    holding the token is that participant, which is the right trade for a
    local playtest.
15. **The server binds to `127.0.0.1` by default.** Exposing the room server
    needs an explicit `HOST=0.0.0.0`, since there is no authentication.

## Not verified

- The RunPod adapter (no live endpoint was used, by instruction).
- A real multi-person playtest. Two browser tabs in one room were verified
  end to end (a stroke drawn in tab A appeared in tab B, presence showed both
  members, the shared prompt propagated, and an AI patch was composited into the
  right-hand view with the crop/apply overlays aligned across both views). Three
  people on three machines has not been tried.
- Touch/pen pressure on real tablet hardware (Pointer Events are wired up, but
  only a mouse was used).
