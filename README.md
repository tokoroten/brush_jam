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
pnpm test         # 440 tests across shared / server / web
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
| `AI_BACKEND` | auto | `stream`, `comfyui`, `mock`, `runpod`, or unset for auto-detect |
| `STREAM_URL` | `http://127.0.0.1:8790` | Model-resident worker (`apps/stream-worker`) |
| `STREAM_TIMEOUT_MS` | `120000` | Per-generation deadline for the stream worker |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | Local ComfyUI |
| `COMFYUI_CHECKPOINT` | `waiNSFWIllustrious_v150.safetensors` | Must exist in ComfyUI |
| `CANVAS_SIZE` | `1024` | World canvas size in px (square, multiple of 64, 512-4096) |
| `AI_MODE` | `full` | `full` regenerates the whole canvas; `patch` uses crops + dirty regions |
| `AI_PROFILE` | `fast` | Starting profile for new rooms: `fast` or `quality` |
| `AI_WINDOW` | the profile's size (`fast` 768, `quality` 1024), capped by the canvas | Generation size in px (512-2048, multiple of 64) |
| `AI_APPLY` | `768` | Central area the result is allowed to change |
| `AI_STEPS` | `14` | Sampler steps for the quality profile |
| `AI_FAST_STEPS` | `4` | Sampler steps for the fast profile |
| `AI_FAST` | - | Legacy alias: `1` means `AI_PROFILE=fast`, `0` means `quality` |
| `AI_STREAM_AUTO` | `0` | `1` lets `AI_BACKEND=auto` consider the stream worker |
| `COMFYUI_FAST_LORA` | `lcm-lora-sdxl.safetensors` | LoRA for the fast profile; empty disables it |
| `AI_DENOISE` | `0.7` | img2img strength; the starting value of each room's slider |
| `AI_CFG` | `5.5` | CFG scale |
| `AI_VAE_TILE` | `512` | VAEDecodeTiled tile size; `0` uses a plain `VAEDecode` |
| `AI_DEBOUNCE_MS` | `400` | Quiet time before a generation starts |
| `AI_WATCHDOG_MS` | `180000` | A generation past this is abandoned, not left in flight |
| `ROOM_IDLE_MS` | `1800000` | Empty rooms are reclaimed after this long |
| `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY` | — | Only for `AI_BACKEND=runpod` |
| `WEB_DIST` | `apps/web/dist` | Static client directory |

Fixed limits, not configurable: 64 live rooms per server (further creates and
upgrades get `429`/close 1013), 64 session tokens per room (the oldest
*disconnected* session is evicted first), 4 pending strokes per user, 20 000
committed strokes per room, 32 uploaded images per room, a 512 MiB budget for
decoded reference pixels shared process-wide (LRU eviction), a 60 s idle window
on a pending stroke, and a 2 min grace period before an unreferenced upload is
swept.

### Backend selection

An explicit `AI_BACKEND` always wins. `auto` (the default) probes ComfyUI's
`/system_stats` and otherwise falls back to the mock. Each probe has a 2 s
deadline so a dead endpoint cannot delay startup, and the chosen backend is
logged with the reason.

**The stream worker is explicit-only.** `auto` ignores it unless
`AI_STREAM_AUTO=1`, because its answering `/healthz` only means it is holding
~5 GB of VRAM - which on this 8 GB card starves ComfyUI - and because it needs
a different denoise to look right, so silently routing to it would change output
quality as well as speed. Which model owns the GPU is a deployment decision, not
something a reachability probe should infer.

When it is opted in, a reachable worker still has to be *usable*: auto requires
`warm: true` (a worker with no model loaded answers `ok` and then echoes the
input back) and `max_size >= AI_WINDOW` (a worker capped below the window
answers `ok` and then 400s every request, which full mode retries forever). Each
rejection is logged with its reason. An explicit `AI_BACKEND=stream` is always
honoured, but the same checks run and print a warning.

## ComfyUI requirements

ComfyUI 0.28.0 with `waiNSFWIllustrious_v150.safetensors` and core nodes only.
The workflow is built in TypeScript (`apps/server/src/ai/backends/comfyui.ts`,
node ids fixed so tests can assert on it):

```
CheckpointLoaderSimple → 2× CLIPTextEncode
LoadImage(image) → VAEEncode ┐
LoadImage(mask) → ImageToMask ┴→ SetLatentNoiseMask → KSampler → VAEDecodeTiled → SaveImage
```

### Fast and quality profiles

Each room chooses one, from the segmented control next to the prompt, and the
choice is shared like the prompt. It is a property of the request, not of the
process: one server serves rooms that disagree.

| profile | workflow | steps | cfg | default size | measured |
| --- | --- | --- | --- | --- | --- |
| `fast` | LCM LoRA, `lcm` / `sgm_uniform` | 4 | 1.5 | 768 | ~3.7 s |
| `quality` | plain checkpoint, `euler_ancestral` / `normal` | 14 | 5.5 | 1024 | ~10.3 s |

In the fast profile a `LoraLoader` (node 12) is inserted between the checkpoint
and its consumers - both `CLIPTextEncode` nodes and the `KSampler` read
MODEL/CLIP from it, while the VAE still comes from the checkpoint. Steps are
passed straight through: ComfyUI runs exactly that many sampler steps at any
denoise (it builds the longer schedule then keeps the last `steps + 1` sigmas,
so denoise only picks the starting noise level). cfg 5.5 burns the image out at
4 steps, and the sampler settings come from a named profile keyed off the LoRA,
because few-step LoRAs are not interchangeable: a name containing `dmd2` uses
cfg 1.0 (DMD2 is distilled and wants no guidance), anything else LCM at 1.5.

Switching profile also moves the room's generation size to that profile's
default, because the two go together - fast is only worth having if it is also
smaller. Everything stays adjustable in the Advanced panel afterwards.

`COMFYUI_FAST_LORA=''` disables the fast profile entirely: every room starts on
quality, rather than running a 4-step `euler_ancestral`, which is neither.

Switching profiles makes ComfyUI load or unload the LoRA, so the first
generation after a switch is a few seconds slower; the server logs it rather
than leaving it looking like a random stall.

### Backend capabilities

A backend declares what it can do (`capabilities()`), probed once at startup,
and rooms are capped by it: the UI disables a profile the backend does not have
and stops the denoise slider at its ceiling, and the server refuses anything
outside it rather than silently substituting.

| backend | profiles | max resolution | max denoise |
| --- | --- | --- | --- |
| comfyui / runpod | fast + quality (quality only without a LoRA) | 2048 | 0.95 |
| stream | **fast only** | worker's `max_size`, else 1024 | worker's `max_denoise`, else 0.9 |
| mock | fast + quality | 2048 | 0.95 |

The stream worker holds one fused LCM LoRA, so it has no quality mode at all -
asking it for 14 steps would silently run 4. Its defaults also differ, and the
server applies them once the backend is known (`auto` only resolves at startup):
resolution 768 and denoise 0.8, because 0.7 barely moves the drawing at 4 LCM
steps. An explicit `AI_WINDOW` or `AI_DENOISE` still wins.

```
pnpm dev:stream    # same as pnpm dev with AI_BACKEND=stream (works on Windows)
```

**Why these numbers.** Measured on an RTX 3070 8 GB - see
[docs/experiments/2026-09-05-comfyui/REPORT.md](docs/experiments/2026-09-05-comfyui/REPORT.md)
for ComfyUI and
[docs/experiments/2026-09-05-stream/REPORT.md](docs/experiments/2026-09-05-stream/REPORT.md)
for the stream worker, which is faster again (768 in ~1.8 s, 1024 in ~3.5 s).
Denoise 0.5 is a no-op on this checkpoint, 0.65 decorates, 0.8 genuinely
reinterprets (a noise-pen sky becomes buildings) and 0.9 discards the drawing,
which is why the default is 0.7. LCM matches the 14-step result up to about 0.65
but is visibly weaker at 0.8, so `fast` is the responsive default and `quality`
is there for when the model should actually invent something.

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

### Two AI modes

`AI_MODE=full` (the default, and what the playtest uses) treats the whole canvas
as one unit: any change - a stroke, an undo, a layer edit, a prompt or settings
change - marks the room changed, and after the debounce the entire canvas is
rendered, sent with a fully opaque mask, and the result **replaces** the AI
raster. There is no crop selection, no dirty-region bookkeeping and no mask
feathering, so a thin line no longer comes back as a narrow repainted band. The
window is locked to `CANVAS_SIZE`, which must therefore be <= 2048; the server
refuses to boot otherwise and points at patch mode.

**Generation resolution is not the canvas size.** `AI_WINDOW` is what the model
actually runs at: the whole canvas is rendered, resampled down to
`AI_WINDOW`x`AI_WINDOW`, generated, then resampled back up to the canvas and
composited. On an 8 GB card `CANVAS_SIZE=1024 AI_WINDOW=768` (or 512) is much
faster than generating at 1024, at the cost of detail; `AI_WINDOW` may also be
*larger* than the canvas. The effective sizes are logged at startup, and each
room can pick its own value from the advanced panel ("AI resolution", one of
512/768/1024, capped by the server's `AI_WINDOW`), which re-runs like a prompt
change.

`AI_MODE=patch` restores the original large-canvas pipeline (dirty regions ->
crop -> dilated, feathered mask -> apply rect), e.g.
`AI_MODE=patch CANVAS_SIZE=4096 AI_WINDOW=1024 AI_APPLY=768`.

The client never hard-codes a canvas size: it takes `canvasSize` from the
snapshot, sizes its rasters to it and fits the camera to it. In full mode the
crop/apply overlay rectangles are hidden, because they are the whole canvas.

### The noise pen

A third stroke tool next to pen and eraser. It fills the stroke shape with
deterministic RGB noise: the value of a pixel is `hash(seed, worldX, worldY)`
with `seed = FNV-1a(stroke.id)`, so two renders are byte-identical and a
server-side crop (rendered with the crop origin subtracted) produces the same
pixels as the client's full-size layer. The shape's antialiased alpha is kept,
so it composites like any other stroke, and it is a normal stroke everywhere
else: same undo, same eraser interaction, same dirty-region behaviour. It exists
to give the model something richer than white paper to reinterpret.

While a noise stroke is being drawn (yours or someone else's) it is previewed
into a per-stroke raster that is extended with each new segment, so the cost per
frame follows the movement rather than the whole stroke.

Implementation note: `renderStrokes` takes a `createCanvas(w, h)` dependency
(browser: `document.createElement('canvas')`, server: `@napi-rs/canvas`) and
rasterises the stroke into a temporary canvas of its bounding box before
replacing each covered pixel's RGB. Live preview while drawing uses the same
renderer per chunk - the per-frame cost is bounded by the stroke's bbox and was
not noticeable in the browser.

### Moving layers

The move tool moves reference layers *and* draw layers. A draw layer gets
`offsetX`/`offsetY`: the stroke log keeps its original coordinates and the whole
layer is translated at render time, on the client and on the server (including
the bbox maths that feed patch-mode dirty regions). Drawing on a moved layer
records points in layer space, so the line still appears under the pointer. The
hit test picks the topmost unlocked layer under the pointer - a reference by its
image box, a draw layer by the box of its strokes - and falls back to the
selected layer. **Moving a layer is not undoable**, exactly like moving a
reference image.

### Room-level AI settings

Behind the **advanced** toggle next to the prompt, and shared by everyone in the
room exactly like the prompt itself:

- **denoise** — 0.2 to 0.95 in 0.05 steps, starting from `AI_DENOISE`. Low values
  keep the drawing and only clean it up; high values reinterpret it.
- **AI resolution** — 512 / 768 / 1024, capped by the server's `AI_WINDOW`.
  Lower is faster and blurrier; the result is always scaled back to the canvas.
- **negative prompt** — up to 1000 characters. Empty means "use the built-in
  list", which the input shows as its placeholder.

Both debounce for 500 ms, broadcast as `ai_settings_changed`, ride along in the
snapshot, and behave like a prompt change: the last painted area is re-dirtied so
the effect is visible without anyone having to draw.

### Why the decode is tiled

On this box a 1024² KSampler pass finishes in ~22 s, but the plain `VAEDecode`
that follows took **1–4 minutes** whenever VRAM was contended (one run was still
decoding 2.5 minutes after sampling finished, and was cut off by the watchdog).
`VAEDecodeTiled` at `tile_size` 512 / `overlap` 64 decodes in slices instead, and
is the default. Set `AI_VAE_TILE=0` to go back to the plain node.

### Measuring latency

```
pnpm --filter @brushjam/server latency -- --url http://127.0.0.1:8787 --n 10
```

Joins a **running** server as an ordinary client, draws N short strokes one at a
time, and reports three numbers per edit, min/median/max:

- **stroke_end -> pixels on screen** - the whole wait, including fetching the
  result PNG, which is what a person actually experiences
- **stroke_end -> ai_result message** - the same minus that download
- **server pipeline** - the server's own `latencyMs`, which spans input render
  + backend + compositing (it is *not* backend-only, despite the name)

It starts nothing itself, so the `/healthz` line tells you which backend was
really measured.

### Denoise / quality grid

```
pnpm --filter @brushjam/server quality-grid                   # 4 drawings x 4 denoise at 768
AI_BACKEND=mock pnpm --filter @brushjam/server quality-grid   # instant, no GPU
```

Renders four synthetic drawings (line art, stick figure + blob, line art with a
noise sky, mostly noise) and sweeps denoise through whatever backend the config
selects, calling `generate()` directly - no server, no WebSocket, fixed seed,
full-white mask. Writes `docs/experiments/<date>/` with one PNG per cell, a
labelled contact sheet `grid.png` and `results.json` with per-cell latency, then
prints a summary table. Options: `--res`, `--out`, `--drawings`, `--denoise`.
See [docs/experiments/README.md](docs/experiments/README.md).

## Measured behaviour

Smoke test on the local box (RTX 3070 8 GB, `AI_WINDOW=1024`, 14 steps,
denoise 0.55): first generation ~73 s including checkpoint load, warm generations
~14–31 s (13.7 s on the most recent run with more free RAM). That is slower than the plan's 8–15 s estimate; the machine had ~3 GB
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

### Dev restart on Windows

`tsx watch` restarts used to fail with `EADDRINUSE 127.0.0.1:8787`: open
WebSocket connections kept the old process's listener alive. The server now
terminates every socket, calls `closeAllConnections()`, and exits on
SIGINT/SIGTERM/SIGHUP/SIGBREAK or an IPC `shutdown` message, with a hard 2 s
deadline; `listen` also retries EADDRINUSE a few times. Verified by editing a
server file twice under `tsx watch` with a WebSocket client connected - both
restarts served the new code.

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
15. **Uploads are structurally validated in pure JS before any decode.**
    `@napi-rs/canvas`'s `loadImage` *segfaults* (verified: exit 139, not a
    catchable exception) on a file with a valid PNG signature and IHDR but no
    image data, so a `try/catch` around the decoder cannot protect the process.
    `validateStructure()` walks PNG chunks / checks the JPEG SOS+EOI markers /
    checks the RIFF size before the single verifying decode, which also confirms
    the header's declared dimensions.
16. **A cancelled generation cleans up after itself.** Once `/prompt` has
    returned a `prompt_id`, any later failure interrupts the job if it is
    running or `POST /queue {delete:[id]}`s it if it is only queued, so an
    abandoned request does not keep occupying the GPU. A transient `/history`
    failure is retried while the generation deadline still holds.
17. **A prompt change during an in-flight run is not lost.** The scheduler keeps
    a prompt epoch; if it moved while a generation was running, the rect that
    result just repainted is re-dirtied so the new prompt gets applied there.
18. **The no-progress guard is scoped to one region.** Only the region that was
    actually selected can be judged stuck, and only after the same
    crop+region signature has already failed once — other overlapping regions
    are never discarded.
19. **Client asset loads are bounded and cancellable.** Reference images and
    `ai.png` load with a 10 s timeout outside the ordered message queue and are
    aborted on dispose/reconnect, so a stalled image can never wedge frame
    application; a failed AI patch schedules a revision-guarded refresh instead.
20. **Socket teardown is connection-scoped.** A superseded socket (React
    StrictMode's mount/unmount/mount, or a reconnect that beat the old close
    event) closes *after* its replacement joined, so `leave()` ignores a close
    from a socket that is no longer the member's current one. The client is
    likewise re-connectable after `dispose()`.
21. **Upload quota is reserved before the decode await**, so concurrent uploads
    cannot all measure the same pre-upload totals and collectively overshoot the
    per-room cap. A failed decode releases the reservation.
22. **A full `ai.png` load is discarded if anything newer was painted** while it
    was in flight (a paint generation counter), so a slow recovery fetch can
    never undo a fresher patch.
23. **A full session table costs nobody their identity.** If every recorded
    session belongs to someone still connected, the newcomer gets a working but
    non-resumable identity instead of evicting a live participant's token.
24. **The server binds to `127.0.0.1` by default.** Exposing the room server
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
