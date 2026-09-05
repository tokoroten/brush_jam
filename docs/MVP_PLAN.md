# Brush Jam — MVP Implementation Plan (vertical slice)

> **Historical.** This plan describes the original TypeScript server in
> `apps/server`, which has been replaced by the single Python process in
> `apps/brushjam` and deleted. The design it describes is still the design;
> only the file paths have moved. See [`PYTHON_SERVER.md`](PYTHON_SERVER.md).

Derived from `BRUSHJAM_CONTEXT.md` §23 phases 1–3 and §26. Everything here is MVP scope. Nothing marked Post-MVP in the context doc is to be built.

## 0. Local environment facts (verified 2026-09-05)

- Windows 11, Node 22, pnpm, Python 3.12, uv, git.
- **ComfyUI 0.28.0 is installed at `E:\ComfyUI` and already running at `http://127.0.0.1:8188`.**
- GPU: RTX 3070, 8 GB VRAM. Checkpoint: `waiNSFWIllustrious_v150.safetensors` (SDXL / Illustrious class).
- Core nodes available: `CheckpointLoaderSimple, LoadImage, VAEEncodeForInpaint, SetLatentNoiseMask, InpaintModelConditioning, KSampler, VAEDecode, SaveImage, ImageCompositeMasked`. Custom nodes: IPAdapter_plus, GGUF, VideoHelperSuite, Manager (none required).
- `.env` (git-ignored, repo root) holds `RUNPOD_API_KEY, HF_TOKEN, CIVITAI_TOKEN, VASTAI_*`. Not needed for the local prototype.

Consequence: **the local AI backend is ComfyUI over HTTP.** No custom Python worker. Later, the exact same workflow JSON can be sent to RunPod's `worker-comfyui` serverless image, so "cloud" is a transport change, not a pipeline change.

## 1. Stack (decided)

- **Monorepo**: pnpm workspaces, TypeScript strict, Node 22, `vitest`.
- `packages/shared` — protocol types + pure logic: geometry, dirty-region merge, crop selection, revision / stale rules, per-user undo selection, mask geometry, and a `renderStrokes(ctx, …)` that draws onto any `CanvasRenderingContext2D`-compatible ctx. No DOM / Node imports. Fully unit-tested.
- `apps/server` — Node + `ws` + `node:http` (no web framework). In-memory room state. Server raster via `@napi-rs/canvas` (prebuilt binaries; works on Windows). Serves the built client in prod.
- `apps/web` — Vite + React + TS. Canvas 2D, Pointer Events, one WebSocket per room.
- Env for server: `PORT=8787`, `AI_BACKEND=comfyui|mock|runpod` (default `comfyui` if `COMFYUI_URL` reachable at boot, else `mock`), `COMFYUI_URL=http://127.0.0.1:8188`, `COMFYUI_CHECKPOINT=waiNSFWIllustrious_v150.safetensors`, `AI_WINDOW=1024`, `AI_APPLY=768`, `AI_STEPS=14`, `AI_DENOISE=0.55`, `AI_DEBOUNCE_MS=400`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`.

## 2. World model

- Logical canvas **4096×4096**, origin top-left, world unit = pixel at zoom 1.
- `Camera = { centerX, centerY, zoom }` — ONE camera object drives both Human and AI views.
- Layers: max 8, kinds `draw | reference`. The AI canvas is a separate room-level raster, not a layer. `reference` layers (pasted images) are visible but **excluded from AI input by default** (`includeInAI: false`).
- Strokes are the unit of truth: `{ id, userId, layerId, tool: 'pen'|'eraser', color, width, points: [x,y,pressure?][], revision, bbox }`. Server keeps an append-only stroke log per room plus a set of undone stroke ids. Raster is derived, never authoritative.
- `humanRevision` increments on every accepted mutating event (stroke_end, undo, clear, layer_*). `aiRevision` = the humanRevision the last accepted AI result was rendered from.

## 3. Rendering strategy

- **Client**: one offscreen canvas per layer at full 4096² (≈64 MB RGBA each; 8 layers is acceptable in a browser). Composite visible layers → Human view under the camera transform. Undo / clear re-render only the affected layer from the stroke log. Eraser = `destination-out` on the layer canvas.
- **Server**: does NOT keep full per-layer rasters. For an AI request it creates an `AI_WINDOW`² `@napi-rs/canvas`, translates by `-crop.x, -crop.y`, and replays only strokes whose bbox intersects the crop (log order, skipping undone ids, honoring layer visibility / opacity / order / `includeInAI`). Reference layers with `includeInAI` draw their image. Same `renderStrokes` from `packages/shared` runs on both sides so pixels agree. Background is white (SDXL needs an opaque input).
- **AI canvas on server**: one persistent 4096² `@napi-rs/canvas`, initially transparent. Server composites each accepted AI patch into it through the soft mask and sends clients the composited rect as PNG. Late joiners fetch `GET /rooms/:id/ai.png` once (4096² PNG, mostly transparent, acceptable for MVP).

## 4. Protocol (WebSocket JSON; images over HTTP)

`ws://host/ws/rooms/:roomId?name=…`

Client → server: `cursor, stroke_start, stroke_chunk, stroke_end, undo, clear_layer, layer_create, layer_update, layer_delete, layer_reorder, set_prompt`.
Server → client: `snapshot` (members, layers, prompt, revisions, full stroke log + undone ids), `presence`, `cursor`, relayed `stroke_start/chunk/end` with `userId`, `stroke_committed`, `undo_applied {strokeId}`, `clear_applied {layerId}`, `layer_*`, `prompt_changed`, `ai_status {state: idle|queued|generating|error, message?, forRevision}`, `ai_result {rect, url, aiRevision}`, `error`.

- Cursor: client throttles to 20 Hz, latest-wins, server relays without queueing per-message state.
- Stroke chunks: client batches points every animation frame (≈16 ms). Remote in-progress strokes are drawn live from chunks on a scratch canvas, then committed on `stroke_end`.
- **Collaborative undo**: `undo` reverts the sender's latest not-yet-undone stroke (context doc §4.1 Alice/Bob example). No redo in MVP. `clear_layer` clears one layer (all users' strokes on it), room-level, not undoable.
- Pasted images: `POST /rooms/:id/images` (PNG/JPEG/WebP; client downscales to ≤2048 on the long side first) → `{imageId, width, height}`; then `layer_create {kind:'reference', imageId, x, y, scale}`. Move / scale after paste = `layer_update` (drag with the "move" tool when a reference layer is selected; scale via a slider in the layer panel — no transform handles).
- `GET /rooms/:id/images/:imageId` serves the original bytes.

## 5. AI pipeline (server, `apps/server/src/ai/scheduler.ts`)

1. Every committed stroke / undo / clear adds its bbox to `dirtyRegions` (merge with any existing region whose expanded bbox (+256 px) intersects).
2. Debounce `AI_DEBOUNCE_MS` after the last activity.
3. Crop: `AI_WINDOW`² centered on the bbox of the **most recent dirty region**, clamped to the canvas. If the dirty union is larger than the apply area, still one crop (no tiling in MVP).
4. Render input (§3) and build the mask: union of dirty-stroke bboxes since the last accepted result → dilate 48 px → feather 32 px (draw with a blurred shape onto a mask canvas) → multiplied by a soft rectangle limited to the central `AI_APPLY`² (context vs applied region, §5.3 of context doc). Mask is 8-bit L, white = regenerate.
5. `backend.generate({ prompt, negativePrompt, imagePng, maskPng, denoise, steps, seed, size })` tagged with `humanRevision`. At most **one in-flight request per room**; new activity during flight sets `pending`, which triggers one more run after completion using the merged dirty regions.
6. Result: discard if `forRevision < room.lastAcceptedAIRevision` (latest useful wins). Otherwise composite through the mask into the AI canvas, set `aiRevision = forRevision`, broadcast `ai_result`, and remove the dirty regions that were fully inside the applied area (regions partially outside remain and will trigger a follow-up crop).
7. Errors: broadcast `ai_status error` with message, back off 2 s, keep the dirty regions.
8. Human drawing never waits on any of this.

## 6. AI backends (`apps/server/src/ai/backends/`)

Common interface:
```ts
interface AIBackend {
  generate(req: {
    prompt: string; negativePrompt: string;
    imagePng: Buffer; maskPng: Buffer;       // both size×size
    size: number; denoise: number; steps: number; seed: number;
  }, signal: AbortSignal): Promise<Buffer /* PNG RGB size×size */>;
}
```

- **`comfyui`** (primary for the local prototype):
  1. `POST {COMFYUI_URL}/upload/image` (multipart, `overwrite=true`, unique filename per room+revision) for the image and for the mask.
  2. Build the API-format workflow (a plain JSON object, built in TS — see below), `POST /prompt` with `{prompt, client_id}`.
  3. Poll `GET /history/{prompt_id}` every 250 ms until outputs exist (or use the `/ws` progress socket — polling is fine for MVP). Respect the abort signal by `POST /interrupt` only if this prompt is the running one; otherwise just drop the result.
  4. `GET /view?filename=…&type=output` → PNG bytes.
  5. Delete uploaded inputs opportunistically (not required).

  Workflow (API format), node ids fixed so tests can assert on it:
  ```
  1 CheckpointLoaderSimple  { ckpt_name: COMFYUI_CHECKPOINT }
  2 CLIPTextEncode          { text: prompt + ", masterpiece, best quality", clip: [1,1] }
  3 CLIPTextEncode          { text: negative ("lowres, bad anatomy, bad hands, text, error, worst quality, low quality, jpeg artifacts, signature, watermark, blurry"), clip:[1,1] }
  4 LoadImage               { image: <uploaded image name> }
  5 LoadImage               { image: <uploaded mask name> }   → use output 0 (IMAGE)
  6 ImageToMask             { image: [5,0], channel: "red" }
  7 VAEEncode               { pixels: [4,0], vae: [1,2] }
  8 SetLatentNoiseMask      { samples: [7,0], mask: [6,0] }
  9 KSampler                { model:[1,0], positive:[2,0], negative:[3,0], latent_image:[8,0],
                              seed, steps: AI_STEPS, cfg: 5.5, sampler_name: "euler_ancestral",
                              scheduler: "normal", denoise: AI_DENOISE }
  10 VAEDecode              { samples:[9,0], vae:[1,2] }
  11 SaveImage              { images:[10,0], filename_prefix: "brushjam/<roomId>" }
  ```
  `SetLatentNoiseMask` (not `VAEEncodeForInpaint`) keeps the human drawing as the img2img base so the AI "reinterprets" rather than fills holes. Denoise 0.55 default; the room can't change it (hidden), but env can.
  8 GB VRAM: `AI_WINDOW=1024` default locally. Expect ~8–15 s per generation at 14 steps; that is acceptable for the playtest. Do not implement TensorRT / StreamDiffusion.

- **`mock`** — no GPU: deterministic stylization inside the mask (posterize + edge darken + hue shift derived from a hash of the prompt), 800 ms simulated latency. Used by tests and by anyone without ComfyUI.
- **`runpod`** — same workflow JSON posted to `https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync` in the `worker-comfyui` input format (`{ input: { workflow, images: [{name, image: base64}] } }`), output images base64. Implement the adapter (small) but it is untested against a live endpoint in this slice; document that in the README. Never deploy anything automatically.

## 7. UI (deliberately ugly, functional)

- Top bar: room id, "Copy invite URL", presence chips (name + color), room prompt input (debounced 500 ms → `set_prompt`), AI status pill (idle / generating / error + last latency ms).
- Body: **left Human canvas, right AI canvas**, equal size, same camera. Wheel = zoom at cursor (0.05–8×), space+drag or middle-drag = pan, buttons "Fit" and "100%". A faint rectangle on both views shows the last AI crop and applied area.
- Tool strip: pen, eraser, move (reference layers), color picker, size slider (1–128), undo (Ctrl+Z), clear layer.
- Layer panel: list (top = frontmost), add, delete, rename (double-click), eye, lock, opacity slider, ↑/↓ reorder, "AI input" checkbox (reference layers only; draw layers are always AI input), scale slider for reference layers.
- Remote cursors: colored dot + name on the Human view; also on the AI view (cheap, helps spatial sync).
- Ctrl/Cmd+V: image → reference layer centered in the current viewport; text while the prompt input is focused → normal paste.
- Home page `/`: "Create room" → `/r/<id>`; join by URL; name is prompted once and kept in `localStorage`.
- Narrow screens: stack vertically. No further mobile work.

## 8. Tests required

- shared: dirty-region merge, crop selection + clamping, stale-result rule, per-user undo selection (Alice #100 / Bob #101 / Alice #102 → undo removes #102), mask geometry bounds, renderer determinism (same strokes → identical pixels using `@napi-rs/canvas` in a server-side test).
- server: room reducer (join/leave, strokes, undo, clear, layers, revision increments), scheduler with a fake backend + fake timers (debounce, single in-flight, pending re-run, stale discard, error backoff), ComfyUI adapter against a stubbed `fetch` (upload → prompt → history → view sequence and workflow shape), ws integration test with two `ws` clients seeing each other's strokes and undo.
- web: camera math and paste-downscale helper only.

## 9. Delivery checklist

- `pnpm install && pnpm dev` starts server (:8787) + web (:5173). With ComfyUI running, two browser tabs draw together and AI patches appear on the right. Without ComfyUI, the mock backend is used automatically and a log line says so.
- `pnpm test`, `pnpm typecheck`, `pnpm build` all green on Windows.
- `README.md`: run instructions, env vars, ComfyUI requirements (checkpoint name), how to switch to RunPod later.
- Keep files small and modules testable. No Post-MVP features (regional prompts, timelapse, roles, billing, Discord, redo, blend modes, masks, folders).
