# Working in this repo

> Setting this up for someone to play? Follow `docs/SETUP.md`; the rest of this
> file is for people changing the code. The GPU rule below is about the
> author's shared card - on the machine you were asked to set up, loading the
> model is the whole point, so go ahead.

Brush Jam is a shared canvas that an SDXL img2img model reinterprets continuously:
every committed stroke sends the *whole* drawing back through the model, and the
result appears beside it. One Python process (`apps/brushjam`) owns the rooms, the
rasterisation and inference, and serves the browser client; `apps/web` is a thin
canvas that draws locally and relays strokes.

The contract between them is `packages/shared/src/protocol.ts`. Change a message
and you change three places: that file, the Python validator/reducer
(`validate.py`, `room.py`, `protocol.py`), and the tests on both sides - the
frozen fixtures in `apps/brushjam/fixtures/` exist to catch exactly this drift.

## Commands

```bash
pnpm install
pnpm build      # vite build + scripts/build_web.py -> apps/brushjam/src/brushjam/static
pnpm start      # uv run --project apps/brushjam brushjam, on :8787
pnpm dev        # the same server plus Vite on :5173 (hot reload)

pnpm py:test    # pytest, 435 tests
pnpm test       # vitest, all workspaces
pnpm typecheck
```

`pnpm build` is not optional after a client change. The server prefers the
packaged `static/` over `apps/web/dist` (`app.py:default_web_dist`), and that
directory exists here, so a running server serves the *last build* and your edit
is invisible with no error anywhere.

Any Python goes through uv: `uv run --project apps/brushjam <cmd>`. Add
`--no-sync` when you do not want it touching the venv.

Scripts in `apps/brushjam/scripts/`: `download_models.py` (fetch a checkpoint,
prints the `.env` line), `build_web.py`, `verify_unfuse.py`, `bench_render.py`,
`preset_sheet.py` (drives a *running* server, loads no model), `compare_checkpoints.py`.
Tools in `tools/` speak only public HTTP/WS: `pnpm latency`, `pnpm smoke`,
`pnpm playtest-sim`, each `-- --url <server>`.

Pod: `uv run --project apps/brushjam python deploy/runpod/deploy.py <cmd>` -
`deploy`, `build`, `upload`, `status`, `log`, `stop`, `start`, `watch`,
`terminate --yes`. Runbook in `docs/RUNPOD_POD.md`.

## Safety

- **`.env` holds secrets** (`RUNPOD_API_KEY`, `HF_TOKEN`, `CIVITAI_TOKEN`).
  Never print it, never log it, never commit it. `.env.example` is the
  documented shape - edit that when you add a setting.
- **Never commit generated images.** `data/`, `models/`,
  `docs/experiments/**/*.png|jpg` are gitignored deliberately; the reports that
  reference them stay, the pictures are regenerated. The one exception is
  `docs/media/`: the README's demo GIF and screenshot, chosen by hand.
- **The local GPU is shared.** Do not start the `inproc` server, ComfyUI, a
  benchmark, or anything else that loads a model unless the user has said the
  GPU is free. Use `AI_BACKEND=mock` or `INPROC_DRY_RUN=1`. One model at a time:
  the card is 8 GB and two will OOM.
- **Restarting the server destroys every room.** Strokes live in memory; only
  the history JPEGs are on disk. Ask before restarting anything on :8787 when
  someone may be drawing.

## Rules found the hard way

- **Pressure**: only `pointerType === 'pen'` carries a real reading. A mouse
  reports the spec's constant 0.5, so `strokePressure` returns 1 for it -
  mouse strokes are full width (`brushCursor.ts`).
- **Rasters are world-space.** Layers are translated at render time via
  `offsetX/offsetY`; stroke coordinates in the log never move. The noise pen
  hashes *world* coordinates, so a server crop and the client's full-size layer
  come out pixel-identical (`packages/shared/src/noise.ts`).
- **`aiRevision` is the human revision a result corresponds to**, not a count of
  results. A prompt/denoise/seed/profile change regenerates at the same
  revision. For "a new result arrived" use `aiPaintGeneration` (client) or
  `aiGeneration` (protocol) - this is why the overlay follows the live raster
  instead of `ai.png?v={rev}`.
- **Shared settings go through `sharedDraft.ts`.** Prompt, negative prompt,
  denoise and seed are server state with an echo; the draft state machine
  releases a field only when the server confirms *that* value. Never write the
  client field directly, or typing is lost on a slow link.
- **Sends are gated until the socket is admitted.** A socket is open and accepts
  writes before its snapshot arrives; `RoomClient.admitted` is the gate, and a
  write in flight is tagged with `connectionEpoch` so a dead socket's write is
  not mistaken for a live one.
- **VAE tiling must be 256 px on 8 GB** (`SMALL_CARD_VAE_TILE`). diffusers only
  tiles images larger than its tile size and defaults to 1024, so without this
  nothing at 768 or 1024 is tiled at all: 2.2 s an edit becomes 6.7.
- **The DMD2 LoRA is fused for `fast` and unfused for `quality`.** Fuse/unfuse,
  never a PEFT-attached adapter - attaching costs per-request time. `fp8`
  storage cannot fuse; that is a documented trade, not a bug.
- **Style presets need the `quality` profile.** `fast` is 4 steps at CFG 1.0
  where the prompt barely steers and the negative branch is never evaluated;
  raising denoise there buys a different picture, not a look
  (`docs/experiments/2026-09-07-presets/REPORT.md`).
- **R18 presets exist only when `PRESETS_R18=1`.** That is a gate on the menu,
  not on the room: the server accepts any prompt and never inspects one. The
  dice never lands on R18.
- **History entries are immutable numbered files with a persisted counter.**
  Never reuse a number, even after eviction - the URLs are served immutable. An
  entry is its JPEG *and* its JSON, JSON written last.

## Workflow

- Tests stay green. CI runs `pnpm typecheck`, `pnpm -r test`, `pytest`, then
  `pnpm build` and the deploy tooling's checks. Add tests with behaviour.
- Commit messages explain *why*, in the voice of the existing log - a sentence
  about the problem, not a list of files.
- Docs live in `docs/`: `PYTHON_SERVER.md` is the server reference,
  `RUNPOD_POD.md` the pod runbook, `experiments/<date>-<topic>/` a `REPORT.md`
  plus its json (images gitignored). When you change behaviour, update the
  matching doc and the README section that describes it.
- Codex CLI is the adversarial reviewer:
  `codex exec -m gpt-6-astra --sandbox read-only "<prompt>" < /dev/null`.
  Findings get fixed structurally - close the class of bug, do not patch the
  instance.

## Where things are

| file | responsibility |
| --- | --- |
| `apps/brushjam/src/brushjam/app.py` | HTTP routes, the WS endpoint, static client |
| `.../room.py` | the authoritative reducer: strokes, undo, layers, settings |
| `.../runtime.py` | sockets, presence, leases, the AI raster, room eviction |
| `.../scheduler.py` | debounce, one in-flight per room, admission, stale results |
| `.../raster.py` | server-side rendering, the incremental per-layer cache |
| `.../history.py` | saved results on disk: counter, budgets, eviction |
| `.../export.py`, `.../avi.py` | the history zip and the Motion JPEG AVI |
| `.../settle.py` | waiting out work that cannot be cancelled |
| `.../config.py` | every env var, validated once at startup |
| `.../ai/pipeline.py` | the resident SDXL model: profiles, LoRA, VAE, VRAM |
| `.../ai/backends/*` | inproc, stream, comfyui, runpod, mock |
| `apps/web/src/Room.tsx` | the room UI: tools, advanced panel, history strip, export |
| `apps/web/src/StageView.tsx` | the canvas element, camera, pointer plumbing |
| `apps/web/src/roomClient.ts` | the socket, reducer mirror, AI raster, admission |
| `apps/web/src/sharedDraft.ts` | shared-field drafts and their echo state machine |
| `apps/web/src/raster.ts` | client-side layer rasters and image loading |
| `apps/web/src/overlay.ts` | the AI overlay: follow/pin/peek, and its addresses |
| `apps/web/src/gallery.ts` | the history strip: fetch, backoff, selection |
| `apps/web/src/exportOptions.ts`, `ExportDialog.tsx` | what export offers and why |
| `packages/shared/src/protocol.ts` | every message, both directions |
| `packages/shared/src/presets.ts` | the prompt presets, their groups and costs |
| `packages/shared/src/render.ts` | the renderer both sides use |
| `packages/shared/src/camera.ts` | pan, zoom, fit, screen/world conversion |
