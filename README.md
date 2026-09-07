# Brush Jam

[日本語版 README](README.ja.md)

[![CI](https://github.com/tokoroten/brush_jam/actions/workflows/ci.yml/badge.svg)](https://github.com/tokoroten/brush_jam/actions/workflows/ci.yml)

![A drawing on the left being reinterpreted, generation by generation, on the right](docs/media/drawtest.gif)

![The room: human canvas on the left, AI canvas on the right, the history strip above](docs/media/screenshot.jpg)

Several people draw on one shared canvas, and an AI continuously reinterprets
what they have made. Not a prompt box with a picture under it: the canvas is the
prompt. Every stroke anyone commits sends the *whole* drawing back through an
img2img pass, and the result appears beside it a second or two later. Draw a
rough hill and it comes back as a hill; someone else adds a house on it, and the
next pass has a house on a hill, in the same style, because the same picture
went in.

It is a toy for a room of people, not a tool for one. Drawing never waits for
the model - strokes are local and relayed to everyone else immediately, and the
AI runs behind them. The room shares one prompt, one denoise, one seed and one
speed/quality setting, so what changes for one person changes for everyone, and
arguing about the prompt is half the game.

## What you need

- **Python 3.10-3.12** and [uv](https://docs.astral.sh/uv/). One Python process
  is the whole server.
- **Node 20+ and pnpm**, to build the browser client. Not needed at runtime.
- **An NVIDIA GPU with 8 GB or more**, or a rented one - `deploy/runpod/` puts
  the whole thing on a RunPod pod for a few dollars an hour. Without a GPU
  everything still runs against a mock backend that returns instant grey
  rectangles, which is enough to work on the drawing side.
- **An SDXL checkpoint** (a 6-7 GB `.safetensors`), named by `INPROC_CHECKPOINT`
  in `.env`. Any of them works. The default, which `scripts/download_models.py`
  fetches and every measurement in `docs/` was taken with, is
  [Nova Anime XL IL v19](https://civitai.com/models/376130): Illustrious-based,
  anime-leaning, and chosen because its Civitai licence permits running it as a
  generation service for other people, which is what hosting a room for friends
  is. Read that licence before you download it, and check the same thing about
  any checkpoint you substitute - some popular ones forbid exactly this. A
  photographic or painterly model works too; the presets and the measured
  denoise values were tuned on this one. The 4-step DMD2 LoRA and the fp16-fix
  VAE download themselves on first use.

**How much VRAM.** 8 GB is the design point and it is enough for both profiles:
`fast` at 768, and `quality` at 1024 peaking around 7 GB, because the VAE
decodes in 256 px tiles and the text encoders sit on the CPU. Both of those are
chosen automatically from the card's size. If something else needs the card at
the same time, `INPROC_UNET_STORAGE=fp8` frees another 2.4 GB at the cost of
about a second an edit; `fp16` is faster and is what `auto` picks above 7 GB.

## Or let an agent set it up

Everything above is written to be followed by a coding agent as much as by a
person. If you have [Claude Code](https://claude.com/claude-code) or
[Codex CLI](https://github.com/openai/codex), clone the repository, open it
there and paste this:

> I cloned this repository. Follow `docs/SETUP.md` to get it running on my GPU,
> then expose it with ngrok so friends can join. Tell me what you need from me
> before you need it.

What only you can bring, so have them ready before you ask:

- an NVIDIA GPU with 8 GB or more (or none, for the mock backend);
- a [Civitai](https://civitai.com/) account and API token, for the checkpoint;
- an [ngrok](https://ngrok.com/) account and authtoken, for the tunnel.

The agent will install Node, pnpm and uv, build the client, download the model,
write `.env`, start the server and hand you the room URL. `CLAUDE.md` is aimed
at it too: it is the map of the repository and its sharp edges.

## Quick start

```bash
git clone <this repo> && cd brush_jam
pnpm install
pnpm build                       # builds the web client into the Python package

cp .env.example .env             # then set INPROC_CHECKPOINT (see below)

cd apps/brushjam
uv sync --extra inproc           # torch + diffusers
cd ../..

pnpm start                       # http://localhost:8787
```

The Python server serves the client itself, out of
`apps/brushjam/src/brushjam/static` - and `pnpm build` is what puts it there
(`vite build`, then `scripts/build_web.py` copies `apps/web/dist` across). A
running server keeps serving the last build until that command runs again, so
after changing anything under `apps/web/` either rebuild or use `pnpm dev`,
which puts the Vite dev server on :5173, with hot reload, in front of the same
Python server.

**Without a GPU**, install without the extra (`uv sync`) and set
`AI_BACKEND=mock` in `.env`: everything works except the model, which returns
instant grey rectangles. `AI_BACKEND=inproc` is not a preference but an
instruction, and the server refuses to start rather than quietly running
something else - it is `AI_BACKEND=auto` that falls back to ComfyUI and then to
the mock.

If you do not have a checkpoint yet:

```bash
uv run --project apps/brushjam python apps/brushjam/scripts/download_models.py
```

It downloads one into `./models/checkpoints/` and prints the
`INPROC_CHECKPOINT=` line to paste into `.env`. (Civitai wants an account for
most models: put `CIVITAI_TOKEN` in `.env` first; `CIVITAI_VERSION` chooses a
different model version.)

Then open <http://localhost:8787>, pick a name, press **Create room**, and send
the `/r/<id>` URL to someone. Draw on the left; the AI's version appears on the
right. The startup log says which backend it chose and why:

```
[brushjam] backend: inproc, sdxl-illustrious.safetensors resident in this process
[brushjam] server on http://127.0.0.1:8787
```

With `AI_BACKEND=auto`, no torch and no checkpoint, it falls back to ComfyUI if
one is running and to the mock backend otherwise, saying which and why. With
`AI_BACKEND=inproc` it refuses to start instead: an explicit choice that cannot
be honoured is an error, not something to work around silently.

## Playing with friends

**On your LAN**, bind to every interface and give people your machine's address:

```bash
HOST=0.0.0.0 pnpm start          # then http://<your-ip>:8787/r/<id>
pnpm dev:lan                     # the same, with the Vite dev server alongside
```

**Over the internet from your own machine**, put a tunnel in front of it -
`ngrok http 8787`, `cloudflared tunnel --url http://127.0.0.1:8787`, or
Tailscale for a closed group. The server serves everything, WebSocket
included, on that one port, and the client picks `wss://` by itself when the
page is https. Raise `ROOM_CREATE_PER_MIN` first: the server rate-limits by the
socket's own address and does not read `X-Forwarded-For`, so behind a tunnel
every player shares one bucket. Step by step, in Japanese, in
[`docs/SETUP.md`](docs/SETUP.md).

**Over the internet on a rented GPU**, put the server on a RunPod pod. `deploy/runpod/` creates
a RunPod pod, ships this repository into it as a tarball and starts it, with no
SSH, no registry and no git remote involved:

```bash
uv run --project apps/brushjam python deploy/runpod/deploy.py deploy
```

The runbook - first boot, what a redeploy costs, how to watch it - is
[`docs/RUNPOD_POD.md`](docs/RUNPOD_POD.md). Anyone with the pod's URL can join,
so treat the link as the only access control there is.

## The controls

- **tools.** Pen, eraser, a **noise pen** whose texture is hashed from world
  coordinates (identical for everyone, stable under any crop), and a move tool
  for whole layers. The colour swatch is greyed out for all but the pen: noise
  brings its own colours, the eraser removes, and move paints nothing.
  Ctrl/Cmd+V pastes a reference image as a layer, excluded from the AI's input
  until you tick "AI input" beside it in the layer panel.
- **brush size.** The ring under the pointer is the brush, at the size it will
  actually land: the slider is in world pixels and the canvas is usually zoomed
  out, so the number alone says very little. It replaces the mouse pointer while
  a drawing tool is active, and becomes a small cross when the brush is smaller
  than a ring can show. Pressure is only read from an actual pen - a mouse
  reports the constant 0.5 the Pointer Events spec assigns to hardware with no
  sensor, so mouse strokes are drawn full width.
- **layers.** The panel on the right: visibility, opacity, order, lock, and the
  move tool for sliding a whole draw layer around. A layer is translated at
  render time, so the strokes in the log never move - and moving a layer is not
  undoable.
- **overlay.** At the top of the layer panel: lays the AI's result over your own
  canvas at an opacity you choose (40% by default), so you can trace it - draw a
  rough hill, let the model make a hill of it, then draw over what it invented.
  Hold **Tab** to see what you have actually drawn. **pin** freezes the picture
  so the next generation does not move it under your hand, and any entry in the
  history strip can be pinned the same way. It is yours alone: nobody else in
  the room sees it, it is not part of "save drawing", and the AI never sees it
  either.
- **prompt.** Shared. It steers the whole canvas. The negative prompt is in
  Advanced, and is inert on `fast`: a distilled model at CFG 1.0 never evaluates
  the negative branch, and the UI greys it out rather than pretending.
- **presets.** A picker beside the prompt fills it from about two dozen looks,
  grouped as 基本 / basic (anime girl, landscape, impressionist, architecture,
  background art), 画風 / style (sumi-e, ukiyo-e, stained glass, pixel art,
  watercolour picture book, claymation, papercraft), 題材 / subject (fantasy
  map, creature design, nebula, food, satellite view, mecha blueprint,
  botanical), 雰囲気 / mood (neon city, horror, retro poster, photorealistic),
  and R18 (NSFW) on its own at the end - which the server only offers when it
  was started with `PRESETS_R18=1`, and hides otherwise. That is a gate on the
  menu and nothing else: any client can send any prompt, the server accepts it,
  and it does not look at what a prompt says. The ⚀ beside the picker rolls one
  at random, never R18, whether or not the group is shown. A preset only fills
  the fields - the prompt stays editable and nothing on the server knows one was
  used - though most also nudge the denoise, and ten of them switch the room to
  `quality`, because that is what their look costs (below).
- **fast / quality.** In Advanced, with the rest of the AI settings. `fast` is a
  4-step distilled LoRA at 768 - a couple of seconds an edit on a 3070, which is
  what makes the loop feel alive. `quality` is 14 steps at 1024, several times
  slower and considerably better. Room-wide.

  **A style costs steps, not denoise.** On `fast` the sampler runs four steps at
  CFG 1.0: the prompt barely steers and the negative branch is never evaluated
  at all, so turning the denoise up there does not buy a look, it buys a
  different picture. On `quality` the same words land *and* the composition
  survives all the way to denoise 0.8. That is why every style preset except
  pixel art asks for `quality` - the honest price of the look
  ([`docs/experiments/2026-09-07-presets/`](docs/experiments/2026-09-07-presets/REPORT.md)).
- **denoise.** How far the model may depart from the drawing. Low values
  recolour, high values reinterpret. 0.8 suits `fast`.
- **seed.** The room keeps one seed rather than drawing a new one per
  generation, so adding a stroke changes the picture instead of reshuffling it.
  The dice beside the field asks for a different picture from the same drawing.
- **resolution.** Generation size; the result is scaled onto the canvas.

Measured through `pnpm latency`, `stroke_end` to pixels:

| | `fast` / 768 | `quality` / 1024 |
| --- | --- | --- |
| RTX 3070 8 GB, local | 2.2 s | 9.4 s |
| RTX 4090 RunPod pod | 0.8 s pipeline, 1.5 s to the client | 3.1 s |

## Saving and history

**export** in the header opens one dialog with every way out of a room:

- **Download drawing (PNG)** - the visible layers composited in the browser at
  canvas size, on white, exactly as they are on the stage.
- **Download AI image (PNG)** - the current AI result, as a PNG from the
  server. Both land as `brushjam-<room>-<revision>-drawing.png` / `-ai.png`.
- **Export history as ZIP** - `history.zip`: every stored frame as
  `draw_NNNNN.jpg` (the canvas the model was handed) and `gen_NNNNN.jpg` (what
  it made of it), plus a `manifest.json` of the settings behind each one -
  prompt, negative prompt, denoise, seed, profile, resolution, latency, and the
  checkpoint and LoRA that produced it. Entries written before the server kept
  the input have no `draw_NNNNN.jpg`, and say so in the manifest.
- **Download video (Motion JPEG AVI)** - `history.avi`: the drawing on the left
  and the result on the right, one frame per generation, at 2, 4 or 8 fps. It
  plays in VLC and opens in any video editor. Past `HISTORY_EXPORT_MAX_FRAMES`
  (3000, twelve minutes at the default 4 fps) the answer is the zip instead.

The last two are greyed out, with the reason beside them, on a server started
with `HISTORY_ENABLED=0` or in a room that has not generated anything yet.

**history** opens a strip of every AI result the room has made, newest first.
The server writes each accepted result to `HISTORY_DIR` (`./data/history` by
default) as a JPEG plus a JSON entry recording the prompt, negative prompt,
denoise, seed, profile, resolution, latency and model that produced it. Clicking
a thumbnail shows it large with those settings, a download link, **use these
settings** - which puts the prompt, negative prompt, denoise and seed back into
the room, but not the profile or the resolution, which are what the machine can
do now rather than part of the look - and **pin as overlay**.

It is on disk, not in the room, so it survives the room being evicted and the
server restarting, and `GET /rooms/{id}/history` still answers for a room that
no longer exists. Each room also keeps a small `counter` file beside its
entries: image URLs are served as immutable, so a number is spent when it is
handed out and never comes round again, even after every entry that used one
has been evicted. A room written before those counters existed gets one at
startup, before anything can be evicted, and is kept rather than evicted if
that write fails - the names of its files are then the only record of which
numbers are spent. An entry is its JPEG *and* its JSON - the JSON is written
last - and a half-written pair, like a leftover `.part` file, is cleaned up
when the store is next read, or counted and retried if it will not delete. Two
budgets keep it from filling the disk - `HISTORY_ROOM_MB` (200) and
`HISTORY_TOTAL_MB` (2000), oldest evicted first - and `HISTORY_ENABLED=0` turns
the whole thing off.

## How it works

One Python process (`apps/brushjam`) serves the built client, the room protocol
and inference. It is the only server.

- **Truth is a log.** The room owns an append-only stroke log plus a set of
  undone stroke ids. Every raster is derived and can be rebuilt, so a client
  that reconnects gets a `snapshot` and is exactly caught up. `undo` reverts the
  *sender's* own latest stroke, which is what makes it usable with several
  people drawing at once.
- **The AI sees everything.** After `AI_DEBOUNCE_MS` of quiet the scheduler
  renders the whole canvas server-side - with the same renderer the browser uses
  - and hands it to the backend. One generation is in flight per room; activity
  during one queues exactly one more; a result older than the last accepted one
  is discarded. Each draw layer's raster is kept between generations and only
  new strokes are drawn onto it, so a room that has been going for an hour
  renders as fast as one that just started.
- **Nobody gets the whole machine.** A generation is admitted through a
  process-wide slot taken *before* rasterising, and holds it until the work has
  physically stopped rather than until the await returned. Joining a room takes
  a lease, so a reconnecting tab takes over its own place instead of queuing
  behind itself. Room creation, sockets per room, sockets in total and the size
  of a room's stroke log all have ceilings (`.env.example`, "limits").
- **Drawing never waits.** Strokes are drawn locally on pointer input and
  relayed as chunks every ~40 ms. Nothing in that path touches the model.
- **The protocol is the contract.** `packages/shared/src/protocol.ts` defines
  every message, and the server validates each one before the reducer sees it.

## Layout

```
apps/brushjam/         the server: client, protocol and inference in one process
  src/brushjam/room.py        the authoritative reducer (strokes, undo, layers, settings)
  src/brushjam/runtime.py     sockets, presence, the AI raster, room eviction
  src/brushjam/scheduler.py   debounce, single in-flight, stale results
  src/brushjam/history.py     saved results on disk
  src/brushjam/export.py      the zip and the video
  src/brushjam/ai/pipeline.py the resident SDXL model
  src/brushjam/ai/backends/   inproc | stream | comfyui | runpod | mock
  src/brushjam/static/        the built client (pnpm build puts it here)
  scripts/                    download_models, build_web, bench_render, preset_sheet, ...
  tests/                      435 tests, including the frozen parity fixtures

apps/web/              the browser client (Vite + React 19)
packages/shared/       protocol, geometry, presets, the renderer both sides use
tools/                 scripts against a running server, over HTTP/WS only
apps/stream-worker/    optional: the model on another machine
deploy/runpod/         put the server on a rented GPU
```

## Development

```bash
pnpm dev            # Python server on :8787, Vite on :5173 with hot reload
pnpm dev:lan        # ...both reachable from the LAN

pnpm py:test        # the Python suite
pnpm test           # the TypeScript suites
pnpm typecheck

pnpm smoke -- --url http://127.0.0.1:8787          # one real generation
pnpm latency -- --url http://127.0.0.1:8787 --n 5  # per-edit latency
pnpm playtest-sim -- --url http://127.0.0.1:8787 --users 3 --minutes 1
```

CI runs all three test commands on every push, then `pnpm build` and the deploy
tooling's own checks, which pack the tarball the built client goes into.

The three scripts in [`tools/`](tools/README.md) speak only the public HTTP and
WebSocket surface, so they measure whatever is actually listening - local,
remote, or a pod. `AI_BACKEND=mock` runs the whole server with no model at all,
which is how the tests and most of the client work get done.

The scripts under `apps/brushjam/scripts/` are the ones that want the card:
`bench_render.py` times the AI input render, `verify_unfuse.py` checks that
unfusing the fast LoRA gives the quality profile its model back, and
`compare_checkpoints.py` renders the same drawings through two checkpoints, one
model at a time. `preset_sheet.py` needs no card of its own: it drives a
*running* server as an ordinary client to contact-sheet the presets.

## Documents

- [`docs/SETUP.md`](docs/SETUP.md) - セットアップマニュアル(日本語): install,
  models, `.env`, LAN play, and exposing a local server with ngrok, Cloudflare
  Tunnel, Tailscale or RunPod.
- [`docs/PYTHON_SERVER.md`](docs/PYTHON_SERVER.md) - the server: backends,
  scheduling, limits, measurements, review history.
- [`docs/RUNPOD_POD.md`](docs/RUNPOD_POD.md) - deploying to a rented GPU.
- [`docs/STREAM_WORKER.md`](docs/STREAM_WORKER.md) - the remote model host, and
  a long account of what actually makes SDXL fast on 8 GB.
- [`docs/RUNPOD.md`](docs/RUNPOD.md) - the serverless endpoint: torn down, but
  measured (historical).
- [`docs/MVP_PLAN.md`](docs/MVP_PLAN.md) - the original plan (historical).
- [`docs/experiments/`](docs/experiments/) - denoise sweeps, the checkpoint
  comparison and the preset rewrite, each with its `REPORT.md`.
- [`CLAUDE.md`](CLAUDE.md) - the house rules, for an AI agent working here.

## Limits and known issues

- **8 GB is the design point**, and it is tight: the VAE decodes in 256 px
  tiles and the text encoders are offloaded to the CPU. `quality` at 1024 peaks
  around 7 GB of the 8. Both of those are automatic, from the card's size;
  `INPROC_UNET_STORAGE=fp8` frees another 2.4 GB when something else needs the
  card, at about a second an edit (docs/PYTHON_SERVER.md, "8 GB cards").
- **One model on one GPU.** Running ComfyUI and this server at once on an 8 GB
  card will OOM one of them. The `stream` backend exists so the model can live
  on a second machine.
- **The negative prompt does nothing on `fast`.** Distilled models at CFG 1.0
  never evaluate it. The UI says so.
- **Everyone behind one address shares the rate limits**, which matters on a
  RunPod pod, where the proxy makes every player look like a single client. The
  pod configuration raises `ROOM_CREATE_PER_MIN` for that reason.
- **Sparse line art on white reinterprets weakly.** The model has little to work
  with; denoise, prompt and a filled background help more than any server
  setting, and the noise pen is the quickest way to give a style something to
  bite on.
- **A restart loses every room.** Strokes live in memory; only the saved history
  JPEGs are on disk.
- No redo, and `clear_layer` is not undoable.

## License

MIT. See [LICENSE](LICENSE).
