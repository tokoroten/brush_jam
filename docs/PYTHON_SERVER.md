# The single Python server (`apps/brushjam`)

One process serves the built client, the room protocol (HTTP + WebSocket) and
inference. It is a port of `apps/server` (Node) plus `apps/stream-worker`, and
the browser client is unchanged: `packages/shared/src/protocol.ts` is still the
contract, byte for byte.

```bash
cd apps/brushjam
uv sync --extra dev            # room server + tests, no GPU
uv sync --extra dev --extra inproc   # + torch/diffusers for the resident model
uv run pytest
uv run brushjam                # http://127.0.0.1:8787
```

`WEB_DIST=../web/dist` (or `uv run python scripts/build_web.py`, which copies
the built client into the package) is what makes it serve the page too.

## Layout

| file | what it is |
| --- | --- |
| `src/brushjam/protocol.py`, `validate.py` | the wire contract and the hand-written validation of every client message |
| `src/brushjam/room.py` | the authoritative reducer: strokes, per-user undo, layers, settings, sessions |
| `src/brushjam/runtime.py` | rooms, sockets, presence, image store, eviction, capability refresh |
| `src/brushjam/scheduler.py` | full-canvas AI scheduler: debounce, one in flight, stale discard, error policy |
| `src/brushjam/raster.py` | Pillow + numpy rendering of the AI input and the AI canvas, and the per-layer raster cache |
| `src/brushjam/ai/pipeline.py` | the resident SDXL img2img model (moved from `apps/stream-worker`) |
| `src/brushjam/ai/backends/` | `inproc`, `stream`, `comfyui`, `runpod`, `mock` |
| `src/brushjam/app.py`, `main.py` | routes, WebSocket, static client, uvicorn entry |

## Backends

`AI_BACKEND` picks one; `auto` (the default) prefers the in-process model, then
a stream worker **only** if `AI_STREAM_AUTO=1`, then ComfyUI, then the mock.
An explicit choice is never silently replaced - `AI_BACKEND=inproc` with no
torch or no checkpoint refuses to boot rather than quietly becoming the mock.

| backend | what it is | profiles |
| --- | --- | --- |
| `inproc` | the model in this process, one copy on the GPU | fast + quality |
| `stream` | `apps/stream-worker` over HTTP (separate process / machine) | fast only |
| `comfyui` | local ComfyUI | fast + quality (quality only without a LoRA) |
| `runpod` | the same ComfyUI graph on a serverless endpoint | fast + quality |
| `mock` | GPU-free stylisation, same contract | fast + quality |

### The in-process pipeline

One resident checkpoint serves both room profiles, switched **per request**:

| profile | LoRA | scheduler | steps | cfg | negative prompt |
| --- | --- | --- | --- | --- | --- |
| `fast` | DMD2 4-step, attached | `LCMScheduler` | 4 | 1.0 | inert (no CFG branch) |
| `quality` | detached | `EulerAncestralDiscreteScheduler` | 14 | 5.5 | active |

The LoRA stays **loaded** and is **fused and unfused per profile**. The worker
this was ported from fused once at load and dropped the adapter, so it paid no
per-step cost and could never offer a quality profile. Leaving the adapter
merely attached is the opposite mistake: PEFT then runs its own matmuls on
every Linear on every step, which measured as roughly a doubling of `fast` at
768. So: `fuse_lora(components=["unet"], adapter_names=["fast"])` for `fast`,
`unfuse_lora(components=["unet"])` for `quality`. The weights are already
resident, so a switch is a weight operation, never a disk load.

Order matters and is not interchangeable: PEFT's forward *unmerges* a merged
layer as soon as adapters are disabled, so it is enable-then-fuse going in and
unfuse-then-disable coming out. A build that cannot fuse falls back to the
attached adapter and says so in the log. `/healthz` reports `lora_fused`, and
each run's timings carry `profile_switch_ms`.

Unfusing has to give the quality profile its model back. PEFT merges by adding
`B @ A * scale` and unmerges by subtracting the same product, so the only
question is fp16 rounding: on a 1280x1280 projection at rank 64 the residual
after a cycle is under 0.1% of the change the fuse made, and ten cycles do not
drift (`tests/test_inproc.py`). `scripts/verify_unfuse.py` checks the same
thing end to end on the real model, comparing a quality render before any fuse
against one after a fuse/unfuse cycle at the same seed.

Everything else is kept as the worker measured it:

- **fp16-fix VAE** by default (`INPROC_VAE=fp16fix`). The checkpoint's own VAE
  sets `force_upcast=True`, so diffusers casts it to fp32 on every encode and
  decode - the dominant fixed cost per request.
- **Text encoders parked in system RAM** between requests (~1.8 GB in fp16),
  with an LRU cache of prompt embeddings keyed by `(prompt, negative, cfg)`.
- **`torch.cuda.empty_cache()` in a `finally`**, on every exit path including a
  cancelled run: without it the allocator reserves ~7.5 GB against 5 GB live on
  an 8 GB card and the next request's activations spill into shared memory.
- **Cooperative cancellation.** Dropping the caller cannot stop GPU work, so a
  cancelled request sets a flag and the diffusion loop's step callback raises at
  the next step boundary. One dedicated worker thread *is* the GPU lock: the
  next request queues behind the unwinding one instead of racing it.
- **`fast` uses an explicit LCM timestep schedule** rather than the pipeline's
  `strength` argument, because two integer roundings made denoise 0.8 and 0.9
  produce byte-identical images at 4 steps.

### Environment

Everything in the README's table still applies. In addition:

| variable | default | meaning |
| --- | --- | --- |
| `AI_BACKEND=inproc` | auto | run the model in this process |
| `INPROC_CHECKPOINT` | the ComfyUI checkpoint path | single-file SDXL `.safetensors` |
| `INPROC_LORA` | `dmd2` | `dmd2` or `lcm`; decides the fast profile's cfg |
| `INPROC_LORA_DIR` | ComfyUI's `loras` | shared on purpose: one file, two consumers |
| `INPROC_VAE` | `fp16fix` | `fp16fix`, `taesd` or `checkpoint` |
| `INPROC_VAE_TILE_SIZE` | `auto` | VAE tile edge in px (multiple of 64, 128-1024); `auto` = 256 under 12 GB |
| `INPROC_UNET_STORAGE` | `auto` | `fp16`, `fp8`, or `auto` (fp8 under 7 GB) |
| `INPROC_EMPTY_CACHE_BEFORE_DECODE` | `0` | free fragmented VRAM just before the decode |
| `INPROC_MAX_SIZE` | `1024` | largest square the model will generate |
| `INPROC_MAX_DENOISE` | `0.9` | ceiling for the room slider |
| `INPROC_WARMUP_SIZE` | `768` | one throwaway generation at startup |
| `INPROC_DRY_RUN` | `0` | serve the whole contract with no model (CI) |
| `INPROC_NO_PRELOAD` | `0` | load on the first generation instead of at startup |

Capacity, all of it reachable by an unauthenticated client and therefore
enforced before anything is allocated:

| variable | default | meaning |
| --- | --- | --- |
| `ROOM_CREATE_PER_MIN` | `10` | room creations per client address, token bucket |
| `UNJOINED_ROOM_TTL_MS` | `300000` | how long a room nobody joined holds its slot |
| `MAX_ROOM_SOCKETS` | `16` | sockets in one room (every join rebroadcasts the member list) |
| `MAX_TOTAL_SOCKETS` | `256` | sockets in the process |
| `MAX_ROOM_POINTS` | `2000000` | aggregate committed points in one room (max 5,000,000) |
| `MAX_ROOM_SNAPSHOT_BYTES` | `6291456` | what a room's log may serialise to (max 7 MiB) |
| `ROOM_IDLE_MS` | `1800000` | how long an empty room keeps its rasters |

`MAX_ROOM_SNAPSHOT_BYTES` is the one that matters, because it is the limit that
decides whether the room stays joinable: a log that serialises past the 8 MiB
outbound frame cap is a room every later join and every reconnect must refuse.
It is *measured* per committed stroke, not estimated - points are clamped but
not rounded, so a client sending `1.2345678901234567` serialises to nearly
three times what a well-behaved one does, and a per-point constant large enough
to bound that would shrink an ordinary session to a fraction of the budget.
`MAX_ROOM_POINTS` bounds process memory alongside it. Past either, `stroke_end`
is cancelled with reason `quota` rather than committed, and deleting or
clearing a layer gives the budget back.

Both budgets count strokes still being drawn, not only committed ones: four
50,000-point strokes each from sixteen members is 3.2 million point dicts that
a commit-time check would never see, so points are charged as they arrive and
released when the stroke ends, is cancelled, is abandoned, or its author or
layer goes away.

Upload bodies have their own admission - one per address, four at a time,
48 MiB in flight - taken before a byte is read, because the body is in memory
before the room limiter or the image store gets a say.

Every path that can bring a room into existence - the POST, an upload to an
unknown id, and a link nobody has opened yet - goes through one rate-limited
`create_named`; `get` is the lookup that creates nothing. Socket capacity is
one synchronous reservation taken before the handshake's first await, so
concurrent connections cannot all pass the same check. A reconnect does not
take a second slot and does not skip the check either: it gets a ticket for the
exact lease held by the socket it replaces, and that ticket owns nothing until
the join installs the replacement. Everything between the reservation and the
join can fail, and until it succeeds the original socket keeps its slot.

There is exactly one counted slot per identity across a reconnect, and it is
never in flight: whichever side lets go first hands it to the other. An
original that releases while a ticket is outstanding transfers the slot -
and with it the room's pin - rather than decrementing, so no unrelated
connection can be admitted into a vacancy that is already spoken for and the
sweeper cannot delete the room before the ticket commits. A ticket that
commits inherits the slot; one that is abandoned returns whatever it was
holding. No path increments a counter without a check having been made for it. A room is never swept while a
handshake holds a slot in it.

One more thing is process-wide rather than per-room: **generation admission**.
Every room shares one FIFO slot, acquired *before* rasterising, so N active
rooms cannot allocate N supersampled rasters and then queue for a GPU that
serves one at a time. The watchdog starts at admission, never before it - a run
that timed out while queueing would retry, turning one backlog into two.

Each falls back to the `STREAM_*` name of the same setting, so an existing
`.env` written for the worker keeps working.

`/healthz` reports `{ok, backend, rooms}` as before and, for a resident
backend, the fields the tooling used to fetch from the worker's own `/healthz`
(`model`, `steps`, `guidance`, `vae`, `lora`, `max_size`, `max_denoise`,
`warm`, `busy`, `memory`), so there is one place to look.

## 8 GB cards

An 8 GB card holds this model with nothing to spare: 5.42 GB of weights
resident, 7.05 GB peak, and the desktop already holding ~0.6 GB. What ran out
first was not the UNet but the **VAE decode**, which allocates one buffer for
the whole image - and when it does not fit, Windows pages it rather than
failing, so a 0.28 s decode on a 4090 became 2.3-3.6 s on a 3070.

Two settings address it, and only one of them is the answer.

**Tile the decode** (`INPROC_VAE_TILE_SIZE`, `auto` = 256 px below 12 GB).
`enable_tiling()` was already on and was doing nothing: diffusers only tiles an
image *larger* than `tile_sample_min_size`, whose default is the VAE's own
sample size - 1024 for SDXL - so neither 768 nor 1024 was ever tiled. Decoding
in 256 px pieces gives every tile activations that fit.

**Store the UNet in fp8** (`INPROC_UNET_STORAGE`, `auto` = fp8 below 7 GB).
The conversion happens while the model is still in system RAM, before anything
is moved to the card, and the text encoders are never sent there at all when
they are being offloaded - a card small enough to want fp8 cannot hold the fp16
UNet on the way to fp8. A conversion that fails part-way is fatal rather than a
fallback: diffusers converts module by module with no rollback, so half the
UNet would be fp8 while the loader called it fp16 and let the LoRA fuse into
quantised weights. diffusers' layerwise casting keeps the weights as
`float8_e4m3fn` and casts each module up to fp16 as it runs: 5.42 GB resident becomes 2.98, and 0.1 GB
free becomes 3.0. It is not free - the casting costs UNet time, and the LoRA
can no longer be *fused* (fusing writes the merged weights back into the
parameter, which fp8 would quantise, and `unfuse_lora` would then have nothing
exact to take back out), so the adapter stays attached and pays its per-step
tax as well: 0.68 s of UNet at 768 becomes 1.9-2.1 s.

Measured on a 3070 (8 GB), through the server with `pnpm latency`, stroke_end
to pixels on screen, median of 4 edits at `fast`/768 and 2 at `quality`/1024:

| configuration | fast 768 | quality 1024 | resident | free |
| --- | --- | --- | --- | --- |
| fp16, untiled (what this used to do) | 6.66 s | 34.8 s | 5.42 GB | 0.10 GB |
| **fp16 + 256 px tiles (the default now)** | **2.17 s** | **9.44 s** | 5.42 GB | 0.10 GB |
| fp8 + 256 px tiles | 3.41 s | 9.77 s | 2.98 GB | 3.02 GB |
| fp8, untiled | 4.39 s* | 16.0 s* | 2.98 GB | 2.94 GB |

\* the fp8-untiled row is from `scripts/`-style direct measurement rather than
through the server; the others are `pnpm latency` medians. The untiled fp16
row is also the least stable of them: the same server gave 6.7 s cold and
anything from 1.9 s to 4.5 s afterwards, which is what paging looks like.

Stage breakdown at 768 (`/healthz` `last_timings`, ms):

| configuration | unet | vae_encode | vae_decode | total |
| --- | --- | --- | --- | --- |
| fp16, untiled | 669 | 211 | 372-3558 | 1299-6025 |
| fp16 + tiles | 694 | 237 | 569 | 1535 |
| fp8 + tiles | 1842 | 229 | 640 | 2744 |

So **the tile is the fix and fp8 is the fallback**: an 8 GB card keeps the
fused fp16 UNet and is quicker on both profiles, and fp8 exists for a card that
cannot hold the fp16 UNet at all, or for one that has to share. `auto` puts the
line at 7 GB for that reason rather than at the 12 GB the headroom argument
alone would suggest.

Quality, fp8 against fp16 with everything else held equal, same seed, prompt
and input: **30.7 dB PSNR at 768 and 34.5 dB at 1024** - above the 30 dB this
app needs. Tiling changes the picture more than fp8 does (25.7 dB / 28.4 dB
against an untiled decode) because the tiles are blended rather than exact, but
side by side there are no visible seams: the differences are in fine detail, in
an image the model is about to reinterpret again anyway.

Two other things were tried and are not defaults. `torch.cuda.empty_cache()`
immediately before the decode (`INPROC_EMPTY_CACHE_BEFORE_DECODE`) is worth
nothing once the decode is tiled (2.72 s against 2.66 s at 768, inside the
noise). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by `main.py`
before torch is imported, and torch reports it as unsupported on Windows - it
is there for the Linux pods, where it costs nothing and helps fragmentation.

## Where an edit's time goes

`LOG_LEVEL=DEBUG` makes every run print its stages, and `/healthz` carries the
last generation's `last_timings` so a slow edit can be explained without
turning logging on first:

```
[brushjam.ai] run r4 768px: mask 0 render 50 backend 35 apply 41 = 128 ms
[brushjam.ai.inproc] backend <id>: decode_in 3 queue 25 pipeline N encode_out 4 ms
```

`render` is the AI input (raster + PNG), `apply` is decode + upsample +
composite + the PNG the client downloads, `backend` is the whole generate call.
Measured with `INPROC_DRY_RUN=1` at 768 on this box, everything the Python
server does around the model costs **110-130 ms** per edit, and a real
photographic result adds at most ~40 ms of PNG encoding on top (measured: a
768 RGB encode is 20 ms at level 6, a 1024 RGBA one 36 ms). The gap between
`fast`/768 here (~3.7 s of server pipeline) and the dedicated worker process
(~1.4 s of generation) is therefore inside the GPU call, not in the plumbing.

Four things keep the plumbing off the critical path:

- **Layer rasters are kept between generations.** The AI input used to be
  rendered from the whole stroke log every time, which is free at a few hundred
  strokes and most of a second at a few thousand: a 30-minute soak (3 users,
  mock backend) watched throughput fall from 41 generations a minute to 11, and
  the reported `latencyMs` median reach 2.5 s with a backend that returns
  instantly. Strokes are composited one at a time onto a layer's buffer in log
  order, so the state after stroke *k* is a prefix of the state after stroke
  *n>k*: `LayerCacheStore` keeps that float32 buffer per (room, layer) and draws
  only what arrived since. Continuing the buffer is the same sequence of
  operations on the same bytes, so it is not an approximation -
  `tests/test_incremental.py` compares incremental against from-scratch renders
  after random sequences of pen, noise and eraser strokes with alpha, undo,
  clear and layer moves, and requires them to be byte-identical.

  A cached prefix is validated by the committed stroke *records* it consumed -
  the objects themselves, not their ids: the layer's current list must start
  with the same objects, so an undo inside the prefix, a cleared layer or an
  expired stroke rebuilds that layer and nothing else. Ids were the first
  version of this check and were wrong, because `clear_layer` frees the ids it
  removes and the same id could come back carrying a different drawing
  (review 3, finding 2). A committed stroke is appended once and never
  mutated, so identity is content. `clear_layer` also bumps a per-layer
  generation counter, which a render started before the clear fails on when it
  tries to install its result. Those counters are retired as soon as they
  protect nothing - no cached raster, no render in flight - so room and layer
  churn does not leave a dictionary growing behind the byte budget. Layer
  opacity and visibility are applied on the way into the canvas and cost
  nothing; a layer *move* does rebuild, because at a fractional offset every
  stroke is rasterised at a different sub-pixel phase and shifting finished
  pixels would not produce the same image.

  Measured with `scripts/bench_render.py` at 4,000 strokes on a 1024 canvas:
  **2,541 ms from scratch, 97 ms incrementally** (20 new strokes per render),
  and the incremental figure does not grow with the log. The price is 16.8 MB
  per cached draw layer, bounded by an LRU across all rooms
  (`LAYER_CACHE_BUDGET_BYTES`, 16 layers) and dropped when a room is evicted or
  has been empty for three minutes.

- **No PNG codec on the event loop.** The input render and the composite were
  already in threads; the backend's input decode and output encode are too.
  They are deliberately *not* on the GPU thread - that thread is the device
  lock, and CPU work there delays the next request's diffusion for nothing.
- **The AI input and mask are encoded at `compress_level=1`.** Those bytes
  never leave the process. Level 6 costs ~2x the CPU for a file nobody
  transmits; the composited canvas a client downloads stays at level 6.
- **The full mask is built once per size.** In full-canvas mode it is an opaque
  square, identical for every run.

## The saved history, and taking it away

Every accepted `ai_result` is written to `HISTORY_DIR/<roomId>/` and served
back regardless of whether that room still exists - it outlives the room, the
eviction sweep and the process. One entry is up to three files plus a shared
`counter`:

| file | what it is |
| --- | --- |
| `<n>.jpg` | the AI canvas after that result was composited |
| `<n>.in.jpg` | the human canvas that generation was made *from* - optional |
| `<n>.json` | the settings that produced it, and the commit point of the entry |

`<n>.in.jpg` is the raster the pipeline was handed, re-encoded from those exact
PNG bytes rather than rendered a second time: a second render would capture
whatever has been drawn since, and the pair would no longer be a before and an
after. It is stashed on the accepted path only, so a run discarded as stale
cannot leave its input behind for the next result to claim, and it is written
*before* the pair, because the pair is what defines the entry and the optional
file must never be the thing that is half installed. It is stored at the
generation size (`aiResolution`), not the canvas size, for the same reason: it
is what the model saw.

The entry is still defined by its JSON, so an entry with no input - one written
before this existed, or one whose encode failed - is a perfectly good entry.
Its bytes are charged to the entry, eviction removes all three files, and the
listing carries `inputUrl` only when the file is there.

| route | what it returns |
| --- | --- |
| `GET /rooms/{id}/history?limit=` | newest first, JSON; each entry has `url` and maybe `inputUrl` |
| `GET /rooms/{id}/history/{n}.jpg` | the AI result, `immutable` |
| `GET /rooms/{id}/history/{n}.in.jpg` | its input, `immutable` |
| `GET /rooms/{id}/history.zip` | every frame plus a `manifest.json` |
| `GET /rooms/{id}/history.avi?fps=&width=` | a Motion JPEG video of the whole room |

The zip is the archival form: the stored JPEGs byte for byte, `ZIP_STORED`
because deflating a JPEG costs the export's CPU again and wins a fraction of a
percent, named `draw_00000.jpg` / `gen_00000.jpg` by the entry's own number so a
name in the zip and a number in the gallery are the same thing. `manifest.json`
carries `roomId`, `canvasSize`, `exportedAt` and one `frames` entry per result
with its file names (`draw` is `null` when there is no input) and its settings.

The AVI is the form you can watch: one frame per entry, oldest first, the
drawing on the left and what the model made of it on the right, entries with no
input getting a white left half - the same background the AI input is
composited on. Default frame size is the stored one (two 1024 halves = 2048 x
1024); `width` (256-4096, the whole frame) scales both halves, `fps` is clamped
to 1-30 and defaults to 4. Both are clamped rather than refused: they are the
two knobs on a download link.

`src/brushjam/avi.py` writes the container by hand - `avih`, one `strl` with a
`strh`/`strf` whose `biCompression` is `MJPG`, a `movi` list of `00dc` chunks
padded to even lengths, and an `idx1` index - about a page of `struct` calls.
ffmpeg is used to *verify* it (`tests/test_export.py`, skipped when ffmpeg is
not on PATH: `ffprobe` must report an `mjpeg` stream of the right size, frame
count and rate, and `ffmpeg -f null -` must decode it silently) but is
deliberately not a runtime dependency, because an export that needs a binary
this box happens to have is a feature that works here and nowhere else. The
header is conservative for the sake of desktop players rather than of ffmpeg:
`dwTotalFrames`, `dwLength`, `dwMaxBytesPerSec` and a `dwSuggestedBufferSize`
big enough for the largest chunk are all patched in by `close`, and
`AVIF_HASINDEX` is set because there is an `idx1`. Plain AVI's 32-bit offsets
mean the writer refuses at ~1.5 GB rather than growing into OpenDML.

Both exports are built on a worker thread into a temp file under
`HISTORY_DIR/.exports/` and streamed from there with `Cache-Control: no-store`
and a `Content-Disposition` attachment name; a `BackgroundTask` deletes the
file afterwards, and a build that fails or is abandoned deletes it on the way
out. Nothing is held in memory: a room's history is hundreds of megabytes by
the budget's own definition. Both are guarded - one export per room and two in
the process, `429` with `Retry-After` past either - because an export reads and
re-encodes everything a room ever made and is an unauthenticated GET. A room
with nothing saved is `404`; more entries than `HISTORY_EXPORT_MAX_FRAMES`
(3000) is `413` on the video, and the zip has no such limit.

| variable | default | meaning |
| --- | --- | --- |
| `HISTORY_ENABLED` | `1` | `0` keeps nothing at all, and turns the exports off with it |
| `HISTORY_DIR` | `./data/history` | where the JPEGs and the temp exports go |
| `HISTORY_ROOM_MB` | `200` | one room's budget; oldest first past it |
| `HISTORY_TOTAL_MB` | `2000` | the whole store's budget |
| `HISTORY_EXPORT_MAX_FRAMES` | `3000` | most frames one video may have |

## Differences from the Node server

1. **Patch mode is not ported.** `AI_MODE=full` only; `AI_MODE=patch` refuses to
   boot with a message rather than pretending. The Node implementation stays in
   git history.
2. **Heartbeat is uvicorn's** (30 s ping, 60 s timeout = the same two misses)
   rather than a hand-rolled sweep, and `ws_max_size` enforces the 1 MiB frame.
3. **Antialiasing is not pixel-identical to the browser.** Shapes are drawn at
   4x and box filtered; the output is deterministic run to run, which is what
   the AI input actually needs.
4. **Messages are plain dicts with a hand-written validator**, not pydantic
   models, so the error strings a client sees are identical to the Node ones.
   `apps/server/scripts/export-fixtures.ts` writes fixtures from the real Node
   modules and `tests/test_fixtures.py` replays them - that is the parity proof.

## The parity fixtures are frozen

`apps/brushjam/fixtures/` was written by the Node server's `export-fixtures`
script, driving the **real** reference implementation: its
`validateClientMessage`, `applyClientMessage`, `snapshot`, `fnv1a`/`noiseRGB`
and `renderStrokes` through a canvas. 45 protocol samples, 27 noise vectors, a
29-step reducer trace (result *and* full snapshot after every step), 256 noise
placement cases at fractional layer offsets, and real canvas pixels for five
strokes.

That server has been deleted, so nothing can regenerate them. This is
deliberate: keeping them regenerable would have meant keeping the Node reducer
alive as a library forever, and the thing worth keeping is the record, not the
generator. They are no longer a parity oracle - they are the behaviour this
server was built to match, replayed in full by `tests/test_fixtures.py` on
every run. A change that breaks one is a change to the wire behaviour, and has
to be argued for rather than re-recorded.


## History: retiring the Node server

`apps/server` - the original TypeScript room server, 530 tests - was deleted
once this one had been through six rounds of adversarial review and the
cross-language fixtures showed the two agreeing byte for byte.

What went with it, and what replaced it:

| gone | replacement |
| --- | --- |
| the room server and its tests | this server, and `apps/brushjam/tests` |
| `export-fixtures` | nothing: the fixtures are frozen (above) |
| `quality-grid` (denoise sweep) | nothing. It called `generate()` directly with a fixed seed and a full-white mask, so it could not be a client of a running server. The reports in `docs/experiments/` stay as records |
| `runpod-smoke` | `pnpm smoke -- --url ...` against a server running `AI_BACKEND=runpod` |
| root scripts `dev:stream`, `export-fixtures` | - |

Two environment differences for anyone with an old `.env`:

- **`AI_MODE=patch` is gone.** This server implements full-canvas mode only and
  refuses to boot on `patch`, with a message saying so.
- `AI_CFG` and `AI_VAE_TILE` are still parsed and validated, but they only
  reach the ComfyUI backend. They do nothing for `inproc`.

`apps/stream-worker` was **kept**: the `stream` backend talks to it over HTTP,
which is how a model on another machine is reached, and the in-process backend
cannot do that by construction. Its pipeline is now a slightly older copy of
`apps/brushjam/src/brushjam/ai/pipeline.py` (it fuses the LoRA once at load, so
it serves `fast` only). The right fix, when someone needs it, is to have the
worker import the pipeline from `apps/brushjam` rather than keep its own.

## Review

Six rounds of adversarial review by Codex (gpt-5.6-sol) against the port, each
followed by fixes and tests:

| round | findings | what they were about |
| --- | --- | --- |
| 1 | 4 High, 5 Medium, 2 Low | room and socket exhaustion, unbounded stroke memory, rendering ahead of the GPU queue, resume not revoking the old socket; AI-canvas and LRU thread safety, a partially-applied locked-layer patch, noise hashing at fractional offsets; path containment and explicit JSON `null` |
| 2 | 5 High, 3 Medium, 1 Low | the round-1 limits were bypassable: uploads and WebSockets created rooms unthrottled, the threaded join dropped messages, resume left a mapping gap, caps were not atomic, the point quota still allowed unjoinable rooms |
| 3 | 2 High, 4 Medium | pending strokes bypassed both budgets, resume did not transfer its lease; cancellation released admission early, the sweeper could delete a room mid-handshake, buffered output could exceed its cap, upload bodies had no admission |
| 4 | 2 High | the lease moved before the join committed, and `stroke_end` ignored other users' pending points |
| 5 | 1 High | an original lease releasing mid-crossover opened a vacancy its own ticket was holding |
| 6 | none | GO |

The pattern is worth naming: every round after the first found a way *round*
the previous round's limit rather than a new kind of problem. A limit that
reads correctly and a limit that holds are different things, and the difference
was usually an `await` in the middle of a check.

Python tests went 198 -> 278 over those rounds. The Node suite (530) and the
cross-language fixtures were green throughout, which is what made it safe to
keep changing the reducer.

## Measurements

Measured 2026-09-05 during a GPU window with the local stream worker and the
Node server stopped, on the RTX 3070 8 GB this repo was built on. Server:
`HOST=0.0.0.0 PORT=8787 AI_BACKEND=inproc CANVAS_SIZE=1024`, checkpoint
`waiNSFWIllustrious_v150.safetensors` + `dmd2_sdxl_4step_lora_fp16.safetensors`,
fp16-fix VAE.

| what | command | result |
| --- | --- | --- |
| warm-up | server start, `INPROC_WARMUP_SIZE=768` | 16.5 s load + 11.5 s warm-up run = **28.0 s** to `warm: true` |
| one `fast` generation at 768 | browser, five-stroke sketch | **3.28 s** stroke end -> pixels |
| one `quality` generation at 1024 | browser, same room, profile switched live | **9.75 s** stroke end -> pixels |
| per-edit latency, `fast` / 768 | `latency -- --url http://127.0.0.1:8787 --n 5` | to pixels median **4.12 s** (3.95-4.23), server pipeline median 3.72 s |
| per-edit latency, `quality` / 1024 | same, room pre-set to quality/1024 | to pixels median **8.75 s** (5.72-8.95), server pipeline median 8.86 s |
| 3 users for 1 minute | `playtest-sim -- --url http://127.0.0.1:8787 --users 3 --minutes 1` | **PASS** - 118 strokes, 18 AI results (16.7/min), gap median 3.08 s / p90 3.66 s, server latency median 2.95 s, convergence OK, no errors |
| VRAM | `/healthz` `memory` after the above | allocated 5.6 GB, reserved 7.1 GB, **peak allocated 6.97 GB** of 8.0 GB |

The profile switch is a live LoRA detach plus scheduler swap inside the one
resident model: the log shows `profile -> quality (lora off)` and `/healthz`
flips to `steps: 14, guidance: 5.5, profile: "quality"` with no reload and no
change in resident memory. `negative_prompt_active` correctly reports
`{fast: false, quality: true}`, and the client shows the negative-prompt box
only under `quality`.

The 8 GB card is the ceiling here: `device_free_gb` sits at 0.0-0.1 with the
model resident at 1024, which is why `max_size` is capped at 1024 and why the
first `quality` run in a batch is the slow one (allocator growth), later runs
settle.

For comparison, the same box measured through the Node server: the stream
worker at 768 end-to-end ~2.4 s (`docs/experiments/2026-09-05-stream/REPORT.md`)
and ComfyUI 14-step at 1024 ~10.3 s
(`docs/experiments/2026-09-05-comfyui/REPORT.md`). In-process `fast` is ~0.9 s
slower per edit than the dedicated worker process (no HTTP hop, but the render
and PNG encode now share the event loop with the room), and `quality` at 1024
is ~1.5 s faster than ComfyUI for the same 14 steps.
