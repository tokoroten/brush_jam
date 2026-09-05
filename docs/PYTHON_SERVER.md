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
| `src/brushjam/raster.py` | Pillow + numpy rendering of the AI input and the AI canvas |
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

The LoRA is **attached, not fused**. Fusing was right for a fast-only worker -
no per-step PEFT overhead, no second copy of the deltas - but it cannot be
undone cheaply, which is exactly what made a quality profile impossible there.
Switching is `enable_lora()` + `set_adapters(["fast"])` or `disable_lora()`
plus a scheduler swap: milliseconds, which is what makes a per-request choice
reasonable at all.

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
| `INPROC_MAX_SIZE` | `1024` | largest square the model will generate |
| `INPROC_MAX_DENOISE` | `0.9` | ceiling for the room slider |
| `INPROC_WARMUP_SIZE` | `768` | one throwaway generation at startup |
| `INPROC_DRY_RUN` | `0` | serve the whole contract with no model (CI) |
| `INPROC_NO_PRELOAD` | `0` | load on the first generation instead of at startup |

Each falls back to the `STREAM_*` name of the same setting, so an existing
`.env` written for the worker keeps working.

`/healthz` reports `{ok, backend, rooms}` as before and, for a resident
backend, the fields the tooling used to fetch from the worker's own `/healthz`
(`model`, `steps`, `guidance`, `vae`, `lora`, `max_size`, `max_denoise`,
`warm`, `busy`, `memory`), so there is one place to look.

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

## Measurements

**Not yet measured.** The numbers below are recorded during a GPU window with
the local stream worker stopped, on the RTX 3070 8 GB this repo was built on.

| what | command | result |
| --- | --- | --- |
| warm-up | server start, `INPROC_WARMUP_SIZE=768` | _pending_ |
| one `fast` generation at 768 | browser | _pending_ |
| one `quality` generation at 1024 | browser | _pending_ |
| per-edit latency | `pnpm --filter @brushjam/server latency -- --url … --n 5` | _pending_ |
| 3 users for 1 minute | `pnpm --filter @brushjam/server playtest-sim -- --url … --users 3 --minutes 1` | _pending_ |

For comparison, the same box measured through the Node server: the stream
worker at 768 end-to-end ~2.4 s (`docs/experiments/2026-09-05-stream/REPORT.md`)
and ComfyUI 14-step at 1024 ~10.3 s
(`docs/experiments/2026-09-05-comfyui/REPORT.md`).
