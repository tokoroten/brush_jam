# Brush Jam stream worker

A small, model-resident image-to-image HTTP service. It loads the local
Illustrious/SDXL checkpoint **once**, fuses a 4-step distillation LoRA into the
UNet, and then answers `POST /generate` in a few hundred milliseconds instead of
the ~8–15 s a cold ComfyUI graph costs.

It speaks the same request contract as the Brush Jam AI backends
(`docs/MVP_PLAN.md` §6), so the room server can talk to it through
`apps/brushjam/src/brushjam/ai/backends/stream.py`.

Why diffusers and not StreamDiffusion: see `docs/STREAM_WORKER.md`.

## Install

Requires `uv`, an NVIDIA GPU with CUDA 12.x drivers, and an SDXL checkpoint.
`STREAM_CHECKPOINT` says where it is; there is no default, because the file is
6-7 GB and lives wherever you keep such things.

```bash
cd apps/stream-worker
uv sync --extra dev --python 3.10
```

`torch` comes from the PyTorch CUDA 12.4 index (pinned in `pyproject.toml`);
plain PyPI would install the CPU build on Windows.

The LCM LoRA is downloaded on first run into `STREAM_LORA_DIR` (`./models/loras`)
(`lcm-lora-sdxl.safetensors`, ~394 MB) so ComfyUI can use the exact same file.
`HF_TOKEN` from the repo-root `.env` is used if the download needs auth.

## Run

```bash
cd apps/stream-worker
uv run stream-worker              # http://127.0.0.1:8790
# or
uv run python -m stream_worker
```

Startup loads the checkpoint and runs one warm-up generation; `GET /healthz`
reports `warm: false` until that finishes (roughly 40–70 s from a cold file
cache, ~25 s warm).

### Endpoints

`GET /healthz`

```json
{ "ok": true, "backend": "diffusers-sdxl-lcm", "model": "waiNSFWIllustrious_v150.safetensors+dmd2_sdxl_4step_lora_fp16.safetensors",
  "size": 768, "max_size": 1024, "steps": 4, "guidance": 1.0,
  "lora": "dmd2", "vae": "fp16fix", "negative_prompt_active": false, "max_denoise": 0.9,
  "warm": true, "loaded": true, "busy": false, "current_request_id": null, "error": null,
  "memory": { "allocated_gb": 5.05, "reserved_gb": 5.32, "max_allocated_gb": 6.59, "device_free_gb": 1.21, "device_total_gb": 8.0 } }
```

`memory` is the VRAM accounting: `allocated_gb` is live tensors, `reserved_gb`
is what PyTorch's caching allocator holds, and `device_free_gb` is what the
driver has left. If `device_free_gb` reaches 0 the next request will spill into
shared system memory and get several times slower - see the troubleshooting
notes below.

`POST /unload` — drop the model and free its VRAM without exiting (the process
stays up and `/healthz` reports `loaded: false`); `POST /load` puts it back.
These exist so this worker and ComfyUI can hand the GPU over on a card that
cannot hold both — see `docs/STREAM_WORKER.md` §3.1. A `/generate` that arrives
while unloaded reloads transparently, but pays the ~90 s cold start.

`POST /generate`

```jsonc
{
  "image_b64": "<PNG base64>",       // required, the human drawing
  "mask_b64":  "<PNG base64>",       // optional, L/RGB, white = regenerate
  "prompt": "anime style, fantasy town, vibrant colors",
  "negative_prompt": "lowres, bad anatomy, ...",
  "denoise": 0.8,                    // alias: "strength"
  "steps": 4,                        // denoising steps actually performed
  "seed": 12345,
  "width": 1024, "height": 1024,     // alias: "size" for squares
  "request_id": "optional-caller-id",// for /cancel; generated if omitted
  "queue": false,                    // true = wait for the GPU instead of 409
  "strict_steps": false              // true = 400 instead of running fewer steps
}
```

→ `{ "image_b64": "<PNG base64>", "width": 1024, "height": 1024, "request_id": "…",
     "steps": 4, "timings": { … } }`

The PNG is always **exactly** `width x height`; the worker returns 500 rather
than a differently-sized image.

`timings`: `wait_ms` (queued behind another request), `prompt_ms` +
`prompt_cached` (1.0 on an embedding-cache hit), `diffusion_ms` and its
breakdown `unet_ms` / `vae_encode_ms` / `vae_decode_ms`, `steps_run`,
`composite_ms`, `png_encode_ms`, `empty_cache_ms`, `total_ms`. `unet_ms` is the
remainder after the two VAE spans, so it also carries scheduler overhead - treat
it as an upper bound rather than a pure UNet number.

Status codes: 200 ok, 400 bad image or out-of-range size, 409 busy (unless
`queue: true`; carries `retry-after`), 499 cancelled, 500 generation failed,
503 model failed to load.

`POST /cancel`

```jsonc
{ "request_id": "the-id-you-sent" }
// -> { "ok": true, "request_id": "...", "state": "running" | "pending" }
```

Stops a running job at its **next diffusion step** (0.1-1 s depending on size),
or prevents a queued one from starting. Closing the HTTP connection is *not*
enough: the diffusion loop runs on a background thread and would finish anyway,
holding the GPU. Cancelling an unknown or already-finished id is not an error.

Notes on the contract:

- `steps` means *steps actually run*, and the response echoes the effective
  count rather than the requested one. The worker picks the LCM start timestep
  from `denoise` directly and runs exactly `steps` from there
  (`lcm_timesteps_for_strength`), but at low denoise there may not be that many
  distinct timesteps left: **denoise 0.2 with 20 steps runs 10**. Send
  `strict_steps: true` to get a `400` naming both numbers instead. Off by
  default; a clamp is logged at INFO. 4 steps needs only denoise >= 0.08, so the
  playtest path never approaches this.
- The mask is applied by compositing the output over the input
  (`out = in*(1-m) + gen*m`), so a black mask returns the drawing untouched.
  This is what keeps the ComfyUI backend's semantics.
- One generation runs at a time (single GPU). Extra requests wait on a lock and
  their wait shows up as `wait_ms`; latest-wins is the caller's job.

## Environment variables

| var | default | meaning |
| --- | --- | --- |
| `STREAM_HOST` | `127.0.0.1` | bind address |
| `STREAM_PORT` | `8790` | port |
| `STREAM_CHECKPOINT` | *(required)* | single-file SDXL checkpoint |
| `STREAM_LORA_DIR` | `./models/loras` | where the LoRA is downloaded to |
| `STREAM_LORA` | `dmd2` | `dmd2` or `lcm`. DMD2 is 1384 ms vs LCM's 1798 ms at 768, reinterprets a denoise step earlier, and retains line weight better (`docs/experiments/2026-09-05-stream/REPORT.md` section 8). |
| `STREAM_STEPS` | `4` | default steps |
| `STREAM_GUIDANCE` | `1.0` | CFG. `1.0` means CFG is off: ~22% faster end-to-end (the UNet is about half a 768 request) and free with `dmd2`, which is guidance-distilled. **`negative_prompt` has no effect at 1.0** — `/healthz` reports `negative_prompt_active: false` so the UI can say so. Set `1.5` to turn it back on (+27% latency). With `STREAM_LORA=lcm`, CFG 1.0 also makes line art fade badly at denoise 0.8+. |
| `STREAM_MAX_DENOISE` | `0.9` | highest denoise honoured; higher requests are clamped, not refused. Published in `/healthz` as `max_denoise` so the server can cap its slider to a value the backend will actually act on. |
| `STREAM_WARMUP_SIZE` | `768` | size of the startup warm-up run; `0` disables |
| `STREAM_MAX_SIZE` | `1024` | requests above this are rejected with 400 |
| `STREAM_OFFLOAD_TEXT_ENCODERS` | `1` | park the two CLIP encoders in system RAM between requests (saves ~1.8 GB VRAM) |
| `STREAM_VAE` | `fp16fix` | which VAE to run: `fp16fix` (`madebyollin/sdxl-vae-fp16-fix`, same weights rescaled so fp16 does not overflow), `taesd` (`madebyollin/taesdxl`, distilled, ~1.6x faster; inspected at 1:1 - crisper on line art, visibly flattens continuous tone, see `docs/experiments/2026-09-05-stream/REPORT.md` section 7), or `checkpoint` (the one baked into the checkpoint, which forces an fp32 upcast on every call). |
| `STREAM_VAE_TILING` | `1` | tiled/sliced VAE |
| `STREAM_EMPTY_CACHE` | `1` | return the allocator's cache after every generation. **Leave this on**: without it PyTorch reserves ~7.5 GB against 5.05 GB of live tensors, the card reports 0 bytes free, and 768²/1024² spill to shared memory and get 2–8× slower (`docs/STREAM_WORKER.md` §4.3). |
| `STREAM_EMBED_CACHE` | `16` | prompt-embedding cache entries |
| `STREAM_QUALITY_SUFFIX` | `, masterpiece, best quality` | appended to every prompt (matches the ComfyUI backend) |
| `STREAM_DRY_RUN` | `0` | serve the contract without a GPU (echoes the input); for CI |
| `STREAM_LOG_LEVEL` | `INFO` | logging level |
| `HF_TOKEN` | — | only used if the LoRA has to be downloaded |
| `PYTORCH_CUDA_ALLOC_CONF` | — | set to `expandable_segments:True` on 8 GB cards; it reduces allocator fragmentation and is part of the §4.3 fix |

## VRAM

| what | fp16 |
| --- | --- |
| SDXL UNet (resident) | ~5.0 GB |
| VAE (resident, tiled) | ~0.2 GB |
| text encoders | ~1.8 GB, **on CPU** by default, paged in only on an embedding-cache miss |
| activations at 1024², 4 steps, CFG on | ~0.6–1.0 GB |

Practical requirement on this box: **~5.5–6.0 GB free VRAM**. ComfyUI holds
3–4 GB with a checkpoint loaded, so on this 8 GB card the two **cannot both be
resident**: they thrash instead of failing, and both get several times slower
(`docs/STREAM_WORKER.md` §3.1). Free ComfyUI first — this unloads its models but
leaves ComfyUI running, and only takes effect once its queue is empty:

```bash
curl -X POST http://127.0.0.1:8188/free -H "content-type: application/json" \
     -d '{"unload_models":true,"free_memory":true}'
```

## Benchmark

```bash
# one configuration, against a worker you already started
uv run python scripts/bench.py --sizes 512 768 1024 --runs 5 --mask

# every VAE x every size, starting and stopping its own workers
# (nothing else may be listening on the port, and free ComfyUI first)
uv run python scripts/bench_vae.py

# rehearse that harness without a GPU first - proves the orchestration,
# the tables and the cleanup before you spend a GPU window on it
uv run python scripts/bench_vae.py --dry-run --runs 2 --warmups 1

# sweep denoise to pick AI_DENOISE
uv run python scripts/quality_probe.py 512
```

Measured on this machine (RTX 3070 8 GB, ComfyUI stopped, warm, 4 steps,
denoise 0.55, 5 runs each):

| size | wall median |
| --- | --- |
| 512² | 1682 ms |
| 768² | 6025 ms |
| 1024² | 15234 ms |

Cold start ~35-41 s (load + one warm-up run). Full analysis, including the
allocator bug that made 768²/1024² 2-8× slower before it was fixed and the
VAE fp32 upcast that is still costing ~1 s per request, is in
`docs/STREAM_WORKER.md` §4.

## Tests

```bash
uv run pytest            # pure-python tests, no GPU (uses STREAM_DRY_RUN)
```

## Troubleshooting

- **Everything takes several times longer than the table above.** Almost always
  VRAM, and it shows up as slowness rather than `CUDA out of memory`: on Windows
  an allocation that does not fit silently spills into shared system memory over
  PCIe. Check `GET /healthz` → `memory.device_free_gb`; if it is at or near 0,
  either something else is holding VRAM (`nvidia-smi`, and the ComfyUI `/free`
  call above) or `STREAM_EMPTY_CACHE` has been turned off.
- **`checkpoint not found`.** Set `STREAM_CHECKPOINT`.
- **First request is slow.** The prompt-embedding cache is cold and the text
  encoders have to be paged to the GPU (~0.6 s). Repeat prompts are free.
- **The output ignores `negative_prompt`.** `STREAM_GUIDANCE` is ≤ 1.0.
- **Result looks washed out / plastic.** LCM at 4 steps is weak on
  Illustrious-class models; try `STREAM_LORA=dmd2`, or raise steps to 6–8.
- **`from_single_file` tries to reach huggingface.co.** diffusers fetches the
  small SDXL *config* JSONs (not weights) on the first load; they are cached
  afterwards. Fully offline first-run is not supported.
- **Port already in use.** `STREAM_PORT=8791 uv run stream-worker`.
