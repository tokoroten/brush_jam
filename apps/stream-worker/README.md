# Brush Jam stream worker

A small, model-resident image-to-image HTTP service. It loads the local
Illustrious/SDXL checkpoint **once**, fuses a 4-step distillation LoRA into the
UNet, and then answers `POST /generate` in a few hundred milliseconds instead of
the ~8–15 s a cold ComfyUI graph costs.

It speaks the same request contract as the Brush Jam AI backends
(`docs/MVP_PLAN.md` §6), so `apps/server` can talk to it through
`apps/server/src/ai/backends/stream.ts`.

Why diffusers and not StreamDiffusion: see `docs/STREAM_WORKER.md`.

## Install

Requires `uv`, an NVIDIA GPU with CUDA 12.x drivers, and the checkpoint at
`E:\ComfyUI\models\checkpoints\waiNSFWIllustrious_v150.safetensors`.

```bash
cd apps/stream-worker
uv sync --extra dev --python 3.10
```

`torch` comes from the PyTorch CUDA 12.4 index (pinned in `pyproject.toml`);
plain PyPI would install the CPU build on Windows.

The LCM LoRA is downloaded on first run into `E:\ComfyUI\models\loras`
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
{ "ok": true, "backend": "diffusers-sdxl-lcm", "model": "waiNSFWIllustrious_v150.safetensors+lcm-lora-sdxl.safetensors",
  "size": 768, "max_size": 1024, "steps": 4, "guidance": 1.5, "warm": true, "busy": false, "error": null }
```

`POST /generate`

```jsonc
{
  "image_b64": "<PNG base64>",       // required, the human drawing
  "mask_b64":  "<PNG base64>",       // optional, L/RGB, white = regenerate
  "prompt": "anime style, fantasy town, vibrant colors",
  "negative_prompt": "lowres, bad anatomy, ...",
  "denoise": 0.55,                   // alias: "strength"
  "steps": 4,                        // denoising steps actually performed
  "seed": 12345,
  "width": 1024, "height": 1024      // alias: "size" for squares
}
```

→ `{ "image_b64": "<PNG base64>", "width": 1024, "height": 1024, "timings": { … } }`

`timings` contains `wait_ms` (queued behind another request), `prompt_ms`,
`prompt_cached` (1.0 on an embedding-cache hit), `diffusion_ms`,
`composite_ms`, `encode_ms`, `total_ms`.

Notes on the contract:

- `steps` means *steps actually run*. diffusers' img2img would otherwise drop
  `steps * (1 - denoise)` of them, so the worker passes `ceil(steps/denoise)` to
  the scheduler.
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
| `STREAM_CHECKPOINT` | `E:\ComfyUI\models\checkpoints\waiNSFWIllustrious_v150.safetensors` | single-file SDXL checkpoint |
| `STREAM_LORA_DIR` | `E:\ComfyUI\models\loras` | where the LoRA is kept (shared with ComfyUI) |
| `STREAM_LORA` | `lcm` | `lcm` or `dmd2` (4-step DMD2 fallback) |
| `STREAM_STEPS` | `4` | default steps |
| `STREAM_GUIDANCE` | `1.5` | CFG. **`1.0` disables CFG and roughly halves UNet time, but then `negative_prompt` is ignored.** |
| `STREAM_WARMUP_SIZE` | `768` | size of the startup warm-up run; `0` disables |
| `STREAM_MAX_SIZE` | `1024` | requests above this are rejected with 400 |
| `STREAM_OFFLOAD_TEXT_ENCODERS` | `1` | park the two CLIP encoders in system RAM between requests (saves ~1.8 GB VRAM) |
| `STREAM_VAE_TILING` | `1` | tiled/sliced VAE (needed for 1024 on 8 GB) |
| `STREAM_EMBED_CACHE` | `16` | prompt-embedding cache entries |
| `STREAM_QUALITY_SUFFIX` | `, masterpiece, best quality` | appended to every prompt (matches the ComfyUI backend) |
| `STREAM_DRY_RUN` | `0` | serve the contract without a GPU (echoes the input); for CI |
| `STREAM_LOG_LEVEL` | `INFO` | logging level |
| `HF_TOKEN` | — | only used if the LoRA has to be downloaded |

## VRAM

| what | fp16 |
| --- | --- |
| SDXL UNet (resident) | ~5.0 GB |
| VAE (resident, tiled) | ~0.2 GB |
| text encoders | ~1.8 GB, **on CPU** by default, paged in only on an embedding-cache miss |
| activations at 1024², 4 steps, CFG on | ~0.6–1.0 GB |

Practical requirement on this box: **~5.5–6.0 GB free VRAM**. ComfyUI holds
3–4 GB while idle, so free it first (this unloads models but leaves ComfyUI
running):

```bash
curl -X POST http://127.0.0.1:8188/free -H "content-type: application/json" \
     -d '{"unload_models":true,"free_memory":true}'
```

## Benchmark

```bash
uv run python scripts/make_sample.py --size 1024
uv run python scripts/bench.py --sizes 512 768 1024 --runs 5
```

Numbers for this machine (RTX 3070 8 GB) are in `docs/STREAM_WORKER.md`.

## Tests

```bash
uv run pytest            # pure-python tests, no GPU (uses STREAM_DRY_RUN)
```

## Troubleshooting

- **`CUDA out of memory` / everything takes 5+ s.** Something else is holding
  VRAM. `nvidia-smi` and the ComfyUI `/free` call above. On Windows an
  allocation that does not fit silently spills into shared system memory and
  gets ~10× slower rather than failing — slow is the usual symptom, not OOM.
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
