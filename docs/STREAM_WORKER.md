# Stream worker — decision record, benchmarks, and wiring instructions

Scope: `apps/stream-worker` (Python) plus `apps/server/src/ai/backends/stream.ts`.
Written for the agent that owns `apps/server` / `apps/web`: **section 6 is the
only part you need to act on.**

Hardware this was measured on: Windows 11, RTX 3070 8 GB, driver 591.86,
CUDA 12.4 wheels, ComfyUI 0.28 running alongside on :8188.

---

## 1. Why a stream worker at all

The ComfyUI backend re-walks a graph for every request. Even warm it costs
~8–15 s at 1024²/14 steps on this GPU (`README.md` "latency"), which is fine for
a playtest but far from the "AI reacts while you draw" feeling
`BRUSHJAM_CONTEXT.md` §11 is aiming at.

Everything expensive in that number is fixed cost we can pay once: model load,
prompt encoding, sampler setup. A resident process that holds the model in VRAM
and runs 4 LCM steps answers the *same* JSON contract in well under a second.

## 2. StreamDiffusion evaluation (step 1 of the brief)

Both candidates were actually installed and run on this machine, not just read.

### 2.1 `cumulo-autumn/StreamDiffusion` (upstream) — rejected

- `src/streamdiffusion/pipeline.py` imports `StableDiffusionPipeline` and
  nothing else; `grep -rn "XL" src/` returns **zero** hits, and the README
  never mentions SDXL. It is an SD1.5-only pipeline.
- Dependencies are frozen at `diffusers==0.24.0` / `onnx==1.15` / torch ~2.1,
  which is 2023-era and conflicts with anything current.

Our checkpoint is SDXL/Illustrious, so upstream is out on capability grounds
before portability even matters.

### 2.2 `livepeer/StreamDiffusion` (fork) — works, but rejected for this job

This fork genuinely does support SDXL (`model_detection.py`, dual text encoders,
`taesdxl`, SDXL micro-conditioning). It **installed and ran on Windows 8 GB**,
producing real SDXL img2img frames from the local Illustrious checkpoint. So the
rejection is not "it doesn't work" — it is a cost/benefit call. What was
measured:

| what | result |
| --- | --- |
| `uv pip install -e ./sd_livepeer` (py3.10, torch 2.6+cu124) | succeeded |
| `import streamdiffusion` | **failed twice** — `cv2` and `controlnet_aux` are imported unconditionally by `preprocessing/processors/__init__.py` but are not in `install_requires` |
| loading `waiNSFWIllustrious_v150.safetensors` | **failed**: `TypeError: StableDiffusionPipeline.encode_prompt() got an unexpected keyword argument 'prompt_2'` |
| loading the same file hard-linked as `sdxl_illustrious_eval_tmp.safetensors` | **succeeded** |
| 512², 3 t-indices, `acceleration="none"`, taesdxl | **5.6–6.3 s per frame**, peak 6.74 GB allocated |

The load failure is a real bug, not a mistake in how it was driven:
`wrapper.py:1110-1125` decides SDXL-vs-SD1.5 by **substring-matching the file
path** for `sdxl`/`xl`/`1024`. `waiNSFWIllustrious_v150.safetensors` contains
none of those, so it builds a `StableDiffusionPipeline` around SDXL weights and
dies at the first `encode_prompt`. Renaming the file is the workaround.

The 5.6–6.3 s is the decisive number. It is not the fork being slow per se: the
wrapper keeps both text encoders resident, so peak allocation is 6.74 GB against
~6.1 GB free, and Windows silently spills the overflow into shared system memory
— every UNet step then crawls. On this GPU the fork cannot hold SDXL *and* its
text encoders at once, and it has no option to offload them.

Other mismatches with the Brush Jam contract:

- **No mask.** StreamDiffusion has no mask input at all, so the soft-mask
  compositing would have to be bolted on outside it anyway.
- **No per-request denoise.** Noise level is `t_index_list`, fixed at
  construction. `denoise` is a first-class field of our contract (the room can
  change it) and would mean rebuilding the stream.
- **The wins do not apply yet.** Its real advantages — stream batching, R-CFG,
  frame-to-frame latent reuse, TensorRT — assume a continuous video stream. Our
  scheduler sends debounced, latest-wins, one-at-a-time requests at a size that
  changes per request; TensorRT is explicitly out of scope for this slice and
  builds fixed-resolution engines anyway.
- Dependency mass: opencv, controlnet-aux, timm, scikit-image, mediapipe,
  insightface, a forked `diffusers` pinned to a git SHA. That is a lot of
  surface for a service whose job is "call the UNet 4 times".

### 2.3 Decision

**Implement the loop directly with diffusers**:
`StableDiffusionXLImg2ImgPipeline.from_single_file` + LCM-LoRA (fused) +
`LCMScheduler`, fp16, 4 steps. ~400 lines total, stock dependencies, and it
gives us the two things the fork could not: a real mask and per-request denoise.

The one design idea worth stealing from StreamDiffusion is kept: **prompt
embeddings are cached**, keyed by `(prompt, negative, cfg)`.

Re-open this decision if we move to a dedicated GPU (≥12 GB), where the fork's
resident-text-encoder layout stops hurting and TensorRT becomes worthwhile.

## 3. What the worker does

`apps/stream-worker/src/stream_worker/`

- `config.py` — env-driven `Settings`.
- `pipeline.py` — model load, LoRA fuse, embedding cache, generation, mask
  composite.
- `app.py` — FastAPI: `POST /generate`, `GET /healthz`, one asyncio lock so the
  single GPU serves one request at a time.

Decisions worth knowing:

- **Single-file load.** The Illustrious `.safetensors` is loaded directly
  (`from_single_file`); no HF model repo, no second copy of the weights on disk.
  diffusers still fetches the small SDXL *config* JSONs from HF on first load.
- **LoRA is fused** (`fuse_lora()` then `unload_lora_weights()`): zero per-step
  PEFT overhead, no second copy of the deltas in VRAM. The trade-off is that the
  LoRA cannot be swapped at runtime — restart the worker to change
  `STREAM_LORA`.
- **Text encoders live on the CPU.** ~1.8 GB fp16 that only matters on an
  embedding-cache miss. This is what makes SDXL fit next to an idle-but-running
  ComfyUI on an 8 GB card, and it is precisely what the livepeer fork would not
  let us do.
- **`steps` means steps actually run.** diffusers' img2img keeps only the last
  `num_inference_steps * strength` timesteps, so the worker sends
  `ceil(steps/denoise)` to the scheduler. Asking for 4 steps at denoise 0.55
  really runs 4.
- **The mask is applied by compositing**, `out = in*(1-m) + gen*m`, with the
  server's already-feathered 8-bit mask. A fully black mask returns the drawing
  byte-identically, which keeps the `AIBackend` contract's meaning intact.
- **CFG is on by default** (`STREAM_GUIDANCE=1.5`) so `negative_prompt` is not
  silently ignored. It costs ~2× UNet time; `STREAM_GUIDANCE=1.0` halves the
  UNet cost and drops the negative prompt. Numbers for both are below.

## 4. Benchmarks

<!-- BENCHMARK -->

## 5. ComfyUI: the same LoRA, a 4-step Illustrious workflow

The worker downloads the LCM LoRA **into ComfyUI's own directory** so both use
one file:

```
E:\ComfyUI\models\loras\lcm-lora-sdxl.safetensors     (latent-consistency/lcm-lora-sdxl, ~394 MB)
```

(If `STREAM_LORA=dmd2`, the file is `dmd2_sdxl_4step_lora_fp16.safetensors` from
`tianweiy/DMD2`.)

To make the existing ComfyUI backend 4-step, add one node and change `KSampler`.
Against `buildWorkflow()` in `apps/server/src/ai/backends/comfyui.ts`:

**1. Insert a `LoraLoader` as node `12`, between the checkpoint and everything
that consumes it:**

```jsonc
"12": {
  "class_type": "LoraLoader",
  "inputs": {
    "model": ["1", 0],
    "clip":  ["1", 1],
    "lora_name": "lcm-lora-sdxl.safetensors",
    "strength_model": 1.0,
    "strength_clip": 1.0
  }
}
```

**2. Repoint the three consumers of the checkpoint's MODEL/CLIP at it** (the VAE
output `["1", 2]` stays as it is — `LoraLoader` has no VAE output):

| node | input | before | after |
| --- | --- | --- | --- |
| `2` CLIPTextEncode (positive) | `clip` | `["1", 1]` | `["12", 1]` |
| `3` CLIPTextEncode (negative) | `clip` | `["1", 1]` | `["12", 1]` |
| `9` KSampler | `model` | `["1", 0]` | `["12", 0]` |

**3. KSampler values for 4-step LCM** (replacing `steps: 14, cfg: 5.5,
sampler_name: "euler_ancestral", scheduler: "normal"`):

```jsonc
"9": { "class_type": "KSampler", "inputs": {
  "model": ["12", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["8", 0],
  "seed": <seed>,
  "steps": 8,               // = ceil(4 / denoise); ComfyUI also scales steps by denoise
  "cfg": 1.5,               // LCM wants 1.0-2.0. NEVER 5.5 - it burns out at 4 steps
  "sampler_name": "lcm",
  "scheduler": "sgm_uniform",
  "denoise": 0.55
}}
```

Notes:

- ComfyUI's `KSampler` applies `denoise` the same way diffusers does — it runs
  `steps * denoise` actual steps — so pass `ceil(desiredSteps / denoise)` to get
  4 real steps at denoise 0.55. At `denoise: 1.0`, `steps: 4`.
- `sampler_name: "lcm"` with `scheduler: "sgm_uniform"` is the pairing that
  behaves at 4 steps. `normal` visibly under-denoises.
- For **DMD2** instead: `lora_name: "dmd2_sdxl_4step_lora_fp16.safetensors"`,
  `cfg: 1.0`, `sampler_name: "lcm"`, `scheduler: "sgm_uniform"`. DMD2 usually
  holds line art better on Illustrious-class checkpoints; LCM is softer.
- Keep `VAEDecodeTiled` — it is the reason 1024 decode is seconds not minutes on
  this card.
- The LoRA is optional per request only if you build two workflows; simplest is
  a config flag (`COMFYUI_LORA=""` → omit node 12 and keep the current values).

## 6. Wiring instructions for `apps/server` (the only thing you need to do)

`apps/server/src/ai/backends/stream.ts` and `apps/server/test/stream.test.ts`
are committed and green (8 tests, stubbed `fetch`). **They are deliberately not
registered** — registration touches files owned by another agent.

### 6.1 `config.ts`

Add to the backend union and read two vars:

```ts
// AI_BACKEND=comfyui|mock|runpod|stream
streamUrl: env.STREAM_URL ?? 'http://127.0.0.1:8790',
streamTimeoutMs: num(env.STREAM_TIMEOUT_MS, 120_000),
```

| env var | default | meaning |
| --- | --- | --- |
| `AI_BACKEND` | `auto` | now also accepts `stream` |
| `STREAM_URL` | `http://127.0.0.1:8790` | base URL of the worker |
| `STREAM_TIMEOUT_MS` | `120000` | per-generation deadline |

Recommended companion settings when `AI_BACKEND=stream`: `AI_STEPS=4`,
`AI_DENOISE=0.55`, and a shorter `AI_DEBOUNCE_MS` (150–250) — the whole point is
that the round trip is now sub-second.

### 6.2 `backends/index.ts`

```ts
import { StreamBackend, streamReachable } from './stream.js';

export { StreamBackend, streamReachable } from './stream.js';

// inside createBackend(), before the comfyui branch:
if (config.aiBackend === 'stream') {
  log(`[ai] backend: stream at ${config.streamUrl}`);
  return new StreamBackend({ url: config.streamUrl, timeoutMs: config.streamTimeoutMs });
}
```

Optionally give `auto` a first look at the worker (it is the fastest backend, so
it should win when it is up):

```ts
if (await streamReachable(config.streamUrl)) {
  log(`[ai] backend: stream at ${config.streamUrl} (auto-detected)`);
  return new StreamBackend({ url: config.streamUrl, timeoutMs: config.streamTimeoutMs });
}
// ...existing comfyReachable check
```

`streamReachable()` returns true only when `/healthz` answers `ok: true`; a
worker that is still loading its model answers `ok: true, warm: false` and will
simply block the first request until it is ready.

### 6.3 Nothing else changes

`StreamBackend` implements the existing `AIBackend` interface exactly
(`generate(req, signal) → Buffer` of a `size×size` PNG), honours the abort
signal, and has its own request timeout, so `scheduler.ts` needs no changes at
all.

## 7. Honest limitations

- No TensorRT, no torch.compile (compile costs minutes per resolution on
  Windows and would defeat "selectable sizes").
- No cross-request latent reuse. The contract is stateless per request, which
  matches the server's latest-wins scheduler; frame-to-frame consistency between
  successive generations is therefore no better than ComfyUI's.
- One generation at a time per process. Multiple rooms serialise. Multi-room
  batching (`BRUSHJAM_CONTEXT.md` §11.2) is not implemented.
- The worker trusts its caller: no auth, no rate limit, bind is loopback by
  default. Do not expose `STREAM_HOST=0.0.0.0` without putting something in
  front of it.
- Requires ~6 GB free VRAM. With ComfyUI holding its models it will spill to
  shared memory and get several times slower rather than fail; call ComfyUI's
  `/free` (see the worker README) before benchmarking.
