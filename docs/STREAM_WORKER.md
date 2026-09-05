# Stream worker — decision record, benchmarks, and wiring instructions

Scope: `apps/stream-worker` (Python) plus the `stream` backend, now
`apps/brushjam/src/brushjam/ai/backends/stream.py`.
Written for the agent that owned the room server / `apps/web`: **section 6 is the
only part you need to act on.**

Hardware this was measured on: Windows 11, RTX 3070 8 GB, driver 591.86,
CUDA 12.4 wheels, ComfyUI 0.28 running alongside on :8188.

---

## 1. Why a stream worker at all

The ComfyUI backend re-walks a graph for every request, costing ~10 s at
1024²/14 steps — far from the "AI reacts while you draw" feeling
`BRUSHJAM_CONTEXT.md` §11 wants. A resident process that holds the model in VRAM
and runs 4 LCM steps answers the *same* JSON contract far faster.

**Measured, warm, on an RTX 3070 8 GB: 0.76 s at 512², 1.73 s at 768², 3.35 s at
1024²** (§4.1), against 2.6 / 3.7 / 5.7 s for the same 4-step workflow through
ComfyUI (§4.6). It wins at every size.

That was not true for most of this component's life. Until the VAE was replaced
it was 1.6 / 5.7 / 14.3 s and lost to ComfyUI at two sizes out of three; §4.2
and §4.3 record how that was diagnosed, including two confident A/B tests that
found nothing. The history is kept deliberately — the wrong turns are the
useful part.

## 2. StreamDiffusion evaluation (step 1 of the brief)

**Short answer: yes, StreamDiffusion was really installed and really run here.**
The livepeer fork was installed into a Python 3.10 venv on this Windows box and
successfully generated SDXL img2img frames from the local Illustrious
checkpoint. It was rejected on measured speed and contract fit, not on a guess.
Upstream `cumulo-autumn/StreamDiffusion` was cloned and read but deliberately
**not** installed — see §2.1 for why that would have been wasted time.

Exact commands, so this is reproducible:

```bash
git clone --depth 1 https://github.com/cumulo-autumn/StreamDiffusion.git sd_eval
git clone --depth 1 https://github.com/livepeer/StreamDiffusion.git   sd_livepeer

uv venv --python 3.10 sdenv
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
uv pip install -e ./sd_livepeer
uv pip install opencv-python "controlnet-aux==0.0.10"    # undeclared, see below
python sd_smoke.py                                        # SDXL img2img, 512, 3 t-indices
```

Versions actually used: Python 3.10.17, torch 2.6.0+cu124, driver 591.86,
streamdiffusion 0.1.1 (livepeer `main`), transformers 4.56.0, its pinned
`diffusers` fork, `acceleration="none"` (no TensorRT, per the brief).

### 2.1 `cumulo-autumn/StreamDiffusion` (upstream) — rejected, not installed

Cloned and read; **not** installed, because the source shows it cannot do the
job and installing it would have meant a second 2.5 GB torch download to prove
a foregone conclusion:

- `src/streamdiffusion/pipeline.py` imports `StableDiffusionPipeline` and
  nothing else; `grep -rn "XL" src/` returns **zero** hits, and the README
  never mentions SDXL. It is an SD1.5-only pipeline. There is no code path that
  could load an SDXL checkpoint, so "try it and see" has one possible outcome.
- Dependencies are frozen at `diffusers==0.24.0` / `onnx==1.15` / torch ~2.1,
  2023-era pins that conflict with anything current.

Our checkpoint is SDXL/Illustrious, so upstream is out on capability grounds
before portability even matters. (If someone wants SD1.5 realtime specifically,
upstream is the right starting point — but that is a different product
decision, since Illustrious is the model the room is styled around.)

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

### 2.4 What StreamDiffusion would buy us later

Honest accounting of its four real optimisations against what this worker
already does. **The per-technique factors below are from the StreamDiffusion
paper and the TensorRT/TAESD literature, not measured here** — only the "already
have it" rows are things this box actually demonstrated.

| technique | what it does | do we have it | estimated gain on this pipeline |
| --- | --- | --- | --- |
| prompt-embedding cache | skip the two CLIP encoders when the prompt is unchanged | **yes**, implemented | already banked (~0.5–0.6 s per cache miss avoided; the room prompt rarely changes, so nearly every request hits) |
| Tiny VAE (`taesdxl`) | replace the SDXL VAE with a ~1 M-param distilled one | no | **the largest single win available**, and it does not need StreamDiffusion. Our fp32 tiled VAE decode is a substantial slice of §4's numbers; taesdxl decodes in tens of ms. Worth doing in this worker directly. |
| RCFG (residual CFG) | keep a negative prompt at ~1.0–1.3× UNet cost instead of 2× | no | ~1.5–1.9× on the UNet portion. We can already get ~2× by setting `STREAM_GUIDANCE=1.0`, but that *drops* the negative prompt; RCFG is the version that keeps it. |
| stream batching | run the N denoising steps of *consecutive* frames as one batched UNet call | no | ~N× **throughput** on a continuous video stream, ~0 for us. It needs a steady frame feed; our scheduler sends debounced, latest-wins, one-at-a-time requests whose size changes per request. Realising this would mean redesigning `scheduler.ts` around a persistent per-room stream, not just swapping backends. |
| TensorRT | compile the UNet to fixed-resolution engines | no (out of scope per the brief) | ~1.5–2.5× on the UNet on Ampere. Costs minutes of engine build per resolution, which fights the "512/768/1024 selectable per request" requirement — you would build three engines. |

Rough combined estimate, stacking only the parts that apply to a request/response
worker (taesdxl + RCFG + TensorRT, no stream batching): **~2.5–4× faster than
§4 on this RTX 3070**, i.e. a 1024² 4-step generation in the low hundreds of ms
rather than seconds. On a 4090 the same stack is comfortably sub-200 ms at
1024², and *with* stream batching and a redesigned streaming scheduler, 512²
interactive rates (10–15 fps) become plausible — which is the regime
StreamDiffusion was actually built for.

The sequencing that follows: **take the tiny VAE first** (biggest win, no new
dependency, no architecture change), then reconsider RCFG/TensorRT on better
hardware, and only adopt StreamDiffusion itself if and when the product moves to
a continuous-stream model (`BRUSHJAM_CONTEXT.md` §11.2's sticky room→GPU
sessions) rather than the current stateless latest-wins requests.

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

### 3.1 One model at a time on 8 GB

**On this machine ComfyUI and the stream worker must never be resident at the
same time.** This is a constraint of an 8 GB card, not of the design — on a card
with headroom they coexist fine.

The arithmetic: ComfyUI holds ~3–4 GB with the Illustrious checkpoint loaded,
the stream worker holds ~5.5–6 GB, the Windows desktop takes ~0.6–1.3 GB. That
is 10+ GB of demand on 8 GB of card. Windows does not fail this allocation — it
silently backs the overflow with shared system memory over PCIe, and *both*
processes get several times slower with no error anywhere.

Measured during this work, with both resident:

| job | normal | while both were resident |
| --- | --- | --- |
| ComfyUI 1024² / 14 steps | 10–24 s | **116 s and 567 s** |
| worker 768² / 4 steps | see §4 | 17–24 s |
| worker 1024² / 4 steps | see §4 | never returned (300 s client timeout) |

Rules that follow:

1. **The server must not use `comfyui` and `stream` at the same time locally.**
   Pick one via `AI_BACKEND`. Do not add an "auto-detect and fall back" path that
   could end up talking to both, and do not run a ComfyUI smoke test against a
   room while the worker is up.
2. **Hand the GPU over explicitly.** The worker exposes `POST /unload`, which
   drops the pipeline and frees its VRAM while keeping the process alive
   (`GET /healthz` then reports `loaded: false`). `POST /load` puts it back, and
   a `/generate` that arrives while unloaded reloads transparently — at the cost
   of the ~90 s cold start, so prefer an explicit `/load`.
3. The mirror image for ComfyUI is
   `POST /free {"unload_models": true, "free_memory": true}`, which only takes
   effect once its queue is empty (`GET /queue`).
4. Benchmark numbers taken while the other side was loaded are worthless. Every
   figure in §4 was measured with ComfyUI stopped.

## 4. Benchmarks

Conditions for every number below: **ComfyUI stopped**, GPU otherwise idle,
worker warm, 4 steps, `denoise=0.55`, feathered mask, prompt
`"anime style, fantasy town, vibrant colors"` held constant so the embedding
cache hits every time. 2 discarded warm-up runs, then 5 measured runs per size
(`scripts/bench.py --sizes 512 768 1024 --runs 5 --warmups 2 --mask`).

Cold start: checkpoint load + LoRA fuse **~30 s**, plus one warm-up generation
(3.0 s at 512², 9.9 s at 768²) — **~35–41 s to first useful request** with a warm
file cache, ~90 s from cold.

### 4.1 Final numbers (shipped configuration: `STREAM_VAE=fp16fix`)

| size | runs | wall median | min | max | UNet | VAE encode | VAE decode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512² | 5 | **764 ms** | 754 ms | 813 ms | 517 ms | 70 ms | 108 ms |
| 768² | 5 | **1729 ms** | 1719 ms | 1736 ms | 1029 ms | 208 ms | 383 ms |
| 1024² | 5 | **3354 ms** | 3346 ms | 3479 ms | 1880 ms | 423 ms | 896 ms |

Cold start ~30–50 s (checkpoint load + LoRA fuse + one warm-up run). Prompt
embedding is 0 ms on a cache hit, PNG encode 22–78 ms, `empty_cache` 10–11 ms.

The spread is now tight (512² varies by 59 ms across five runs, 768² by 17 ms),
which is itself evidence that the VRAM spill described in §4.3 is gone.

**The goal is met at 512² and close at 768².** "Sub-second, AI reacts while you
draw" is true at 512²; 1024² at 3.35 s is a different interaction — fast enough
to feel responsive between strokes, not fast enough to feel live.

### 4.2 How it got there — including two confident wrong turns

Four A/B tests. **Two found nothing**, and they are the instructive ones,
because both were plausible and both were wrong:

| variable | 512² | 768² | 1024² | verdict |
| --- | --- | --- | --- | --- |
| baseline (CFG 1.5, tiling on, checkpoint VAE) | 1542 ms | 12269 ms | 122763 ms | — |
| `STREAM_GUIDANCE=1.0` (no CFG: half the UNet work) | 1670 ms | 11740 ms | — | **no effect** |
| `STREAM_VAE_TILING=0` (plain VAE decode) | 1674 ms | 12233 ms | — | **no effect** |
| `STREAM_EMPTY_CACHE=1` (+ `expandable_segments`) | 1621 ms | 5654 ms | 14309 ms | **2× / 8×** |
| **`STREAM_VAE=fp16fix`** | **764 ms** | **1729 ms** | **3354 ms** | **2× / 3× / 4×** |

1. **Halving the UNet work changed nothing.** With CFG on, 512² runs 8 UNet
   evaluations in ~1.5 s; with CFG off it runs 4 in ~1.6 s.
2. **Turning off VAE tiling changed nothing either.**
3. **Returning the allocator's cache was worth 2–8×** (§4.3).
4. **Replacing the VAE was worth another 2–4×** (§4.4).

Why (1) and (2) both missed: neither touches the VAE's *dtype*, and the fp32
upcast was ~80% of the time at 1024². Test (2) in particular looked like it had
exonerated the VAE — it had only exonerated *tiling*. Ruling out a component by
testing one of its knobs is not ruling out the component.

### 4.3 The actual bug: the allocator held 2.5 GB it was not using

`GET /healthz` now reports a `memory` block, which made this visible in one
request. Idle, after loading and one 512² warm-up:

```jsonc
// before
{ "allocated_gb": 5.05, "reserved_gb": 7.52, "max_allocated_gb": 6.59, "device_free_gb": 0.00 }
// after STREAM_EMPTY_CACHE=1 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{ "allocated_gb": 5.05, "reserved_gb": 5.32, "max_allocated_gb": 6.59, "device_free_gb": 1.21 }
```

Live tensors were only 5.05 GB, but PyTorch's caching allocator had reserved
**7.52 GB** of the 8 GB card and **`device_free_gb` was 0.00**. The transient
peak during a generation is 6.59 GB (the fp32 VAE upcast is most of the gap over
5.05 GB), so every request after the first had to find its activations in a
fragmented pool with nothing left underneath — and on Windows that does not
fail, it silently spills to shared system memory over PCIe.

That is why the scaling looked impossible: 768² has 2.25× the pixels of 512² but
took 8× the time, and 1024² took 80×. It was not compute, it was paging. The
huge 1024² spread in the old data (83 s to 256 s) was allocation luck.

The fix is `torch.cuda.empty_cache()` after each generation (`STREAM_EMPTY_CACHE`,
on by default) plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It costs
70 ms at 512² and ~600 ms at 1024², and buys 2× at 768² and 8× at 1024².

### 4.4 The VAE was the fixed cost — measured, not inferred

This checkpoint's VAE config sets `force_upcast=True`, so diffusers cast it to
**fp32** for every encode and decode. `STREAM_VAE` now selects an alternative,
and the three-way comparison (`scripts/bench_vae.py`, 5 runs per cell, ComfyUI
stopped) settles it:

| VAE | cold start | 512² | 768² | 1024² |
| --- | --- | --- | --- | --- |
| `checkpoint` (fp32 upcast) | 22.3 s | 1621 ms | 5654 ms | 14309 ms |
| **`fp16fix`** (default) | 48.5 s | **764 ms** | **1729 ms** | **3354 ms** |
| `taesd` (distilled) | 20.3 s | 697 ms | 1176 ms | 2113 ms |

The phase breakdown shows exactly where it went. **The UNet is unchanged across
all three** — the entire difference is VAE decode:

| VAE | 1024² UNet | 1024² VAE encode | 1024² VAE decode |
| --- | --- | --- | --- |
| `checkpoint` | 1915 ms | 2500 ms | **9133 ms** |
| `fp16fix` | 1880 ms | 423 ms | 896 ms |
| `taesd` | 1878 ms | 47 ms | 65 ms |

An fp32 decode of a 1024² latent was costing **9.1 s** against 0.9 s in fp16 —
a 10× penalty for a dtype flag, on a model whose actual sampling takes 1.9 s.

`fp16fix` is `madebyollin/sdxl-vae-fp16-fix`: the same architecture with weights
rescaled so activations do not overflow fp16. It is a numerical fix, not an
approximation, so output should be equivalent to the checkpoint VAE. Its slower
cold start (48.5 s) is the one-time HF download on first run.

**On `taesd`:** faster again (2.1 s at 1024²) and it frees ~0.7 GB of VRAM. It
is a *distilled* VAE, so unlike `fp16fix` it genuinely trades fidelity for
speed, and **I have not looked at its output**. Do not default to it without a
visual comparison. It is the right lever if 1024² under 2 s is worth softer
detail.

Remaining leads, now that the VAE is fixed:

1. `STREAM_EMPTY_CACHE` costs 10–11 ms with `fp16fix`, down from 67–755 ms with
   the old VAE, because there is far less to reclaim. Worth re-testing with it
   off — it may no longer be needed at all.
2. The UNet is now the dominant cost (1.9 s of 3.35 s at 1024²), so RCFG and
   TensorRT (§2.4) are finally the right things to look at. They were not before.

### 4.5 Output quality at 4 steps is also disappointing

Separate from speed, and worth knowing before anyone wires this up. Sample
inputs and outputs are in `apps/stream-worker/samples/`.

At the server's current default of `denoise=0.55` with 4 LCM steps, the output
is **very close to the input**: the drawn shapes get a soft shade and some
speckle, but the model does not reinterpret the sketch into "anime fantasy
town". That is expected behaviour rather than a bug — an LCM-distilled model's
consistency function is trained to jump toward a clean image from *high* noise
levels, so entering the trajectory at 55 % gives it little to do. `AI_DENOISE`
0.55 was tuned for 14-step `euler_ancestral` and does not transfer.

Recommendation: use **`AI_DENOISE` around 0.75–0.85**, not 0.55.

**This has since been measured** — not by me, and not on this worker, but on the
same checkpoint through the ComfyUI backend, by the room-server agent's
quality grid (`docs/experiments/2026-09-05-comfyui/`). Their finding, on a 48-cell
sweep of four drawings:

| denoise | result |
| --- | --- |
| 0.5 | no-op — the output is the input with softer lines |
| 0.65 | decorates without touching composition |
| **0.8** | **genuinely reinterprets** — a noise-pen sky resolves into buildings |
| 0.9 | past the cliff: the drawing is discarded for the checkpoint's own priors |

So the estimate was right, and it is a property of the checkpoint rather than of
few-step sampling — it holds for the 14-step workflow too. What is still
unmeasured is whether **this worker** behaves identically at those values;
`scripts/quality_probe.py 512` sweeps it in one command when a GPU window
allows.

One caveat that matters more for this worker than for ComfyUI: the same grid
found LCM tracks the 14-step result closely up to ~0.65 but is **visibly weaker
at 0.8**, where a noise-pen sky becomes speckled clutter instead of
architecture. 0.8 is exactly where reinterpretation starts, so few-step sampling
is weakest precisely at the setting that makes the feature worth having. Still
worth comparing `STREAM_LORA=dmd2`, which usually holds line art better on
Illustrious-class checkpoints.

### 4.6 Against a fair baseline: the worker now wins at every size

The right baseline is not ComfyUI at 14 steps, it is ComfyUI running §5's own
4-step LoRA workflow. Measured by the room-server agent on this box with
nothing else resident:

| size | ComfyUI 14-step | ComfyUI 4-step (`AI_FAST`) | this worker (`fp16fix`) |
| --- | --- | --- | --- |
| 512 | - | 2.6 s | **0.76 s** |
| 768 | - | 3.7 s | **1.73 s** |
| 1024 | 10.3 s | 5.7 s | **3.35 s** |

**Caveat that matters:** their figures are end-to-end through the server
(render, mask, composite, broadcast); mine are worker-only. Their logs put the
server-side share at roughly 400-450 ms, so the fair 1024 comparison is about
3.8 s versus 5.7 s, not 3.35 versus 5.7. The worker still leads at every size,
by 1.5-3x rather than 1.7-3.4x.

For most of this component's life the opposite was true - at 1.6 / 5.7 / 14.3 s
it lost at two sizes out of three, and an earlier version of this section
recommended deleting it in favour of `AI_FAST`. The VAE fix (§4.4) reversed
that. The recommendation is now **keep the worker, default `STREAM_VAE=fp16fix`**.

Two caveats that survive the reversal, both from the room-server agent's
quality grids and neither addressed by making things faster:

- **Downsampling destroys the noise pen.** At 768 a noise-pen stroke already
  averages to flat grey before the model sees it. That makes 1024 the size that
  matters for texture reinterpretation - which is why 1024 dropping from 14.3 s
  to 3.35 s is the important cell in the table, not 512.
- **LCM is not a free win at high denoise.** Up to ~0.65 it tracks the 14-step
  result closely; at 0.8 - the denoise §4.5 recommends for real
  reinterpretation - it is visibly weaker, resolving textures into speckled
  clutter where 14 steps resolves them into structure. Few-step sampling is a
  latency/quality trade, and it is weakest exactly where the feature earns its
  keep. This applies to `AI_FAST` and to this worker equally, and it is a
  question about *sampling*, not about which process runs it.

## 5. ComfyUI: the same LoRA, a 4-step Illustrious workflow

The worker downloads the LCM LoRA **into ComfyUI's own directory** so both use
one file:

```
E:\ComfyUI\models\loras\lcm-lora-sdxl.safetensors     (latent-consistency/lcm-lora-sdxl, ~394 MB)
```

(If `STREAM_LORA=dmd2`, the file is `dmd2_sdxl_4step_lora_fp16.safetensors` from
`tianweiy/DMD2`.)

To make the existing ComfyUI backend 4-step, add one node and change `KSampler`.
Against `buildWorkflow()` in the ComfyUI backend (now `apps/brushjam/src/brushjam/ai/backends/comfyui.py`):

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
  "steps": 4,               // the REAL step count; do NOT divide by denoise
  "cfg": 1.5,               // LCM wants 1.0-2.0. NEVER 5.5 - it burns out at 4 steps
  "sampler_name": "lcm",
  "scheduler": "sgm_uniform",
  "denoise": 0.8            // 0.55 is a near-no-op; see 4.5
}}
```

Notes:

- **`steps` is the real step count at any denoise. Do not scale it.** An earlier
  version of this document said to pass `ceil(steps / denoise)`, on the
  assumption that ComfyUI truncates the schedule the way diffusers does. It does
  not, and the room-server agent caught the error in review. `comfy/samplers.py`
  `KSampler.set_steps` is explicit:

  ```python
  new_steps = int(steps / denoise)
  sigmas = self.calculate_sigmas(new_steps).to(self.device)
  self.sigmas = sigmas[-(steps + 1):]      # exactly `steps` sampling steps
  ```

  ComfyUI already does the division internally. Passing `ceil(4 / 0.55)` made
  "4-step" mode run **8** steps — double the intended cost, and 20 steps at the
  room's minimum denoise. Pass `steps: 4` and mean it.

  This is a genuine asymmetry between the two runtimes, which is what made it
  easy to get wrong: **diffusers keeps only the last `n * strength` timesteps
  without compensating**, so this worker's `steps_for_strength()` really does
  need `ceil(steps / denoise)`. Same parameter names, opposite conventions. Do
  not copy the compensation from one into the other.
- `sampler_name: "lcm"` with `scheduler: "sgm_uniform"` is the pairing that
  behaves at 4 steps. `normal` visibly under-denoises.
- For **DMD2** instead: `lora_name: "dmd2_sdxl_4step_lora_fp16.safetensors"`,
  `cfg: 1.0`, `sampler_name: "lcm"`, `scheduler: "sgm_uniform"`. DMD2 usually
  holds line art better on Illustrious-class checkpoints; LCM is softer.
- Keep `VAEDecodeTiled` — it is the reason 1024 decode is seconds not minutes on
  this card.
- The LoRA is optional per request only if you build two workflows; simplest is
  a config flag (`COMFYUI_LORA=""` → omit node 12 and keep the current values).

## 6. Wiring instructions for the room server (historical)

The `stream` backend and its tests (then TypeScript, now
`apps/brushjam/src/brushjam/ai/backends/stream.py` and
`apps/brushjam/tests/test_backends_http.py`)
are committed and green (8 tests, stubbed `fetch`). **They are deliberately not
registered** — registration touches files owned by another agent.

> **Before you wire this up, read §4.** Register it — that is cheap and nothing
> routes to it until someone asks. But on this card **make it explicit-only:
> `AI_BACKEND=stream` and nothing else.** Do not put it in the `auto` probe
> order. §6.2 explains why; an earlier draft of this document suggested the
> opposite and was wrong.

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
**`AI_DENOISE=0.8`** (not the 0.55 default — at 4 LCM steps 0.55 barely changes
the drawing, see §4.5), **`AI_WINDOW=768`** (768 is both twice as fast as 1024
*and* transforms the drawing more — see
`docs/experiments/2026-09-05-stream/REPORT.md` §5), and `AI_DEBOUNCE_MS` around
300. The round trip is 0.76 s at 512², 1.8 s at 768² and 3.5 s at 1024², so a
very short debounce just queues work the GPU cannot absorb.

**The worker now defaults to `STREAM_LORA=dmd2` and `STREAM_GUIDANCE=1.0`**, so
there is nothing to set. Measured over four 768 grids
(`docs/experiments/2026-09-05-stream/REPORT.md` §8), DMD2 is 1384 ms against
LCM's 1798 ms and reinterprets a whole denoise step earlier with cleaner
objects; DMD2 is guidance-distilled, so dropping CFG costs it nothing, whereas
the same change makes LCM's line art fade (1.64% surviving ink at denoise 0.8
against the input's 2.36%).

**One thing the server must handle: at CFG 1.0 the negative prompt is inert.**
The worker still accepts `negative_prompt` and still computes the embeddings; it
simply never uses them, because classifier-free guidance is what gives a
negative prompt its effect. Read `negative_prompt_active` from `/healthz` (§6.4)
and grey out or annotate the room's negative-prompt field accordingly, rather
than letting a user type into a box that does nothing. To turn it back on, set
`STREAM_GUIDANCE=1.5` (+27% latency); DMD2 still beats LCM at that setting.

**Do not hard-code the top of the denoise range.** Read it from `/healthz`
(`max_denoise`, §6.4) and clamp the room slider to it. The worker enforces the
same ceiling server-side, so a request above it is silently clamped rather than
rejected — the point of publishing the number is that the UI can stop offering a
value that will not be honoured.

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

**Do not add it to the `auto` probe order.** `streamReachable()` exists, and an
earlier draft of this section suggested using it that way, but on reflection
that is the wrong default for two reasons:

1. **It changes output quality silently.** The room's `AI_DENOISE` default of
   0.55 is tuned for 14-step `euler_ancestral`. Routed to this backend at 4 LCM
   steps, the same 0.55 produces a near-no-op — the AI panel keeps updating and
   keeps looking like the drawing (§4.5). A probe that silently reroutes to a
   backend needing different settings turns a forgotten worker process into
   "the AI stopped doing anything", which is a horrible thing to debug.
2. **It is not clearly the faster backend.** It wins at 512² and 768² and is a
   wash at 1024² (§4.1). "Fastest backend, so it should win when it is up" was
   an assumption, and the measurements did not support it.

Reachability is also the wrong signal here: `/healthz` answering means the
worker is *holding 5 GB of VRAM*, which on this card means ComfyUI is already
being starved (§3.1). Which model owns the GPU is a deployment decision someone
should make deliberately, not something a probe should infer.

So: explicit `AI_BACKEND=stream` only. Revisit auto-detection if the §4.4 VAE
work lands and the worker becomes decisively faster at every size.

For reference, `streamReachable()` returns true only when `/healthz` answers
`ok: true`. A worker still loading its model answers `ok: true, warm: false` and
will block the first request until it is ready — useful for a startup log line
or a health page, just not for backend selection.

### 6.3 Cancellation, `request_id` and the busy contract

**Aborting the HTTP request does not stop the GPU.** When `scheduler.ts` aborts
a stale generation, the worker is already inside a diffusion loop on a
background thread; the socket closing does not reach it. Left alone it runs to
completion, holds the GPU, and the next request queues behind work nobody will
ever read. On a 15 s 1024² job that is 15 s of dead time per abandoned request,
and the scheduler abandons requests routinely (prompt changes, stale revisions,
its own watchdog).

So cancellation is cooperative:

| call | meaning |
| --- | --- |
| `POST /generate { …, request_id, queue }` | `request_id` is caller-chosen (the worker invents one if absent) and comes back in the response. `queue: true` waits for the GPU; `queue: false` (default on the worker) is refused with **409** while busy. |
| `POST /cancel { request_id }` | Ask it to stop. Returns `{ ok, request_id, state: "running" \| "pending" }`. Cancelling an unknown or already-finished id is **not** an error — the server may cancel something the worker just finished. |
| `GET /healthz` → `busy`, `current_request_id` | What the GPU is doing right now. |

Status codes `/generate` can return: **200** success, **400** undecodable image
or out-of-range size, **409** busy (only when `queue` is false; carries
`retry-after: 1`), **499** cancelled (either cancelled mid-flight or cancelled
while queued), **500** generation failed, **503** the model failed to load.

A cancelled job stops at its **next diffusion step**, so cancellation latency is
one step — 0.1–1 s depending on size. It cannot interrupt a step already on the
GPU; nothing in PyTorch can.

**`StreamBackend` already implements all of this** — you do not need to write
it. It generates a `request_id` per call, sends `queue: true` (the scheduler
already allows one in-flight request per room, so a 409 would only mean "another
room is generating", which is worth waiting for), fires `POST /cancel` when the
caller's signal aborts *or* its own timeout fires, maps 499 → `AbortedError` and
409 → a "stream worker is busy" error. `queue: false` is available via
`new StreamBackend({ url, queue: false })` if you ever want fail-fast instead.

If you want the server to wait for a free GPU rather than send `queue: true`,
poll `GET /healthz` until `busy: false` — but prefer `queue: true`, which does
the same thing without a polling loop.

### 6.4 What the server relies on from `/healthz`

*Added by the server side (review 6 finding 1). The fields already exist; this
records which ones are now load-bearing, so they are not dropped or renamed.*

`AI_BACKEND=auto` no longer picks the worker at all unless `AI_STREAM_AUTO=1`
(reachability is the wrong signal: a worker answering `/healthz` is holding
~5 GB of VRAM, which starves ComfyUI on an 8 GB card). When it *is* opted in,
`createBackend` requires all three of:

| field | requirement | why |
| --- | --- | --- |
| `ok` | `true` | basic liveness |
| `warm` | `true` | a worker with no model loaded answers `ok` and then echoes the input back; the scheduler cannot tell that from a real result |
| `max_size` | `>= AI_WINDOW` (or absent/0, which is treated as unknown and accepted) | an undersized worker answers `ok` and then 400s every request, which full mode retries forever |

Two further `/healthz` fields are advisory rather than gating:

| field | default | meaning |
| --- | --- | --- |
| `vae` | `fp16fix` (`STREAM_VAE`) | which VAE is installed. `stream.ts` already reads this into its quality-grid reports; it is the single largest performance variable (§4.4) and appears nowhere else in the artefacts |
| `lora` | `dmd2` (`STREAM_LORA`) | which 4-step distillation LoRA is fused. Worth logging, because it changes output character substantially and appears nowhere else |
| `negative_prompt_active` | `false` (derived: `guidance > 1.0`) | whether `negative_prompt` has any effect. **False by default**, because the shipped `STREAM_GUIDANCE=1.0` turns CFG off. Grey out or annotate the negative-prompt field when this is false; do not infer it from `guidance` yourself |
| `max_denoise` | `0.9` (`STREAM_MAX_DENOISE`) | highest denoise the worker will honour; requests above it are clamped, not refused. Cap the room's denoise slider to this so the UI cannot offer a value the backend will ignore. Absent means an older worker: fall back to 1.0. |
| `steps` | `4` (`STREAM_STEPS`) | the worker's own step count. The server's `AI_STEPS` is *not* forwarded as sampling truth for this backend, so this is the only place the real number appears |

`max_denoise` exists because the useful ceiling is a property of the backend,
not of the room: at 4 LCM steps everything above ~0.9 stops reinterpreting the
drawing and starts replacing it outright. It was set to 0.9 rather than 1.0 for
that reason, and because the denoise-quantisation bug
(`docs/experiments/2026-09-05-stream/REPORT.md` §4, fixed in §6) showed that a
slider offering values the sampler cannot distinguish is worse than a shorter
slider.

Each rejection is logged with its reason and falls through to ComfyUI. An
explicit `AI_BACKEND=stream` is always honoured, but the same probe runs and
prints a warning.

One extra requirement on `current_request_id`: after `POST /cancel`, it should
keep naming the cancelled job until that job has really stopped, then clear (or
change) along with `busy`. `StreamBackend` waits on exactly that signal before
issuing its next request — a retry sent while the abandoned job is still running
would queue behind it and pay for it twice — with a 15 s deadline after which it
gives up and sends anyway. If `current_request_id` clears before the GPU is
actually free, that wait becomes useless.

### 6.4.1 `steps` is a request, not a promise

At low denoise there are fewer distinct timesteps left below the start point
than the requested step count, so the run is shorter than asked: **denoise 0.2
with 20 steps runs 10**, because a 50-point distillation schedule has only 10
entries below timestep 199. The worker used to run 10 and let the caller believe
it had run 20.

Two things now prevent that:

* **`/generate` responses carry a top-level `steps`** — the count actually run,
  never the count requested. `timings.steps_requested` and
  `timings.steps_effective` carry both if you want the difference. A clamp is
  logged at INFO.
* **`strict_steps: true`** in the request body turns the clamp into a `400`
  naming both numbers, for callers that would rather fail than silently get a
  different sampling than they asked for. It is **off by default**, so the
  server's existing contract is unchanged.

The server's own defaults (4 steps, denoise 0.5–0.9) are nowhere near this
boundary — 4 steps needs denoise ≥ 0.08 — so this matters for benchmark scripts
and for anyone exposing a steps control, not for the playtest path.

### 6.5 Verified against a real worker

The TS backend and the Python worker have now been driven against each other
end-to-end, with no GPU, using the worker's dry-run mode:

```bash
cd apps/stream-worker && STREAM_DRY_RUN=1 STREAM_PORT=8795 uv run python -m stream_worker
# then, against the real StreamBackend:
#   streamReachable() sees the worker              PASS
#   generate() returns a Buffer                    PASS
#   result is a PNG, exactly size x size           PASS
#   oversized request rejects with the worker's message   PASS
#   caller abort surfaces as AbortedError          PASS
```

`STREAM_DRY_RUN=1` serves the whole contract — status codes, field names,
`request_id`, sizes — without loading a model or touching the GPU. Use it in CI
and for any server-side work on this backend; it is the cheapest way to catch a
contract mismatch, and it costs no GPU minutes.

### 6.6 Nothing else changes

`StreamBackend` implements the existing `AIBackend` interface exactly
(`generate(req, signal) → Buffer` of a `size×size` PNG), honours the abort
signal, cancels the GPU job when it does, and has its own request timeout, so
`scheduler.ts` needs no changes at all.

## 7. Honest limitations

**Read §4.1 first: the worker is committed and works, but it did not hit its
performance goal, and at 768²/1024² it is currently no better than the ComfyUI
backend.** Everything below is on top of that.

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
