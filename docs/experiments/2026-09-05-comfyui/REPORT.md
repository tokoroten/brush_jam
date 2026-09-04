# Experiment C - ComfyUI on the local RTX 3070 (8 GB)

2026-09-05. Real GPU, `waiNSFWIllustrious_v150.safetensors`, ComfyUI 0.28.0 on
`:8188`, stream worker stopped, ~16.8 GB system RAM free. Prompt for every cell:
`anime style, fantasy town, vibrant colors`, built-in negative, seed 424242,
full-white mask (full-canvas mode), `AI_VAE_TILE=512`.

Run **after** the review-6 step-1 fixes, which matter here: fast mode now really
runs 4 sampler steps at every denoise, and downsampled renders no longer drop
noise strokes outside the top-left corner.

## 1. Latency (`scripts/latency.ts`, n=8, `CANVAS_SIZE=1024`)

`stroke_end` -> `ai_result` as felt by a client; "backend" is the scheduler's own
`latencyMs` (input render + backend + composite, see caveat 3).

| config | steps | gen size | end-to-end median | min-max | backend median |
|---|---|---|---|---|---|
| normal | 14 | 1024 | **10 294 ms** | 10 212-11 867 | 9 844 ms |
| `AI_FAST=1` | 4 | 1024 | **5 665 ms** | 5 303-8 644 | 5 227 ms |
| `AI_FAST=1 AI_WINDOW=768` | 4 | 768 | **3 710 ms** | 3 688-3 753 | 3 284 ms |
| `AI_FAST=1 AI_WINDOW=512` | 4 | 512 | **2 611 ms** | 2 574-4 168 | 2 193 ms |

Raw logs in `latency/`. Roughly: LCM halves 1024, and each resolution step down
takes off another ~30 %. Nothing here is interactive, but 2.6 s is a usable
"draw, wait a beat, see it" loop where 10 s is not.

**The 9.8 s baseline is itself news.** The same 1024 normal workflow measured
14-31 s in the earlier smoke tests. Nothing about the workflow changed; what
changed is that no stream worker was resident and RAM was free. The earlier
numbers were contention, as `opus-stream` suspected - which also means the
"plain VAEDecode takes 1-4 minutes" figure that motivated `VAEDecodeTiled` was
measured under the same contention and should be re-tested before it is treated
as fact.

## 2. Quality grid

Contact sheets: `normal-1024/grid.jpg`, `fast-1024/grid.jpg`,
`fast-768/grid.jpg` (rows = drawings, column 0 = the input, then denoise
0.5 / 0.65 / 0.8 / 0.9; 384 px cells, JPEG so the three fit in ~1.8 MB).
Per-cell latency in each `results.json`.

| grid | median cell | notes |
|---|---|---|
| normal 1024, 14 steps | 9 641 ms | first cell 18 s (checkpoint load) |
| fast 1024, 4 steps | 5 094 ms | first cell 7.4 s (LoRA load) |
| fast 768, 4 steps | 3 051 ms | |

Cell time is flat across denoise - direct confirmation of the step-math fix.
Before it, 0.5 would have cost ~2x 0.9.

### What the denoise levels actually do (normal workflow)

- **0.5 - no-op.** Output is the input with slightly softened lines. Not worth
  a GPU second.
- **0.65 - decoration, not reinterpretation.** The tree grows foliage texture,
  the door gains a handle, the blob acquires shading. Composition untouched.
  This is the "my drawing, tidied" setting.
- **0.8 - the interesting one.** The house becomes an architectural sketch, the
  stick figure becomes an ornate staff-like object, and - the striking result -
  **the noise-pen sky in drawing (c) becomes a row of buildings**: the model
  reads high-frequency noise as structure and resolves it into a plausible
  town. The noise pen works as a "put something here" seed exactly as hoped.
- **0.9 - the drawing is gone.** Every row collapses into unrelated character
  art (the checkpoint's prior: anime characters, character sheets, even fake
  watermark text). Composition, colour and layout are all discarded.

So: reinterpretation starts at 0.8 and the drawing survives up to about 0.85.
0.9 is past the cliff for this checkpoint.

### LCM (fast) vs normal

Same seeds and denoise, different character:

- At 0.5-0.65 LCM is close to the normal workflow - slightly flatter and
  paler, but the same idea. **For "tidy my drawing", 4 steps is a fair trade.**
- At 0.8 LCM does *not* reinterpret as well. The noise sky becomes speckled
  clutter instead of buildings, and the house sprouts an incoherent blob.
  Normal-14 clearly wins the case where the model is meant to invent.
- At 0.9 both collapse, LCM slightly sooner.

The honest summary: LCM buys 2x speed and costs the top end of creative
reinterpretation. It is a good default for the responsive loop, not for the
"make something of this" moment.

### Generation resolution has a hidden cost for the noise pen

Compare the input column of `fast-768` with `fast-1024`: at 768 the noise band
is already averaging towards flat grey, and at 512 it would be greyer still.
Downsampling destroys exactly the high-frequency detail the noise pen exists to
create, so **the noise pen and a low AI resolution work against each other**.
Resolution is a latency knob for line art; noise-pen work wants 1024.

## 3. Recommended default

```
AI_MODE=full  CANVAS_SIZE=1024  AI_WINDOW=1024
AI_DENOISE=0.7  AI_STEPS=14  AI_CFG=5.5  AI_VAE_TILE=512
```

~10 s per edit, reliably. Reasons: 0.55 is a near-no-op on this checkpoint
(the current default is too low to be interesting), 0.7 sits between "tidied"
and "reinterpreted", and the room's denoise slider covers the rest. Fast mode
is worth offering as a documented alternative for a responsive session -
`AI_FAST=1 AI_WINDOW=768`, ~3.7 s, denoise 0.65-0.8 - but not as the default,
because it is visibly weaker at exactly the denoise where the feature is
interesting.

No default was changed in code; this is a recommendation.

## 4. Caveats

1. n=8 per config, one machine, one checkpoint, one prompt, one seed. Enough to
   rank the configs, not to quote as absolute numbers.
2. The three `results.json` files here record `"cfg": 5.5` and no workflow
   detail even for the fast runs, which really used cfg 1.5 and the LCM
   sampler - Codex review 6 finding 11. Fixed in the script immediately after
   this run (`workflow` block); these files predate the fix. The true fast-run
   parameters are: 4 steps, cfg 1.5, `lcm`/`sgm_uniform`,
   `lcm-lora-sdxl.safetensors`.
3. `latencyMs` is labelled "backend generate only" but actually spans input
   render -> backend -> composite, and the end-to-end figure stops at the
   WebSocket message, before the client fetches the PNG (finding 10, still
   open). Both are consistent across configs, so the comparison holds.
4. The first latency sweep of the three fast configs was thrown away: killing
   the `tsx` wrapper left its node child holding port 8799, so the "fast" runs
   silently re-measured the normal server. The numbers above come from a rerun
   that kills the listener itself and asserts the port was free. The tell was
   three configs agreeing to within 1 %.
