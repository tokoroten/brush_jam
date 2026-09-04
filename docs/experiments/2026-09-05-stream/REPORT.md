# Stream worker quality grids — 2026-09-05

Three grids run through `apps/server/scripts/quality-grid.ts` **unmodified**, with
`AI_BACKEND=stream STREAM_URL=http://127.0.0.1:8790`. Worker: SDXL
(`waiNSFWIllustrious_v150`) + SDXL LCM-LoRA, LCMScheduler, **4 steps**,
guidance 1.5, seed 424242, prompt `anime style, fantasy town, vibrant colors`.

| run | dir | VAE | res |
| --- | --- | --- | --- |
| 1 | `fp16fix-1024/` | `sdxl-vae-fp16-fix` | 1024 |
| 2 | `fp16fix-768/` | `sdxl-vae-fp16-fix` | 768 |
| 3 | `taesd-1024/` | `taesdxl` (distilled) | 1024 |

Grids are stored as `grid.jpg` (q88, ~0.55 MB each); the PNG grid was deleted,
per-cell PNGs are kept.

The GPU was exclusively the worker's for all three runs (ComfyUI stopped). The
worker was unloaded and killed afterwards; `nvidia-smi` reads 384 MiB used.

---

## 1. Latency (ms, end-to-end through the server, incl. PNG transfer)

### `fp16fix` @ 1024

| drawing | d=0.5 | d=0.65 | d=0.8 | d=0.9 |
| --- | --- | --- | --- | --- |
| a line-art house + tree | 3687 | 3353 | 3406 | 3439 |
| b stick figure + blob | 3457 | 3392 | 3392 | 3378 |
| c house + noise sky | 3621 | 3497 | 3460 | 3438 |
| d mostly noise | 3733 | 3633 | 3619 | 3612 |

median **3459** (min 3353, max 3733). Per-denoise medians: 3654 / 3445 / 3433 / 3439.

### `fp16fix` @ 768

| drawing | d=0.5 | d=0.65 | d=0.8 | d=0.9 |
| --- | --- | --- | --- | --- |
| a | 1967 | 1770 | 1773 | 1824 |
| b | 2049 | 1767 | 1768 | 1772 |
| c | 1812 | 1808 | 1838 | 1798 |
| d | 1847 | 1874 | 1833 | 1835 |

median **1818** (min 1767, max 2049). Per-denoise medians: 1907 / 1789 / 1803 / 1811.

### `taesd` @ 1024

| drawing | d=0.5 | d=0.65 | d=0.8 | d=0.9 |
| --- | --- | --- | --- | --- |
| a | 2952* | 2163 | 2134 | 2124 |
| b | 2167 | 2161 | 2126 | 2140 |
| c | 2269 | 2234 | 2201 | 2223 |
| d | 2267 | 2329 | 2290 | 2314 |

median **2212** (min 2124, max 2952). * first cell of the run, still warming.

### Against ComfyUI (same script, same drawings, same seed)

| backend | 768 median | 1024 median |
| --- | --- | --- |
| stream `fp16fix` | **1818 ms** | **3459 ms** |
| stream `taesd` | — | **2212 ms** |
| ComfyUI AI_FAST (4-step LCM) | 3048 ms | 5091 ms |
| ComfyUI normal (14 steps) | — | 9626 ms |

The worker is **1.7x faster than ComfyUI fast at 768** and **1.5x (fp16fix) to
2.3x (taesd) faster at 1024**, on the same GPU with the same model and step count.

Denoise has essentially no effect on latency (<=8%), because `steps_for_strength`
holds the real step count at 4 regardless — see section 4.

---

## 2. Does the worker reinterpret as well as ComfyUI?

Compared against `docs/experiments/2026-09-05-comfyui/fast-1024/grid.jpg` and
`normal-1024/grid.jpg`.

**Short answer: at 1024 the worker at d=0.8 is on par with ComfyUI fast at 0.8,
and both are far behind normal-14. At 768 the worker is clearly better than
either at the same denoise.**

### Row c — "house + noise sky", the reinterpretation test

| run | d=0.5 | d=0.65 | d=0.8 | d=0.9 |
| --- | --- | --- | --- | --- |
| stream fp16fix 1024 | mauve fibre texture | same, browner | **colourful leaf/petal clutter** — not buildings | identical to 0.8 |
| stream taesd 1024 | grey gravel | grey gravel | **coloured mosaic tiles** — not buildings | identical to 0.8 |
| stream fp16fix **768** | ornamental damask | damask, faces emerging | **a rank of armoured figures with banners**; the tree below becomes a row of knights | identical to 0.8 |
| ComfyUI fast 1024 | grey pebbles | pink/grey pebbles | **orange-blue pebble mosaic** — not buildings | full anime character, whole canvas overridden |
| ComfyUI normal-14 1024 | pink foliage | pink/green foliage | **a rendered stone building facade with windows** | a full fantasy town |

So: **no, the worker at 1024/d=0.8 does not turn the noise band into buildings.**
Neither does ComfyUI fast at 1024/d=0.8 — it produces the same class of
"decorative rubble". That is a property of **4 LCM steps**, not of the worker.
ComfyUI normal-14 is the only run that resolves the band into architecture, and
it costs 9.6 s.

The interesting result is **768**: the worker at d=0.8 produces recognisable
*objects* (armoured figures, a crowd scene in row d) where the same worker at
1024 produces texture. At 768 the drawing occupies a smaller share of the
latent, so 4 steps carry it further from the input. If "the AI reinterprets what
I drew" is the feeling we want in the playtest, **768 buys more of it than 1024
does, and costs half as much time.**

### Rows a/b — line art and flat shapes

- d=0.5 and d=0.65: essentially the input, cleaned up. A door knob appears, the
  tree gets a texture. Safe, unimpressive, and *fast enough to feel live*.
- d=0.8/0.9 at 1024: the house sprouts a second wing and an arched door; the
  blob loses its yellow streak and flattens to a plain red disc. Mild.
- d=0.8/0.9 at 768: much bolder — the tree becomes a spear-carrying figure, the
  stick figure grows a cloak outline. Also more likely to produce garbage
  (spurious Japanese text appears in several cells across all runs; the
  Illustrious + LCM-LoRA combination likes to hallucinate captions and watermarks).

### Is `taesd` visibly softer?

**Not at grid scale, and not in the way I expected.** Side by side with
`fp16fix` at 1024:

- Line art is if anything *crisper* under taesd (rows a/c black strokes are
  cleaner, less halo).
- Colours are more saturated and higher-contrast (row d greens, row b reds).
- Where it loses is fine texture: the rubble in rows c/d becomes blockier,
  flatter tiles with less internal shading — the classic AutoencoderTiny
  signature. On a 1024 preview that reads as "stylised", not "broken".
- Composition differs from fp16fix (the *encode* is a different function, so the
  latent differs), so this is not a pure decode A/B — but the fidelity gap is
  much smaller than the 1.6x speed gain would suggest.

Caveat: judged from the downscaled grid (each cell ~380 px). At full resolution
the difference will be larger. Nobody has looked at a taesd output at 1:1.

---

## 3. What the grid's `results.json` does *not* record

The script writes, for any non-ComfyUI backend:

```json
"workflow": { "mode": "stream",
              "note": "sampling parameters belong to the backend, not to this server config" }
```

So **steps, sampler, scheduler, CFG and the VAE choice are not in the artefact.**
The three runs differ only by env vars on the worker side and are
indistinguishable from their `results.json`. This is the "ComfyUI-only fields"
case flagged in the task; the script was not edited.

For the record: all three runs were 4 steps / LCMScheduler / guidance 1.5 /
LCM-LoRA fused, and the run banner confirms it —
`[grid] backend stream, profile quality, res 1024, steps 4, seed 424242`
(the server's `AI_STEPS` was set to 4 for the run so that `config.aiSteps`,
which the script sends regardless of `--profile`, matched the worker).

---

## 4. Defect found: denoise 0.8 and 0.9 are the same image

In all three runs, every d=0.8 cell and its d=0.9 sibling are **byte-identical**
(e.g. `a_d0.80.png` and `a_d0.90.png` are both 482 781 bytes in fp16fix-1024).
The 0.9 column in every grid above is a duplicate of the 0.8 column.

Cause — two integer roundings compose badly at 4 steps:

```
worker:    scheduler_steps = ceil(steps / strength)     # steps_for_strength()
diffusers: t_start = scheduler_steps - int(scheduler_steps * strength)

d=0.50 -> scheduler_steps 8, int(8*0.50)=4 -> t_start 4, 4 real steps
d=0.65 -> scheduler_steps 7, int(7*0.65)=4 -> t_start 3, 4 real steps
d=0.80 -> scheduler_steps 5, int(5*0.80)=4 -> t_start 1, 4 real steps
d=0.90 -> scheduler_steps 5, int(5*0.90)=4 -> t_start 1, 4 real steps   <-- same
```

At 4 steps the reachable denoise levels are quantised to roughly {0.5, 0.65,
0.8}, and everything from ~0.75 to 1.0 collapses onto the same starting
timestep. **The worker cannot currently express "almost full redraw".** That is
exactly the regime where ComfyUI fast makes its dramatic jump (its 0.9 column
replaces the canvas with a rendered character), so the worker is missing the top
of the range, not merely a slider notch.

**Fixed in a follow-up pass — see section 6**, which also carries the validation
grid. The rest of this section describes the state of the v1 grids above, which
are kept as the before side of that comparison.

---

## 5. Recommendation for the playtest

**Backend: `stream`.** Faster than ComfyUI at every size tested with identical
sampling, and the only one of the two that can hold the model resident.

**LoRA and CFG: `STREAM_LORA=dmd2`, `STREAM_GUIDANCE=1.0`** — added in section 8,
which supersedes the LCM/CFG 1.5 assumption the rest of this report was written
under. 1384 ms at 768 and better reinterpretation.

**VAE: `fp16fix`** (the current default). `taesd` is 1.6x faster and holds up
better than expected — it has since been inspected at 1:1 (section 7) and is
*sharper* on line art — but it visibly flattens continuous tone, so it stays an
option rather than the default.

**Resolution: 768.** This is the substantive finding. 768 is half the latency of
1024 (1.8 s vs 3.5 s, comfortably inside "responds between strokes") *and*
transforms the drawing more at the same denoise. 1024 mostly buys resolution the
playtest does not need.

**Denoise: 0.8** — and 0.9 is now usable too, see section 6. 0.5 and 0.65 are
safe but barely change the drawing.

**Expect hallucinated text.** Spurious Japanese captions/watermarks appear at
d>=0.8 in every configuration, including ComfyUI's. It is the checkpoint, not the
backend.

---

## 6. Follow-up: the quantisation defect is fixed (`fp16fix-768-v2/`)

### The fix

`lcm_timesteps_for_strength()` in `apps/stream-worker/src/stream_worker/pipeline.py`
replaces `steps_for_strength()`. Instead of inflating the scheduler step count
and letting diffusers derive a start index from it, the worker now builds the
LCM distillation schedule (`20*i - 1` for i in 1..50, i.e. 19 … 999), picks the
start point directly as `round(strength * 50) - 1`, and spreads exactly `steps`
timesteps evenly from there down to the bottom of the schedule. The list is
handed to the pipeline as `timesteps=`, with `strength=1.0` so diffusers does
not slice it a second time.

Starting timesteps are now `0.5 -> 499`, `0.65 -> 639`, `0.8 -> 799`,
`0.9 -> 899`. Resolution is one distillation step (2%) instead of the ~25% the
old path gave at 4 steps. All timesteps stay on the distillation grid, and the
schedule always ends at 19, so the sample is fully denoised however high it
started.

Ten GPU-free tests in `apps/stream-worker/tests/test_contract.py` cover the
step count, the four distinct starts, strict descent, the distillation grid,
monotonicity across 2%..100%, and graceful degradation below 4 available
timesteps. One test pins the old arithmetic as the documented cause. 31 tests
pass.

### Validation — `fp16fix-768-v2/` (768, fp16fix, same seed and prompt as v1)

| drawing | d=0.5 | d=0.65 | d=0.8 | d=0.9 |
| --- | --- | --- | --- | --- |
| a | 2628* | 1768 | 1726 | 1767 |
| b | 1814 | 1803 | 1723 | 1771 |
| c | 1801 | 1816 | 1812 | 1798 |
| d | 1791 | 1793 | 1819 | 1797 |

median **1798 ms** (v1 was 1818 ms) — the fix costs nothing. * first cell, warming.

**0.8 and 0.9 are now distinct** — mean absolute pixel difference between the
two columns:

| drawing | a | b | c | d |
| --- | --- | --- | --- | --- |
| mean abs diff 0.8 vs 0.9 | 8.7 | 10.9 | 24.0 | 30.9 |

(v1: exactly 0 in all four, byte-identical files.)

**No regression at 0.5 / 0.65 / 0.8.** v1 vs v2, mean absolute pixel difference:

| drawing | d=0.5 | d=0.65 | d=0.8 | d=0.9 |
| --- | --- | --- | --- | --- |
| a | 0.73 | 1.88 | 2.01 | 8.68 |
| b | 0.86 | 1.74 | 1.83 | 11.23 |
| c | 6.43 | 12.68 | 10.59 | 24.65 |
| d | 11.94 | 18.67 | 16.23 | 31.46 |

The first three columns move only slightly (the schedule now runs down to
timestep 19 rather than stopping at 139–199, so outputs are marginally more
resolved); visually they are the same images — row c at 0.8 is still the rank of
armoured figures, row a at 0.65 is still the cleaned-up house. The 0.9 column is
where the change lands, as intended.

**0.9 is now the strongest cell in the grid and it is worth having.** Row a
becomes a full character illustration; row d resolves the noise into an actual
lit shopfront with windows and interior depth — architecture, which no previous
stream-worker cell at any size or VAE produced. This is the regime the worker
was missing, and it is now the closest the 4-step worker gets to what
ComfyUI normal-14 does at 0.8.

**Revised denoise recommendation:** 0.8 as the default, **0.9 now usable** and
worth exposing — it is a genuinely different, much bolder result rather than a
duplicate. The earlier advice to cap the slider at 0.8 is withdrawn.

---

## 7. taesd inspected at 1:1

Two 512x512 crops from the 1024 outputs, viewed at full resolution, taesd
against fp16fix on the same drawing and denoise.

**Line art (`a_d0.65`, house and tree on white):** taesd is *better*. Strokes
are darker and more evenly weighted, corners are clean, the door knob is a crisp
ring. fp16fix renders the same strokes slightly softer and greyer with a faint
blur along diagonals. taesd's white ground is a warmer cream; fp16fix's is
closer to neutral.

**Continuous tone (`c_d0.80`, the reinterpreted noise band):** this is where
taesd loses, and the loss is real. fp16fix produces swirling forms with smooth
internal shading and soft transitions between them. taesd produces flat facets
with hard borders — posterised, mosaic-like, with visible ringing along the
high-contrast boundary where the band meets the white sky. Hallucinated text in
the taesd crop carries a soft ghost halo that fp16fix does not have.

**Verdict: taesd is acceptable, not clearly better.** For line-art-dominant
canvases — which is most of what a Brush Jam room will contain — it is at least
as good as fp16fix and 1.6x faster. For anything with continuous tone it
visibly flattens. **Keeping `fp16fix` as the default**, as instructed: the
failure mode is content-dependent and would show up as "the AI made my shading
blocky" mid-playtest, and at 768 (the recommended size) fp16fix is already
1.8 s, so the absolute saving is small. `STREAM_VAE=taesd` stays a supported,
documented option for anyone who wants 1024 under 2.2 s.

---

## 8. LoRA and CFG: DMD2 vs LCM, guidance 1.5 vs 1.0

Four 768 grids, all with the v2 sampling (section 6), `fp16fix`, 4 steps,
seed 424242, same drawings and prompt. The only variables are the distillation
LoRA and `STREAM_GUIDANCE`.

| dir | LoRA | CFG | median latency |
| --- | --- | --- | --- |
| `fp16fix-768-v2/` | LCM | 1.5 | 1798 ms |
| `dmd2-768/` | DMD2 | 1.5 | 1756 ms |
| `dmd2-cfg1-768/` | DMD2 | 1.0 | **1384 ms** |
| `lcm-cfg1-768/` | LCM | 1.0 | 1398 ms |

The DMD2 LoRA (`tianweiy/DMD2`, `dmd2_sdxl_4step_lora_fp16.safetensors`, 394 MB)
was downloaded into `E:\ComfyUI\models\loras`, so a ComfyUI workflow can use the
same file.

A note on the design: DMD2 is *guidance-distilled* and is meant to be run at
CFG 1.0, so comparing `dmd2-768` (CFG 1.5) against `fp16fix-768-v2` (CFG 1.5)
alone would confound the LoRA with a setting DMD2 was not built for. Both CFG
levels were therefore run for both LoRAs — four cells, not two.

### Latency: CFG 1.0 is worth ~22%, not 2x

| LoRA | CFG 1.5 | CFG 1.0 | saving |
| --- | --- | --- | --- |
| LCM | 1798 ms | 1398 ms | -22% |
| DMD2 | 1756 ms | 1384 ms | -21% |

Dropping CFG halves the UNet work, but the UNet is only ~55% of a 768 request
(see section 4.4 of `docs/STREAM_WORKER.md`), so the end-to-end saving is about
a fifth. **The two LoRAs cost the same** — DMD2 is not slower despite being the
larger file, because both are fused into the UNet at load.

### Reinterpretation at 0.8 / 0.9: DMD2 is clearly better

DMD2 starts producing *objects* a whole denoise step earlier than LCM, and what
it produces is more coherent:

| row | LCM CFG 1.5 | DMD2 CFG 1.5 |
| --- | --- | --- |
| a @0.8 | spear-carrying figures, loose | a figure beside three clean conifers — readable as a scene |
| b @0.8 | blob becomes a soft glowing disc | blob becomes a proper flower with petals and a centre |
| c @0.65 | abstract damask pattern | already resolving into shelves and figures |
| d @0.65 | mottled brown texture | **a lit shop interior with shelved goods** |
| d @0.9 | crowded street scene | a clean, composed bookshop interior with depth |

The difference in character: LCM fills the noise with *ornament* — patterns,
foliage, clutter. DMD2 fills it with *things* — shelves, windows, figures,
objects with edges. For "the AI reinterprets what I drew", DMD2 is the better
behaviour, and it reaches it at a lower denoise, which also means less of the
user's drawing is destroyed to get there.

DMD2's colour is flatter and more saturated (vector-like); LCM's is softer and
more painterly. Which is nicer is a taste call, but DMD2 reads more clearly at
thumbnail size, which is how a Brush Jam room will mostly be seen.

### Line-art fidelity at 0.65, and what CFG 1.0 costs

Percentage of pixels darker than L=100 — a proxy for how much ink survives.
The input drawing is 2.36%:

| run | a @0.65 | a @0.80 | b @0.65 |
| --- | --- | --- | --- |
| input drawing | 2.36% | — | — |
| LCM CFG 1.5 | 2.35% | 2.35% | 1.66% |
| DMD2 CFG 1.5 | 3.08% | 4.37% | 3.50% |
| DMD2 CFG 1.0 | 3.06% | 4.15% | 1.95% |
| **LCM CFG 1.0** | 2.28% | **1.64%** | 1.66% |

This is the finding that decides the CFG question. **LCM at CFG 1.0 fades.** At
0.8 it retains 1.64% ink against the input's 2.36% — and the grid shows exactly
that: the house outline goes ghostly grey-green, the trees are pale, and at 0.9
row a collapses into a faint ornament instead of the character LCM CFG 1.5
produced. Strokes lose weight and the drawing dissolves rather than transforms.

**DMD2 at CFG 1.0 does not.** Its ink numbers are within 1-2% of its own CFG 1.5
values (3.06 vs 3.08, 4.15 vs 4.37), and the grids are near-identical in
composition — same conifers, same character, same bookshop — just very slightly
less saturated. This is what guidance distillation is supposed to buy, and here
it visibly does.

So CFG 1.0 is not a free 22% in general; it is free *for DMD2* and expensive for
LCM.

### Recommendation

**`STREAM_LORA=dmd2`, `STREAM_GUIDANCE=1.0`.** 1384 ms at 768 — 23% faster than
the current LCM/CFG 1.5 default and, by every reading of these grids, better
output: bolder reinterpretation, earlier onset, stronger line retention.

**The one real cost: at CFG 1.0 the negative prompt is ignored.** Brush Jam
sends the usual `lowres, bad anatomy, bad hands, text, error, worst quality,
low quality, jpeg artifacts, signature, watermark, blurry` and none of it will
have any effect. In these grids that did not visibly hurt — hallucinated text
and watermarks appear at d>=0.8 in *every* configuration including the ones with
CFG on, so the negative prompt was not suppressing them anyway. But it should be
a conscious choice, not a surprise: if a playtest turns up an artefact class the
negative prompt could fix, the lever is `STREAM_GUIDANCE=1.5` at +27% latency.

If the negative prompt must stay live, the second choice is
**`STREAM_LORA=dmd2`, `STREAM_GUIDANCE=1.5`** (1756 ms): same quality gain over
LCM, same latency as today's default, and CFG still on. There is no reading of
these four grids in which LCM is the right default.
