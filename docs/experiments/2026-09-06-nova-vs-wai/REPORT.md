# Nova Anime XL IL v19.0 vs WAI-illustrious v15.0

2026-09-06. Local RTX 3070 (8 GB), inproc backend, one model resident at a
time. Produced by `apps/brushjam/scripts/compare_checkpoints.py`.

Why: WAI's Civitai permissions exclude "Rent" (use on a generation service
other than Civitai), so the checkpoint is being replaced by Nova Anime XL
(Civitai 376130, version 2940478), which permits Rent and needs no credit.
This is the check that the swap costs nothing in speed or behaviour.

## Setup

Same as the 2026-09-05 stream experiment so the sheets line up: the four
synthetic drawings from `2026-09-05-stream/dmd2-768/2026-09-05/*_input.png`,
768², seed 424242, prompt "anime style, fantasy town, vibrant colors", the
standard negative prompt, full-image (no mask).

| profile | LoRA | steps | cfg |
|---|---|---|---|
| fast | DMD2 4-step, fused | 4 | 1.0 |
| quality | none (unfused) | 14 | 5.5 |

Denoise 0.65 and 0.80 for each. Both checkpoints loaded as fp16 UNet with
the fp16-fix VAE, 256 px VAE tiles.

## Speed (wall ms per 768² generation, 8 cells per profile)

| checkpoint | load | fast median | fast min | quality median | quality min | peak VRAM |
|---|---|---|---|---|---|---|
| Nova Anime XL IL v19.0 | 30.2 s | 1953 | 1718 | 5005 | 4756 | 6.98 GB |
| WAI-illustrious v15.0 | 32.4 s | 2064 | 1688 | 5071 | 4825 | 6.98 GB |

Identical within noise, as expected: same architecture, same LoRA, same VAE.
The first cell of each profile is slower (embedding cache miss plus the
profile switch).

## Look

`novaAnimeXL_ilV190_grid.jpg` and `waiNSFWIllustrious_v150_grid.jpg`, rows =
drawings a-d, columns = input, fast d0.65, fast d0.80, quality d0.65, quality
d0.80.

- **Line art is preserved by both** at every cell except fast d0.80, which
  reinterprets on both models. At fast d0.80 Nova kept the house as a house
  (row a) where WAI produced abstract shapes; on the noisy-sky drawing (row c)
  WAI drew a schematic with text labels, Nova drew a word-like shape. Neither
  is usable at that denoise; the profile default of 0.7 stands.
- **Fewer text and signature artifacts on Nova.** WAI put a signature-like
  string under row b fast d0.80 and row d quality d0.80; Nova produced one
  (row b fast d0.80). Both are within the range seen on 2026-09-05.
- **Colour.** Nova's quality outputs are higher-contrast and more saturated
  (rows c and d), consistent with the v19 release notes. On the noise inputs
  it settles into flatter, more graphic patterns; WAI's are busier. The
  drawing-plus-noise (row d) at quality d0.80 yields characters on both.
- Nothing in these sheets argues for a different denoise or step default.

## Verdict

Nova Anime XL IL v19.0 is a drop-in replacement for WAI v15 on this
pipeline: same speed, same VRAM, same behaviour across the denoise range, and
marginally cleaner at the top of it. `.env` on the dev machine now points
`INPROC_CHECKPOINT` at `novaAnimeXL_ilV190.safetensors`.

Files: per-cell PNGs under `novaAnimeXL_ilV190/` and
`waiNSFWIllustrious_v150/`, the two contact sheets, and `results.json` with
per-cell timings from the pipeline (`unet_ms`, `vae_*`, `prompt_ms`).
