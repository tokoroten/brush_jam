# Experiments

Output of `pnpm --filter @brushjam/server quality-grid`, one directory per run
date (`YYYY-MM-DD`, local time).

The script renders four synthetic drawings with `@napi-rs/canvas` and the shared
`renderStrokes`, then sweeps denoise across them through whatever backend the
config selects (stream worker, ComfyUI or mock) by calling `generate()`
directly - no server, no WebSocket, fixed seed, full-white mask.

```
pnpm --filter @brushjam/server quality-grid            # 4 drawings x 4 denoise at 768
pnpm --filter @brushjam/server quality-grid --res 512 --drawings "a,c" --denoise "0.6,0.8"
AI_BACKEND=mock pnpm --filter @brushjam/server quality-grid   # instant, no GPU
```

Options: `--res` (default 768), `--out` (default `docs/experiments`),
`--drawings a,b,c,d`, `--denoise 0.5,0.65,0.8,0.9`.

Each run directory holds:

- `<drawing>_input.png` - the drawing as it was handed to the backend
- `<drawing>_d<denoise>.png` - one result per cell
- `grid.png` - contact sheet, rows = drawings, column 0 = input, then one
  column per denoise, 384 px cells
- `results.json` - backend, prompt, resolution, steps, cfg, seed and the
  per-cell latency

The drawings: (a) line-art house + tree, (b) stick figure + coloured blob,
(c) (a) with a wide noise stroke across the sky, (d) mostly noise with a few
pen lines.

**`2026-09-05-comfyui/REPORT.md` is the reference measurement** behind the
fast/quality profile defaults (denoise 0.7, fast at 768, quality at 1024).

`2026-09-05-comfyui/` is Experiment C: the real ComfyUI sweep on the local
RTX 3070 (normal 14-step and 4-step LCM, three resolutions) with latency logs
and `REPORT.md`. Contact sheets there are JPEG rather than PNG so three real
grids fit in under 2 MB.

`2026-09-05-mock/` is the mock-backend shakedown of the script (only the sheet
and `results.json` were kept). The mock ignores denoise, so its four columns are
identical by design - it proves the harness, not the model.
