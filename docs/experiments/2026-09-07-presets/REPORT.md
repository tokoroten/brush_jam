# Style presets: a look, not a picture of the medium

2026-09-07, Nova Anime XL IL v19 + DMD2 4-step LoRA, 768, seed 424242.
Profiles: `fast` = 4 steps at CFG 1.0 (~3s here), `quality` = the full sampler
with CFG above 1 (~10s here).

Two problems, found in that order.

1. The style presets made the model *draw the medium*. sumi-e produced a brush
   and a sheet of paper, stained glass a cathedral window, claymation figurines
   on a table, retro poster a sheet of garbled lettering. The players' drawing
   was replaced by a still life of the thing the style is made of.
2. The first fix - describe the medium as a quality, never as an object, and
   drop the denoise to 0.6-0.65 - cured that by producing no style at all. The
   pictures came back as the input line drawing with tidier lines.

## How this was measured

`apps/brushjam/scripts/preset_sheet.py` drives a running server as an ordinary
client - HTTP for the room, the upload and the result image, the WebSocket for
the protocol - and never loads a model of its own. A fixed drawing is uploaded
as a reference layer with `includeInAI`, then each preset's prompt, negative,
denoise and profile are set in turn. One room per input: the input never
changes, the AI result is not fed back as input, so the settings are the only
difference between two generations, and a room per cell would have eaten the
room budget of a server someone else was using. It backs off whenever the
server's generation counter moves without it, so a sweep does not compete with
somebody drawing.

Two inputs, both from the 2026-09-05 set: **a** = a sparse line drawing of a
house and a tree; **c** = the same drawing with a noise-pen band across the
sky, which is much closer to what a room looks like a minute into play.

- `sweep_<preset>.png` - denoise 0.65/0.7/0.75/0.8 down the rows, input x
  profile across the columns. 144 generations.
- `round2_<preset>.png` - four presets re-tested with one medium noun added
  back. 16 generations.
- `sheet.png` - the original before/after pass at fast, 22 generations.
- `sweep.json`, `round2.json`, `results.json` - what was sent for every image.

Images are gitignored; regenerate with

    uv run --no-sync --project apps/brushjam python \
        apps/brushjam/scripts/preset_sheet.py --variant chosen --profiles preset

## What the sweep says

**A look costs steps, not denoise.** On `fast` the sampler runs four steps at
CFG 1.0: the prompt barely steers and the negative branch is never evaluated at
all. Turning the denoise up there does not buy style, it buys a different
picture - at 0.75 the house became a robot, at 0.8 it became lettering, and on
the noisy input it became an anime face. On `quality` the same words land *and*
the composition survives all the way to 0.8, because the model is being guided
rather than left to wander. Every style that needed a look got `quality`; the
one that reads at CFG 1.0 anyway, pixel art, stayed on `fast`.

**The noise band is where the style shows.** On the sparse input a style has
almost nothing to work with - white paper stays white paper. On input c the
same settings turn the noise band into an ink wash, a wave-pattern sky, a
mosaic of glass cells, a plasticine surface, a blossom canopy. This is the
argument for judging presets on a drawing that resembles play.

**One medium noun, on `quality`, is safe and sometimes necessary.** Round 2 put
a single tag back into four presets. For stained glass it was the difference
between a few coloured dots and an actual mosaic - and no window appeared,
because on `quality` the negative that forbids it is finally evaluated. Clay
and papercraft gained their surface. sumi-e gained nothing and kept the round-1
wording.

## Chosen settings

| preset | profile | denoise | style visible | composition kept |
| --- | --- | --- | --- | --- |
| impressionist | quality | 0.80 | yes - painted roof, grass, foliage | yes |
| sumi-e | quality | 0.75 | yes - ink wash, inked line | yes |
| ukiyo-e | quality | 0.70 | yes - flat colour, wave/asanoha sky | yes (0.75 redraws the tree) |
| stained-glass | quality | 0.70 | yes - glass cells, lead lines (needs the tag) | yes |
| pixel-art | fast | 0.70 | yes - blocky shapes, dithered edges | yes |
| watercolor-book | quality | 0.70 | yes - washes and pastel bloom | yes |
| claymation | quality | 0.70 | yes - plasticine surface (needs "clay (medium)") | yes |
| papercraft | quality | 0.70 | yes - paper grain, flat cut shapes | yes |
| retro-poster | quality | 0.70 | yes - halftone, muted palette | yes (fast at 0.75 drew a poster) |

Style presets that keep `fast` at 0.6-0.65 are, on this checkpoint, presets
that do nothing.

## Still to render

The 18-cell confirmation pass - the nine presets above at their chosen setting,
on both inputs, in one sheet - had not run when the server was taken for a
restart. The sweep it would confirm is complete and its cells are in
`sweep.json` / `round2.json`; nothing else is outstanding.

## What this does not fix

The house and the tree themselves are line art in the input, and a style at
0.7-0.8 restyles them only so far: the pictures keep their black outlines. A
drawing with filled colour has more for a style to bite on than an outline
does, and the noise-pen band shows what the same settings do when there is
paint to work with.
