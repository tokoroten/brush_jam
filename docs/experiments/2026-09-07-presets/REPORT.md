# Style presets: a look, not a picture of the medium

2026-09-07, Nova Anime XL IL v19 + DMD2 4-step LoRA, `fast` profile (4 steps,
CFG 1.0), 768, seed 424242, one generation per preset per variant.

The complaint: the style presets made the model *draw the medium*. Asking for
sumi-e produced a brush and a sheet of paper; stained glass produced a
cathedral window; claymation produced figurines on a table. The players'
drawing was replaced by a still life of the thing the style is made of.

## How this was measured

`apps/brushjam/scripts/preset_sheet.py` drives a running server as an ordinary
client - HTTP for the room, the upload and the result image, the WebSocket for
the protocol - and never loads a model of its own. One fixed input
(`docs/experiments/2026-09-05-stream/dmd2-768/2026-09-05/a_input.png`, a line
drawing of a house and a tree) is uploaded as a reference layer with
`includeInAI`, and each preset's prompt, negative and denoise are then set in
turn. One room per variant: the input never changes, the AI result is not fed
back as input, and twenty rooms would have eaten the room budget of a server
someone else was using.

The prompts are read out of `packages/shared/src/presets.ts`; the "before"
variant points the same script at the pre-change copy of that file.

- `sheet.png` - one row per preset: input, after, before.
- `probe.png`, `probe075_*` - four styles re-run at denoise 0.75.
- `results.json` - what was sent for every image, with latencies.

Images are gitignored; regenerate with

    uv run --no-sync --project apps/brushjam python         apps/brushjam/scripts/preset_sheet.py --variant after

## Per preset

| preset | before | after |
| --- | --- | --- |
| sumi-e | drawing kept, but ink marks float free of it; "black ink on white paper" is the wording that invites the objects | drawing kept, clean monochrome, no brush and no paper |
| ukiyo-e | **fail**: Mount Fuji and a wave pattern painted over the house - the scene nouns drew the famous print | drawing kept, flat colour, no Fuji |
| stained-glass | **fail**: house and tree replaced by a cathedral window with tracery | drawing kept; colour appears as small jewel-tone cells |
| pixel-art | house and tree redrawn as generic sprites | drawing kept, tree pixelated in place |
| watercolor-book | drawing kept, but this is the wording ("children's book illustration, storybook") that produced books in the user's own runs | drawing kept, soft watercolour blooms around the house |
| claymation | **fail**: a clay figure standing where the house was, two generic trees | drawing kept, matte rounded forms |
| papercraft | **fail**: the house rebuilt as a block structure on a slope, with conifer cut-outs | drawing kept, flat cut-out shapes |
| retro-poster | **fail**: replaced by garbled poster lettering - "typography, advertisement" is a request for text, and the text a model writes is never text | drawing kept, muted palette, no lettering |
| neon-city, horror, photoreal | unchanged prompts | pixel-identical to before at the same seed: a control that the harness is deterministic |

Every preset that used to draw the medium now keeps the players' composition.

## Denoise: the second half of the fix

Style presets dropped to 0.6 (sumi-e, watercolor-book, impressionist) and 0.65
(the rest); subjects stay at 0.8-0.85. `probe.png` is why: sumi-e, ukiyo-e,
stained glass and claymation re-run at 0.75 with the *new* prompts, where the
house turns into a robot-like block, spare trees appear, and stained glass
invents coloured cells of its own. The wording stops the model drawing the
medium; the denoise is what stops it drawing a different picture.

## What this does not fix

On `fast` the style is quiet. Four steps at CFG 1.0 over a sparse line drawing
at denoise 0.6-0.65 leaves very little room to restyle anything, so sumi-e and
ukiyo-e come back looking much like the input with cleaner lines. That is the
trade the user asked for - a preset that keeps the composition - and there are
two ways to spend more: the `quality` profile, where CFG is above 1 and the
negatives in these presets actually take effect (on `fast` the negative branch
is never evaluated at all), and a denser drawing, which gives the model more to
restyle than white space.
