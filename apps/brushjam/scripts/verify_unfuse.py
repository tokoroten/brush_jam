"""Does unfusing the fast LoRA give the quality profile its model back?

The fast profile fuses the DMD2 LoRA into the UNet weights and the quality
profile unfuses it. That is only sound if unfusing restores the base weights
well enough that a quality generation is unchanged by having passed through a
fuse. This renders the same seed twice - once on a pipeline that has never
fused, once after a fuse/unfuse cycle - and reports the difference.

    uv run python scripts/verify_unfuse.py [--size 768] [--steps 14]

Needs the GPU to itself: it loads the model. Anything above ~50 dB PSNR is the
sampler's own nondeterminism rather than a weight difference; identical bytes
are possible but not required.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brushjam.ai.pipeline import InprocPipeline, PipelineSettings  # noqa: E402


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # A drawing-like input, so the comparison is on the kind of image this
    # actually generates rather than on flat white.
    base = Image.new("RGB", (args.size, args.size), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(base)
    s = args.size
    draw.line([(s * 0.05, s * 0.7), (s * 0.95, s * 0.72)], fill=(20, 20, 20), width=6)
    draw.rectangle([s * 0.3, s * 0.35, s * 0.6, s * 0.7], outline=(20, 20, 20), width=6)
    draw.line([(s * 0.28, s * 0.35), (s * 0.45, s * 0.18), (s * 0.62, s * 0.35)], fill=(20, 20, 20), width=6)

    pipeline = InprocPipeline(PipelineSettings())
    pipeline.load()
    # `load` selects the fast profile, which fuses. Undo that first so the
    # "before" render is a pipeline that has never been fused.
    pipeline._select_profile("quality")
    if pipeline._fused:
        print("! the pipeline is still fused after selecting quality", file=sys.stderr)
        return 1
    if pipeline._fuse_unavailable:
        print("! this build cannot fuse; nothing to verify", file=sys.stderr)
        return 1

    def render() -> Image.Image:
        return pipeline.generate(
            image=base,
            mask=None,
            prompt="anime style, fantasy town, vibrant colors",
            negative_prompt="lowres, bad anatomy, text, watermark",
            strength=0.8,
            steps=args.steps,
            seed=args.seed,
            width=args.size,
            height=args.size,
            profile="quality",
        ).image

    before = render()
    pipeline._select_profile("fast")
    assert pipeline._fused, "the fast profile did not fuse"
    print(f"fuse   took {pipeline._last_switch_ms:.0f} ms")
    pipeline._select_profile("quality")
    assert not pipeline._fused, "the quality profile did not unfuse"
    print(f"unfuse took {pipeline._last_switch_ms:.0f} ms")
    after = render()

    value = psnr(before, after)
    print(f"PSNR quality vs fuse->unfuse->quality: {value:.2f} dB")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        before.save(args.out / "quality-before-fuse.png")
        after.save(args.out / "quality-after-unfuse.png")
        print(f"wrote {args.out}")
    # 40 dB is already indistinguishable; below that the weights did not come
    # back and the fuse/unfuse design is wrong.
    return 0 if value >= 40.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
