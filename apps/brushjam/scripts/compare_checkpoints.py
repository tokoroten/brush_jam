"""Render the same drawings through two checkpoints, one model at a time.

    uv run --project apps/brushjam python apps/brushjam/scripts/compare_checkpoints.py \
        --checkpoint E:/ComfyUI/models/checkpoints/novaAnimeXL_ilV190.safetensors \
        --checkpoint E:/ComfyUI/models/checkpoints/waiNSFWIllustrious_v150.safetensors \
        --out docs/experiments/2026-09-06-nova-vs-wai

Each checkpoint is loaded, run through every (drawing, denoise, profile)
cell, and unloaded before the next one, so the card only ever holds one model
(the 8 GB rule). Inputs are the four synthetic drawings recorded by the
2026-09-05 stream experiment, so the results line up with those sheets.

Writes, per checkpoint, `<name>/<drawing>_<profile>_d<denoise>.png`, a
`grid.jpg` contact sheet (rows = drawings, column 0 = input, then one column
per (profile, denoise)), and a shared `results.json` with per-cell timings.
Needs the GPU to itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

INPUT_DIR = REPO_ROOT / "docs/experiments/2026-09-05-stream/dmd2-768/2026-09-05"
DRAWINGS = ["a", "b", "c", "d"]
PROMPT = "anime style, fantasy town, vibrant colors"
NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, "
    "jpeg artifacts, signature, watermark, blurry"
)
SEED = 424242


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def contact_sheet(
    cells: List[List[Image.Image]], labels: List[str], size: int = 384
) -> Image.Image:
    cols = max(len(row) for row in cells)
    sheet = Image.new("RGB", (cols * size, len(cells) * size + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(labels):
        draw.text((c * size + 4, 4), label, fill=(0, 0, 0))
    for r, row in enumerate(cells):
        for c, im in enumerate(row):
            sheet.paste(
                im.convert("RGB").resize((size, size)), (c * size, 24 + r * size)
            )
    return sheet


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--denoise", action="append", type=float, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    denoises = args.denoise or [0.65, 0.8]

    from brushjam.ai.pipeline import InprocPipeline, PipelineSettings

    inputs = {
        k: Image.open(INPUT_DIR / f"{k}_input.png").convert("RGB") for k in DRAWINGS
    }
    args.out.mkdir(parents=True, exist_ok=True)
    results: Dict[str, object] = {
        "date": time.strftime("%Y-%m-%d"),
        "prompt": PROMPT,
        "negativePrompt": NEGATIVE,
        "seed": args.seed,
        "resolution": args.size,
        "denoises": denoises,
        "profiles": {"fast": "dmd2 4 steps cfg 1.0", "quality": "14 steps cfg 5.5"},
        "checkpoints": {},
    }

    for ckpt in args.checkpoint:
        name = ckpt.stem
        out = args.out / name
        out.mkdir(exist_ok=True)
        settings = PipelineSettings(checkpoint=ckpt)
        pipeline = InprocPipeline(settings)
        t0 = time.perf_counter()
        pipeline.load()
        load_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[{name}] loaded in {load_ms:.0f} ms, {pipeline.describe_storage()}",
            flush=True,
        )
        cells: Dict[str, Dict[str, float]] = {}
        rows: List[List[Image.Image]] = []
        labels = ["input"]
        for key in DRAWINGS:
            row = [inputs[key]]
            for profile in ("fast", "quality"):
                for d in denoises:
                    label = f"{profile} d{d:.2f}"
                    if label not in labels:
                        labels.append(label)
                    steps = settings.steps_for(profile)
                    t = time.perf_counter()
                    result = pipeline.generate(
                        image=inputs[key],
                        mask=None,
                        prompt=PROMPT,
                        negative_prompt=NEGATIVE,
                        strength=d,
                        steps=steps,
                        seed=args.seed,
                        width=args.size,
                        height=args.size,
                        profile=profile,
                    )
                    wall = (time.perf_counter() - t) * 1000
                    fname = f"{key}_{profile}_d{d:.2f}.png"
                    result.image.save(out / fname)
                    cells[fname] = {
                        "wall_ms": round(wall),
                        **{k: round(v, 1) for k, v in result.timings.items()},
                    }
                    print(f"[{name}] {fname} {wall:.0f} ms", flush=True)
                    row.append(result.image)
            rows.append(row)
        contact_sheet(rows, labels).save(args.out / f"{name}_grid.jpg", quality=85)
        results["checkpoints"][name] = {
            "file": str(ckpt),
            "load_ms": round(load_ms),
            "storage": pipeline.describe_storage(),
            "memory": pipeline.memory(),
            "cells": cells,
        }
        pipeline.unload()
        del pipeline
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        (args.out / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
