"""Benchmark a running stream worker at several sizes.

    uv run python scripts/bench.py --sizes 512 768 1024 --runs 5

Writes one output PNG per size into --out and prints a markdown table.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sample import make_drawing, make_mask  # noqa: E402

PROMPT = "anime style, fantasy town, vibrant colors"
NEGATIVE = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, jpeg artifacts, signature, watermark, blurry"


def b64(img: Image.Image, mode: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=mode)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8790")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--denoise", type=float, default=0.55)
    ap.add_argument("--out", type=Path, default=Path("samples"))
    ap.add_argument("--mask", action="store_true", help="send a feathered mask too")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"{args.url}/healthz", timeout=30) as res:
        health = json.loads(res.read().decode("utf-8"))
    print("health:", json.dumps(health))

    rows = []
    for size in args.sizes:
        drawing = make_drawing(size)
        payload = {
            "image_b64": b64(drawing),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE,
            "denoise": args.denoise,
            "steps": args.steps,
            "seed": 12345,
            "width": size,
            "height": size,
        }
        if args.mask:
            payload["mask_b64"] = b64(make_mask(size))

        for _ in range(args.warmups):
            post(f"{args.url}/generate", payload)

        totals, diffusion = [], []
        last = None
        for i in range(args.runs):
            payload["seed"] = 12345 + i
            t0 = time.perf_counter()
            last = post(f"{args.url}/generate", payload)
            totals.append((time.perf_counter() - t0) * 1000.0)
            diffusion.append(last["timings"].get("diffusion_ms", 0.0))
            print(f"  {size}: run {i} wall {totals[-1]:.0f} ms  {json.dumps({k: round(v, 1) for k, v in last['timings'].items()})}")

        if last is not None:
            Image.open(io.BytesIO(base64.b64decode(last["image_b64"]))).save(args.out / f"bench_{size}.png")
            drawing.save(args.out / f"input_{size}.png")
        rows.append(
            {
                "size": size,
                "runs": args.runs,
                "wall_median_ms": round(statistics.median(totals)),
                "wall_min_ms": round(min(totals)),
                "wall_max_ms": round(max(totals)),
                "diffusion_median_ms": round(statistics.median(diffusion)),
            }
        )

    print()
    print("| size | runs | wall median | wall min | wall max | diffusion median |")
    print("| ---- | ---- | ----------- | -------- | -------- | ---------------- |")
    for r in rows:
        print(
            f"| {r['size']} | {r['runs']} | {r['wall_median_ms']} ms | {r['wall_min_ms']} ms | "
            f"{r['wall_max_ms']} ms | {r['diffusion_median_ms']} ms |"
        )
    (args.out / "bench.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
