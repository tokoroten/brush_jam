"""Create a synthetic 'drawing' PNG + a soft mask, like the server would send.

    uv run python scripts/make_sample.py --size 1024 --out samples/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def make_drawing(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    u = size / 1024.0  # every coordinate below is authored at 1024 and scaled

    # horizon + a couple of buildings, drawn as a human would rough them in
    d.line([(0, 640 * u), (size, 640 * u)], fill="black", width=int(5 * u) or 1)
    d.line([(180 * u, 640 * u), (180 * u, 330 * u), (430 * u, 330 * u), (430 * u, 640 * u)], fill="black", width=int(6 * u) or 1)
    d.line([(430 * u, 330 * u), (305 * u, 210 * u), (180 * u, 330 * u)], fill="black", width=int(6 * u) or 1)
    d.line([(560 * u, 640 * u), (560 * u, 420 * u), (760 * u, 420 * u), (760 * u, 640 * u)], fill="black", width=int(6 * u) or 1)
    d.line([(760 * u, 420 * u), (660 * u, 300 * u), (560 * u, 420 * u)], fill="black", width=int(6 * u) or 1)
    d.line([(820 * u, 640 * u), (880 * u, 250 * u), (940 * u, 640 * u)], fill="black", width=int(5 * u) or 1)
    d.line([(60 * u, 760 * u), (980 * u, 720 * u)], fill="black", width=int(4 * u) or 1)

    # a colored blob (a "sun" / balloon) and a green mass
    d.ellipse([700 * u, 90 * u, 860 * u, 250 * u], fill=(255, 170, 60))
    d.ellipse([90 * u, 500 * u, 260 * u, 660 * u], fill=(90, 170, 90))
    d.rectangle([300 * u, 800 * u, 520 * u, 900 * u], fill=(120, 150, 220))
    return img


def make_mask(size: int, feather: int = 32) -> Image.Image:
    """White rounded rect over the drawn area, feathered like the server's."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    inset = int(size * 0.06)
    d.rounded_rectangle([inset, inset, size - inset, size - inset], radius=int(size * 0.08), fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(1, int(feather * size / 1024))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("samples"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    make_drawing(args.size).save(args.out / f"input_{args.size}.png")
    make_mask(args.size).save(args.out / f"mask_{args.size}.png")
    print(f"wrote {args.out / f'input_{args.size}.png'} and mask")


if __name__ == "__main__":
    main()
