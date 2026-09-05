"""Canvas rendering with Pillow + numpy. Port of apps/server/src/raster.ts and
the parts of packages/shared/src/render.ts the server uses.

Antialiasing parity with the browser is not required (this output only ever
reaches the model), but determinism is: the same strokes must produce the same
bytes on every run, and a noise stroke must produce the same pixels whatever
crop it is rendered through. Shapes are rasterised at SUPERSAMPLE x and box
filtered down, which is deterministic and close enough to a canvas2d stroke.
"""

from __future__ import annotations

import io
import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .constants import CANVAS_SIZE, MAX_STROKE_ALPHA, MIN_STROKE_ALPHA
from .geometry import Point, Rect
from .noise import fnv1a
from .protocol import Layer, Stroke
from .room import RenderSnapshot, RoomImage, strokes_for_crop

#: Shapes are drawn this many times oversized, then box filtered down.
SUPERSAMPLE = 4

_BYTES_PER_PIXEL = 4
#: Decoded images are far bigger than the bytes they came from: a 4096x4096 PNG
#: of one flat colour is a few KiB on the wire and 64 MiB decoded. The cache is
#: bounded by decoded pixels, not entries, and evicts least-recently-used first.
DECODED_PIXEL_BUDGET = (512 * 1024 * 1024) // _BYTES_PER_PIXEL

_image_cache: "OrderedDict[str, Tuple[Image.Image, int]]" = OrderedDict()
_cached_pixels = 0


def decoded_cache_stats() -> Dict[str, int]:
    return {"entries": len(_image_cache), "pixels": _cached_pixels}


def forget_images(ids: Iterable[str]) -> None:
    """Drop decoded images that are no longer referenced."""
    global _cached_pixels
    for image_id in list(ids):
        entry = _image_cache.pop(image_id, None)
        if entry is None:
            continue
        _cached_pixels -= entry[1]


def _evict_until_under_budget(budget: int = DECODED_PIXEL_BUDGET) -> None:
    global _cached_pixels
    while _cached_pixels > budget and _image_cache:
        _, entry = _image_cache.popitem(last=False)
        _cached_pixels -= entry[1]


def _decode(image_id: str, data: bytes) -> Image.Image:
    global _cached_pixels
    hit = _image_cache.get(image_id)
    if hit is not None:
        _image_cache.move_to_end(image_id)
        return hit[0]
    image = Image.open(io.BytesIO(data))
    image.load()
    image = image.convert("RGBA")
    pixels = image.width * image.height
    _image_cache[image_id] = (image, pixels)
    _cached_pixels += pixels
    _evict_until_under_budget()
    return image


def decode_upload(data: bytes, width: int, height: int) -> bool:
    """Decode an upload once, up front, and confirm it matches the header we
    already validated."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.width == width and image.height == height
    except Exception:
        return False


# --------------------------------------------------------------------------
# stroke rasterisation
# --------------------------------------------------------------------------


def _pressure_of(p: Point) -> float:
    raw = p.get("p")
    if isinstance(raw, (int, float)) and math.isfinite(raw):
        return max(0.05, min(1.0, float(raw)))
    return 1.0


def alpha_of(stroke: Stroke) -> float:
    """The eraser always removes fully; anything else is clamped into range."""
    if stroke.get("tool") == "eraser":
        return 1.0
    raw = stroke.get("alpha", 1)
    if not isinstance(raw, (int, float)) or not math.isfinite(raw):
        return 1.0
    return max(MIN_STROKE_ALPHA, min(MAX_STROKE_ALPHA, float(raw)))


def stroke_bounds(stroke: Stroke) -> Rect:
    """World-space bounding box of a stroke's painted area (render.ts)."""
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for p in stroke["points"]:
        r = max(0.5, (stroke["width"] * _pressure_of(p)) / 2) + 1
        min_x = min(min_x, p["x"] - r)
        min_y = min(min_y, p["y"] - r)
        max_x = max(max_x, p["x"] + r)
        max_y = max(max_y, p["y"] + r)
    return {
        "x": math.floor(min_x),
        "y": math.floor(min_y),
        "width": math.ceil(max_x - min_x),
        "height": math.ceil(max_y - min_y),
    }


def _temp_box(
    stroke: Stroke, offset_x: float, offset_y: float, bounds: Optional[Dict[str, int]]
) -> Optional[Tuple[int, int, int, int]]:
    """The temp raster a stroke is drawn into, in target space: (left, top, w, h)."""
    box = stroke_bounds(stroke)
    left = box["x"] - offset_x
    top = box["y"] - offset_y
    right = left + box["width"]
    bottom = top + box["height"]
    if bounds is not None:
        # Clipped to what the target can show, with a margin: a pixel's coverage
        # depends on the geometry around it, so cutting exactly at the edge
        # gives the boundary row a different value than an uncropped render.
        pad = math.ceil(stroke["width"]) + 2
        left = max(-pad, left)
        top = max(-pad, top)
        right = min(bounds["width"] + pad, right)
        bottom = min(bounds["height"] + pad, bottom)
    # Integral, because the temp raster is composited by array slicing rather
    # than by a drawImage that accepts a fractional destination.
    left_i = math.floor(left)
    top_i = math.floor(top)
    width = math.ceil(right - left_i)
    height = math.ceil(bottom - top_i)
    if width <= 0 or height <= 0:
        return None
    return left_i, top_i, width, height


def _rasterise_shape(stroke: Stroke, left: float, top: float, width: int, height: int) -> np.ndarray:
    """Antialiased coverage of the stroke shape, as float32 0..1 of shape (h, w).

    Segments are drawn as round-capped lines at their own pressure-scaled width,
    which is what the browser's lineCap/lineJoin 'round' produces.
    """
    s = SUPERSAMPLE
    img = Image.new("L", (width * s, height * s), 0)
    draw = ImageDraw.Draw(img)
    points = stroke["points"]

    def dot(px: float, py: float, radius: float) -> None:
        r = max(0.5, radius) * s
        cx = (px - left) * s
        cy = (py - top) * s
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    if len(points) == 1:
        p = points[0]
        dot(p["x"], p["y"], (stroke["width"] * _pressure_of(p)) / 2)
    else:
        for i in range(1, len(points)):
            a = points[i - 1]
            b = points[i]
            line_width = max(0.5, stroke["width"] * ((_pressure_of(a) + _pressure_of(b)) / 2))
            w = max(1, int(round(line_width * s)))
            draw.line(
                [
                    (a["x"] - left) * s,
                    (a["y"] - top) * s,
                    (b["x"] - left) * s,
                    (b["y"] - top) * s,
                ],
                fill=255,
                width=w,
            )
            # Round caps and joins: ImageDraw's line has butt ends.
            dot(a["x"], a["y"], line_width / 2)
            dot(b["x"], b["y"], line_width / 2)

    small = img.resize((width, height), Image.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


_MASK32 = np.uint64(0xFFFFFFFF)


def _noise_rgb_grid(seed: int, world_x: int, world_y: int, width: int, height: int) -> np.ndarray:
    """noiseRGB(seed, x, y) for a rectangle of world pixels, as uint8 (h, w, 3)."""
    xs = (np.arange(world_x, world_x + width, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
    ys = (np.arange(world_y, world_y + height, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
    hx = (xs * np.uint64(0x9E3779B1)) & _MASK32
    hy = (ys * np.uint64(0x85EBCA77)) & _MASK32
    h = (np.uint64(seed) ^ hx[None, :] ^ hy[:, None]) & _MASK32
    h = ((h ^ (h >> np.uint64(15))) * np.uint64(0x2C1B3C6D)) & _MASK32
    h = ((h ^ (h >> np.uint64(12))) * np.uint64(0x297A2D39)) & _MASK32
    h = (h ^ (h >> np.uint64(15))) & _MASK32
    out = np.empty((height, width, 3), dtype=np.uint8)
    out[:, :, 0] = (h & np.uint64(0xFF)).astype(np.uint8)
    out[:, :, 1] = ((h >> np.uint64(8)) & np.uint64(0xFF)).astype(np.uint8)
    out[:, :, 2] = ((h >> np.uint64(16)) & np.uint64(0xFF)).astype(np.uint8)
    return out


def _parse_color(color: str) -> Tuple[float, float, float]:
    if isinstance(color, str) and len(color) == 7 and color[0] == "#":
        try:
            return (
                float(int(color[1:3], 16)),
                float(int(color[3:5], 16)),
                float(int(color[5:7], 16)),
            )
        except ValueError:
            pass
    return (0.0, 0.0, 0.0)


class LayerRaster:
    """A premultiplied RGBA float buffer, so source-over and destination-out are
    both a multiply and an add."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.rgb = np.zeros((height, width, 3), dtype=np.float32)
        self.alpha = np.zeros((height, width), dtype=np.float32)

    def _clip(self, left: int, top: int, w: int, h: int):
        x0 = max(0, left)
        y0 = max(0, top)
        x1 = min(self.width, left + w)
        y1 = min(self.height, top + h)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1, x0 - left, y0 - top

    def source_over(self, left: int, top: int, rgb: np.ndarray, cover: np.ndarray) -> None:
        clipped = self._clip(left, top, cover.shape[1], cover.shape[0])
        if clipped is None:
            return
        x0, y0, x1, y1, sx, sy = clipped
        c = cover[sy : sy + (y1 - y0), sx : sx + (x1 - x0)][:, :, None]
        src = rgb[sy : sy + (y1 - y0), sx : sx + (x1 - x0)] if rgb.ndim == 3 else rgb
        inv = 1.0 - c
        self.rgb[y0:y1, x0:x1] = src * c + self.rgb[y0:y1, x0:x1] * inv
        self.alpha[y0:y1, x0:x1] = c[:, :, 0] + self.alpha[y0:y1, x0:x1] * inv[:, :, 0]

    def destination_out(self, left: int, top: int, cover: np.ndarray) -> None:
        clipped = self._clip(left, top, cover.shape[1], cover.shape[0])
        if clipped is None:
            return
        x0, y0, x1, y1, sx, sy = clipped
        inv = 1.0 - cover[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
        self.rgb[y0:y1, x0:x1] *= inv[:, :, None]
        self.alpha[y0:y1, x0:x1] *= inv


def render_strokes(
    target: LayerRaster,
    strokes: Sequence[Stroke],
    undone: Optional[set] = None,
    offset_x: float = 0,
    offset_y: float = 0,
) -> None:
    """Draw strokes onto a layer raster, in log order (render.ts)."""
    bounds = {"width": target.width, "height": target.height}
    for s in strokes:
        if undone and s["id"] in undone:
            continue
        if not s["points"]:
            continue
        box = _temp_box(s, offset_x, offset_y, bounds)
        if box is None:
            continue
        left, top, width, height = box
        # The shape is rasterised at full strength in its own box and
        # composited ONCE at the stroke's alpha, so a stroke that crosses
        # itself is one mark at one strength. The box is in target space, so
        # its origin in the stroke's own coordinates is offset by `offset`.
        shape = _rasterise_shape(s, left + offset_x, top + offset_y, width, height)
        alpha = alpha_of(s)
        if s["tool"] == "eraser":
            target.destination_out(left, top, shape)
            continue
        if s["tool"] == "noise":
            # Seeded by the stroke id and addressed in *world* coordinates, so a
            # translated render (the server's crop) yields identical pixels.
            world_x = round(left + offset_x)
            world_y = round(top + offset_y)
            rgb = _noise_rgb_grid(fnv1a(s["id"]), world_x, world_y, width, height).astype(
                np.float32
            )
            target.source_over(left, top, rgb, shape * alpha)
            continue
        colour = np.array(_parse_color(s["color"]), dtype=np.float32)
        rgb = np.broadcast_to(colour, (height, width, 3))
        target.source_over(left, top, rgb, shape * alpha)


# --------------------------------------------------------------------------
# AI input
# --------------------------------------------------------------------------


def render_crop_input(snapshot: RenderSnapshot, crop: Rect, size: int) -> bytes:
    """The AI input for a crop: white background, then every visible AI-input
    layer in order, resampled once at the end."""
    width = int(round(crop["width"]))
    height = int(round(crop["height"]))
    canvas = np.ones((height, width, 3), dtype=np.float32) * 255.0

    for layer in snapshot.layers:
        if not layer.get("visible") or (layer.get("opacity") or 0) <= 0:
            continue
        if not layer.get("includeInAI"):
            continue
        raster = LayerRaster(width, height)
        if layer["kind"] == "reference" and layer.get("imageId"):
            stored = snapshot.images.get(layer["imageId"])
            if stored is None:
                continue
            img = _decode(stored.id, stored.data)
            scale = layer.get("scale") or 1
            target_w = max(1, int(round(stored.width * scale)))
            target_h = max(1, int(round(stored.height * scale)))
            resized = img if (target_w, target_h) == img.size else img.resize(
                (target_w, target_h), Image.LANCZOS
            )
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            left = int(round((layer.get("x") or 0) - crop["x"]))
            top = int(round((layer.get("y") or 0) - crop["y"]))
            raster.source_over(left, top, arr[:, :, :3] * 255.0, arr[:, :, 3])
        else:
            dx = layer.get("offsetX") or 0
            dy = layer.get("offsetY") or 0
            render_strokes(
                raster,
                strokes_for_crop(snapshot, crop, layer["id"], {"x": dx, "y": dy}),
                undone=snapshot.undone,
                offset_x=crop["x"] - dx,
                offset_y=crop["y"] - dy,
            )
        opacity = float(layer.get("opacity") or 0)
        a = (raster.alpha * opacity)[:, :, None]
        canvas = raster.rgb * opacity + canvas * (1.0 - a)

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    if size != width or size != height:
        image = image.resize((size, size), Image.LANCZOS)
    return to_png(image)


def to_png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", compress_level=6)
    return out.getvalue()


class BuiltMask:
    """`png` is what the backend receives (white = regenerate); `alpha` is the
    same shape kept for local compositing."""

    def __init__(self, png: bytes, alpha: Image.Image, empty: bool) -> None:
        self.png = png
        self.alpha = alpha
        self.empty = empty


def build_full_mask(size: int) -> BuiltMask:
    """Full-canvas mode: everything is regenerated, so the mask is opaque."""
    alpha = Image.new("L", (size, size), 255)
    return BuiltMask(to_png(alpha.convert("RGB")), alpha, False)


class AICanvas:
    """The room's persistent AI canvas: a full-size, initially transparent raster."""

    def __init__(self, size: int = CANVAS_SIZE) -> None:
        self.size = size
        self.image = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    def composite(self, patch_png: bytes, crop: Rect, mask: Image.Image) -> bytes:
        """Composite an AI patch through the mask; returns the crop as a PNG."""
        with Image.open(io.BytesIO(patch_png)) as raw:
            raw.load()
            patch = raw.convert("RGBA")
        if patch.size != mask.size:
            patch = patch.resize(mask.size, Image.BICUBIC)
        patch.putalpha(mask)
        cw = int(round(crop["width"]))
        ch = int(round(crop["height"]))
        if patch.size != (cw, ch):
            # The patch is generation-sized; this scales it back to the canvas.
            patch = patch.resize((cw, ch), Image.BICUBIC)
        self.image.alpha_composite(patch, (int(crop["x"]), int(crop["y"])))
        box = (int(crop["x"]), int(crop["y"]), int(crop["x"]) + cw, int(crop["y"]) + ch)
        return to_png(self.image.crop(box))

    def to_png(self) -> bytes:
        return to_png(self.image)

    def clear(self) -> None:
        self.image = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
