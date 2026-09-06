"""Canvas rendering with Pillow + numpy. Port of the retired Node server's src/raster.ts and
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
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

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

#: Rooms render on different worker threads and pruning runs on the event
#: loop, so every read, insert, eviction and removal happens under this.
_cache_lock = threading.Lock()
#: Immutable RGBA pixels, not PIL images: what the cache hands out is shared by
#: concurrent renders, and a shared mutable image is a bug waiting for a second
#: room.
_image_cache: "OrderedDict[str, Tuple[np.ndarray, int]]" = OrderedDict()
_cached_pixels = 0
#: Bumped whenever an id is forgotten, so a decode that started before the
#: forget knows its result is no longer wanted.
_forget_epoch: "Dict[str, int]" = {}


def decoded_cache_stats() -> Dict[str, int]:
    with _cache_lock:
        return {"entries": len(_image_cache), "pixels": _cached_pixels}


def forget_images(ids: Iterable[str]) -> None:
    """Drop decoded images that are no longer referenced."""
    global _cached_pixels
    with _cache_lock:
        for image_id in list(ids):
            # Recorded even when nothing is cached: a decode may be in flight
            # for this id right now, and its result must not be inserted after
            # the room that referenced it has gone.
            _forget_epoch[image_id] = _forget_epoch.get(image_id, 0) + 1
            entry = _image_cache.pop(image_id, None)
            if entry is None:
                continue
            _cached_pixels -= entry[1]
        if len(_forget_epoch) > 4096:
            for key in list(_forget_epoch):
                if key not in _image_cache:
                    del _forget_epoch[key]


def _evict_until_under_budget(budget: int = DECODED_PIXEL_BUDGET) -> None:
    """Caller holds `_cache_lock`."""
    global _cached_pixels
    while _cached_pixels > budget and _image_cache:
        _, entry = _image_cache.popitem(last=False)
        _cached_pixels -= entry[1]


def _decode(image_id: str, data: bytes) -> Image.Image:
    global _cached_pixels
    with _cache_lock:
        hit = _image_cache.get(image_id)
        if hit is not None:
            _image_cache.move_to_end(image_id)
            return Image.fromarray(hit[0], "RGBA")
        epoch = _forget_epoch.get(image_id, 0)

    # Decoding is slow and is deliberately done outside the lock: two threads
    # racing on the same id decode twice, and the accounting below makes that
    # cost one entry, not two.
    with Image.open(io.BytesIO(data)) as raw:
        raw.load()
        pixels_array = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
    pixels_array = np.ascontiguousarray(pixels_array)
    pixels_array.flags.writeable = False
    pixels = int(pixels_array.shape[0]) * int(pixels_array.shape[1])

    with _cache_lock:
        existing = _image_cache.get(image_id)
        if existing is not None:
            # Somebody else won the race; use theirs so the budget counts one.
            _image_cache.move_to_end(image_id)
            return Image.fromarray(existing[0], "RGBA")
        if _forget_epoch.get(image_id, 0) != epoch:
            # Forgotten while this decode was running. The caller still gets
            # its image - the array is theirs - but the cache does not keep a
            # copy of something nothing references any more.
            return Image.fromarray(pixels_array, "RGBA")
        _image_cache[image_id] = (pixels_array, pixels)
        _cached_pixels += pixels
        _evict_until_under_budget()
    return Image.fromarray(pixels_array, "RGBA")


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


def _js_round(v: float) -> int:
    """`Math.round`: half away from zero upwards, not Python's ties-to-even.

    `round(0.5)` is 0 in Python and 1 in JavaScript, and this decides which
    world pixel a noise stroke hashes from.
    """
    return math.floor(v + 0.5)


class TempBox(NamedTuple):
    """The temp raster a stroke is drawn into, in target space.

    `left`/`top` are the canvas's own fractional origin: the shared renderer
    creates its temp canvas here and hands the same number to drawImage.
    `logical_left`/`logical_top` are kept as the names the noise hash is
    addressed from, which is this same unrounded origin - rounding it shifts a
    fractionally translated noise layer onto the previous world pixel.
    """

    left: float
    top: float
    width: int
    height: int
    logical_left: float
    logical_top: float


def _temp_box(
    stroke: Stroke,
    offset_x: float,
    offset_y: float,
    bounds: Optional[Dict[str, int]],
    integral: bool = False,
) -> Optional[TempBox]:
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
    # Fractional, exactly as the shared renderer's temp canvas is: it is drawn
    # with a drawImage at this origin, which resamples. Flooring here first
    # rasterised the shape against a different pixel grid and then composited
    # it a fraction of a pixel away from where the browser puts it.
    #
    # `integral` is for the strokes the shared renderer draws straight onto the
    # target with no temp canvas at all (an opaque pen or eraser): there is no
    # drawImage and so no resampling, and the box is only Python's way of
    # rasterising a shape it composites by slicing.
    origin_x = math.floor(left) if integral else left
    origin_y = math.floor(top) if integral else top
    width = math.ceil(right - origin_x)
    height = math.ceil(bottom - origin_y)
    if width <= 0 or height <= 0:
        return None
    return TempBox(origin_x, origin_y, width, height, left, top)


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



def _place(
    left: float, top: float, cover: np.ndarray, rgb: Optional[np.ndarray] = None
) -> Tuple[int, int, np.ndarray, Optional[np.ndarray]]:
    """Move a temp raster from its fractional origin onto the pixel grid.

    `ctx.drawImage(temp, left, top)` with a fractional `left` does not snap:
    the canvas samples the temp bilinearly, in premultiplied colour, which
    blends each pixel with its neighbours. For a solid stroke that is a
    sub-pixel softening nobody would notice, but noise pixels are independent
    random values, so compositing at `floor(left)` instead put a *completely
    different* colour in every interior pixel - (227,29,100) where the browser
    has (166,63,55).

    Destination pixel `k` has its centre at `k + 0.5`, which lands at `k - f`
    in the temp, so it is `f * temp[k-1] + (1-f) * temp[k]` - one pixel wider
    than the source in each fractional direction. Outside the temp is
    transparent, which is what the canvas samples there too, since the temp's
    own border is transparent (the box is padded by a stroke width).
    """
    fx = left - math.floor(left)
    fy = top - math.floor(top)
    il, it = math.floor(left), math.floor(top)
    if fx == 0.0 and fy == 0.0:
        return il, it, cover, rgb

    def blend(a: np.ndarray) -> np.ndarray:
        pad_tail = ((0, 0),) * (a.ndim - 2)
        if fx:
            prev = np.pad(a, ((0, 0), (1, 0)) + pad_tail)
            cur = np.pad(a, ((0, 0), (0, 1)) + pad_tail)
            a = fx * prev + (1.0 - fx) * cur
        if fy:
            prev = np.pad(a, ((1, 0), (0, 0)) + pad_tail)
            cur = np.pad(a, ((0, 1), (0, 0)) + pad_tail)
            a = fy * prev + (1.0 - fy) * cur
        return a

    # Premultiplied, like the canvas: blending colour without its alpha would
    # drag the fully transparent border's colour into the edge pixels.
    alpha = blend(cover)
    if rgb is None:
        return il, it, np.clip(alpha, 0.0, 1.0), None
    src = np.broadcast_to(rgb, cover.shape + (3,)) if rgb.ndim < 3 else rgb
    premul = blend(src.astype(np.float32) * cover[:, :, None])

    # The canvas stores 8-bit premultiplied pixels and rounds half up. Skipping
    # this is a value or two out on every fractionally placed pixel, which is
    # exactly what a parity fixture compares.
    alpha8 = np.clip(np.floor(alpha * 255.0 + 0.5), 0.0, 255.0)
    premul8 = np.clip(np.floor(premul + 0.5), 0.0, 255.0)
    out_alpha = alpha8 / 255.0
    safe = np.where(alpha8 > 0, alpha8, 1.0)[:, :, None]
    out_rgb = np.clip(premul8 * 255.0 / safe, 0.0, 255.0)
    return il, it, out_alpha.astype(np.float32), out_rgb.astype(np.float32)


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
        # Which strokes the shared renderer puts through a temp canvas, and
        # therefore through a resampling drawImage: the noise pen always, and
        # anything translucent (it is flattened so overlaps cannot accumulate).
        # An opaque pen or eraser is stroked straight onto the target.
        alpha = alpha_of(s)
        resampled = s["tool"] == "noise" or alpha < 1
        box = _temp_box(s, offset_x, offset_y, bounds, integral=not resampled)
        if box is None:
            continue
        left, top, width, height = box.left, box.top, box.width, box.height
        # The shape is rasterised at full strength in its own box and
        # composited ONCE at the stroke's alpha, so a stroke that crosses
        # itself is one mark at one strength. The box is in target space, so
        # its origin in the stroke's own coordinates is offset by `offset`.
        shape = _rasterise_shape(s, left + offset_x, top + offset_y, width, height)
        if s["tool"] == "eraser":
            if resampled:
                il, it, cover, _ = _place(left, top, shape * alpha)
                target.destination_out(il, it, cover)
            else:
                target.destination_out(int(left), int(top), shape)
            continue
        if s["tool"] == "noise":
            # Seeded by the stroke id and addressed in *world* coordinates, so a
            # translated render (the server's crop) yields identical pixels.
            world_x = _js_round(box.logical_left + offset_x)
            world_y = _js_round(box.logical_top + offset_y)
            rgb = _noise_rgb_grid(fnv1a(s["id"]), world_x, world_y, width, height).astype(
                np.float32
            )
            il, it, cover, placed = _place(left, top, shape * alpha, rgb)
            target.source_over(il, it, placed if placed is not None else rgb, cover)
            continue
        colour = np.array(_parse_color(s["color"]), dtype=np.float32)
        rgb = np.broadcast_to(colour, (height, width, 3))
        if resampled:
            il, it, cover, placed = _place(left, top, shape * alpha, rgb)
            target.source_over(il, it, placed if placed is not None else rgb, cover)
        else:
            target.source_over(int(left), int(top), rgb, shape * alpha)


# --------------------------------------------------------------------------
# AI input
# --------------------------------------------------------------------------


def _reference_patch(
    img: Image.Image,
    target_w: int,
    target_h: int,
    left: int,
    top: int,
    crop_w: int,
    crop_h: int,
) -> Optional[Tuple[np.ndarray, int, int]]:
    """The part of a scaled reference image the crop can actually see.

    Scaling first and clipping afterwards is what a naive port does, and it is
    how a legal upload becomes an illegal allocation: a 4096x4096 PNG at the
    accepted scale of 8 is a 32768x32768 image, 16 GiB as float32 RGBA, for a
    crop that can show at most a megapixel of it. So the visible rectangle is
    computed first and only that is resampled, by handing Pillow the matching
    source box.

    The box is padded by the filter's support so the pixels that survive are
    identical to the ones a full resize would have produced - without the pad,
    the edge of the patch would sample a truncated neighbourhood and a crop
    would change the image it shows.
    """
    # Visible region in the scaled image's own pixels.
    vx0, vy0 = max(0, -left), max(0, -top)
    vx1, vy1 = min(target_w, crop_w - left), min(target_h, crop_h - top)
    if vx1 <= vx0 or vy1 <= vy0:
        return None

    sx = target_w / img.width
    sy = target_h / img.height
    # Pillow's filter support is 3 source pixels, stretched by the reduction
    # factor when downscaling; +1 covers the fractional box edges.
    pad_x = math.ceil(3 * max(sx, 1.0)) + 1
    pad_y = math.ceil(3 * max(sy, 1.0)) + 1
    px0, py0 = max(0, vx0 - pad_x), max(0, vy0 - pad_y)
    px1, py1 = min(target_w, vx1 + pad_x), min(target_h, vy1 + pad_y)

    if (target_w, target_h) == img.size:
        patch = img.crop((px0, py0, px1, py1))
    else:
        patch = img.resize(
            (px1 - px0, py1 - py0),
            Image.LANCZOS,
            box=(px0 / sx, py0 / sy, px1 / sx, py1 / sy),
        )
    arr = np.asarray(patch, dtype=np.float32) / 255.0
    return arr, left + px0, top + py0


# --------------------------------------------------------------------------
# incremental layer rasters
# --------------------------------------------------------------------------
#
# The AI input is rendered from the whole stroke log, every time. At a few
# hundred strokes that is nothing; at 3,600 it is most of a second, and a soak
# measured throughput falling from 41 generations a minute to 11 against a
# backend that returns instantly.
#
# Strokes are composited one at a time onto a layer's raster, in log order, so
# the state after stroke k is a prefix of the state after stroke n > k: keeping
# that buffer and drawing only what arrived since is the same sequence of
# operations on the same float32 buffer, and therefore the same bytes. That
# equality is what tests/test_incremental.py checks, against random sequences
# of pen, noise and eraser strokes with undo, clear and layer moves.
#
# What invalidates a cached prefix is any change to the strokes it consumed -
# an undo inside it, a cleared layer, an evicted stroke - and that is detected
# by keeping their ids: the layer's new list has to *start with* the cached
# one, or the layer is rebuilt from scratch, which costs exactly what every
# render cost before this existed.


def _layer_raster_bytes(width: int, height: int) -> int:
    """rgb float32 (3 channels) + alpha float32. 16.8 MB at 1024.

    Kept in float32 rather than packed to 8-bit RGBA because the point is to
    continue the *exact* buffer: rounding it to bytes would make an incremental
    render differ from a from-scratch one, which is the one thing it must not.
    """
    return width * height * 4 * 4


#: Total the layer caches may hold across every room - sixteen layers at 1024.
#: Beyond it the least recently used layer is dropped and rendered from scratch
#: next time: slower, never wrong.
LAYER_CACHE_BUDGET_BYTES = 16 * _layer_raster_bytes(CANVAS_SIZE, CANVAS_SIZE)


class _CachedLayer:
    """One draw layer's committed strokes, and which ones they were."""

    __slots__ = ("raster", "strokes", "offset", "nbytes")

    def __init__(
        self, raster: "LayerRaster", strokes: List[Stroke], offset: Tuple[float, float]
    ) -> None:
        self.raster = raster
        #: The stroke records composited into `raster`, in order - the very
        #: objects the room's log holds, not their ids. A committed stroke is
        #: appended once and never mutated, so identity IS its content, and
        #: comparing a 3,600-long prefix is a pointer walk.
        #:
        #: Ids were the obvious key and were wrong: `clear_layer` drops a
        #: layer's strokes from the log, which frees their ids for reuse, so
        #: the same id could come back carrying a different drawing and the
        #: cache would hand back the old pixels.
        self.strokes = strokes
        #: The layer translation this raster was drawn with. A moved layer is
        #: re-rendered rather than shifted: at a fractional offset the strokes
        #: are rasterised at a different sub-pixel phase, so shifting finished
        #: pixels would not produce the image a full render produces.
        self.offset = offset
        self.nbytes = raster.rgb.nbytes + raster.alpha.nbytes


def _is_prefix(cached: Sequence[Stroke], current: Sequence[Stroke]) -> bool:
    """Is every cached stroke still there, in the same place, still itself?"""
    n = len(cached)
    if n > len(current):
        return False
    if n and cached[n - 1] is not current[n - 1]:
        return False  # the common case of a changed prefix, in one comparison
    for a, b in zip(cached, current):
        if a is not b:
            return False
    return True


class LayerCacheStore:
    """Cached layer rasters for every room, under one memory budget.

    Keyed by (room, layer). Rooms come and go and a room can hold eight layers,
    so the bound that matters is the total: an LRU over entries, evicted to fit
    `budget_bytes`.
    """

    def __init__(self, budget_bytes: int = LAYER_CACHE_BUDGET_BYTES) -> None:
        self.budget_bytes = budget_bytes
        self._entries: "OrderedDict[Tuple[str, str], _CachedLayer]" = OrderedDict()
        self._bytes = 0
        #: How many times each key has been invalidated. A render reads it when
        #: it starts and again when it installs its result: a `clear_layer`
        #: that arrives while a render is running must not be undone by that
        #: render storing the raster it began before the clear.
        self._generation: "Dict[Tuple[str, str], int]" = {}
        #: Renders run on worker threads - one per room at a time, but several
        #: rooms at once - and eviction touches every room's entries.
        self._lock = threading.Lock()

    # -- accounting -------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"layers": len(self._entries), "bytes": self._bytes}

    def _evict_until_under_budget(self) -> None:
        """Caller holds the lock."""
        while self._bytes > self.budget_bytes and self._entries:
            key, entry = self._entries.popitem(last=False)
            self._bytes -= entry.nbytes
            self._generation[key] = self._generation.get(key, 0) + 1

    def _drop(self, key: Tuple[str, str]) -> bool:
        """Caller holds the lock. Bumps the generation whether or not anything
        is cached: what is being invalidated may be mid-render instead."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry.nbytes
        self._generation[key] = self._generation.get(key, 0) + 1
        return entry is not None

    def forget_layer(self, room_id: str, layer_id: str) -> None:
        with self._lock:
            self._drop((room_id, layer_id))

    def forget_room(self, room_id: str) -> int:
        """Drop every layer of one room: it was evicted, or has gone quiet."""
        dropped = 0
        with self._lock:
            for key in [k for k in self._entries if k[0] == room_id]:
                self._drop(key)
                dropped += 1
        return dropped

    def clear(self) -> None:
        with self._lock:
            for key in list(self._entries):
                self._drop(key)
            self._entries.clear()
            self._bytes = 0

    # -- rendering --------------------------------------------------------

    def render_layer(
        self,
        room_id: Optional[str],
        layer_id: str,
        strokes: Sequence[Stroke],
        width: int,
        height: int,
        offset_x: float,
        offset_y: float,
    ) -> "LayerRaster":
        """This layer's committed strokes, drawn on from wherever we left off.

        `strokes` is the layer's log in order, already filtered - undone
        strokes are not in it. The result belongs to the cache: composite it,
        do not modify it.
        """
        key = None if room_id is None else (room_id, layer_id)
        offset = (float(offset_x), float(offset_y))
        kept = list(strokes)

        generation = 0
        if key is not None:
            reusable = None
            with self._lock:
                generation = self._generation.get(key, 0)
                entry = self._entries.get(key)
                if entry is not None and (
                    entry.offset == offset
                    and entry.raster.width == width
                    and entry.raster.height == height
                    and _is_prefix(entry.strokes, kept)
                ):
                    self._entries.move_to_end(key)
                    # Claimed here, drawn below: the lock covers the table, not
                    # the drawing. Holding it across a 100 ms render would make
                    # every room in the process wait for one room's strokes.
                    # Two renders of the SAME layer at once would race, and
                    # cannot happen: a room has one generation in flight.
                    reusable = (entry.raster, len(entry.strokes))
                    entry.strokes = kept
                elif entry is not None:
                    # The prefix is gone: an undo inside it, a cleared layer, a
                    # moved one, or strokes that expired. Rebuild this layer,
                    # and only this layer.
                    self._drop(key)
                    generation = self._generation[key]
            if reusable is not None:
                raster, start = reusable
                render_strokes(raster, kept[start:], offset_x=offset_x, offset_y=offset_y)
                return raster

        raster = LayerRaster(width, height)
        render_strokes(raster, kept, offset_x=offset_x, offset_y=offset_y)
        if key is not None:
            fresh = _CachedLayer(raster, kept, offset)
            with self._lock:
                # Anything that invalidated this layer while it was being drawn
                # invalidates this raster too: it was rendered from strokes read
                # before that happened. Hand it back, do not keep it.
                if self._generation.get(key, 0) == generation:
                    previous = self._entries.pop(key, None)
                    if previous is not None:
                        self._bytes -= previous.nbytes
                    self._entries[key] = fresh
                    self._bytes += fresh.nbytes
                    self._evict_until_under_budget()
        return raster


#: The store every room shares. One budget, one place to look.
LAYER_CACHE = LayerCacheStore()


def layer_cache_stats() -> Dict[str, int]:
    return LAYER_CACHE.stats()


def forget_layer_rasters(room_id: str, layer_id: Optional[str] = None) -> int:
    """Drop one layer's cached raster, or the whole room's.

    Called when a layer is deleted, when a room is evicted, and when a room has
    been empty long enough that holding 16 MB a layer for it stops being worth
    the render it saves.
    """
    if layer_id is not None:
        LAYER_CACHE.forget_layer(room_id, layer_id)
        return 1
    return LAYER_CACHE.forget_room(room_id)


def render_crop_input(
    snapshot: RenderSnapshot, crop: Rect, size: int, room_id: Optional[str] = None
) -> bytes:
    """The AI input for a crop: white background, then every visible AI-input
    layer in order, resampled once at the end.

    With a `room_id`, each draw layer's committed strokes come from that
    room's cached raster and only what arrived since is drawn (see
    LayerCacheStore).
    Without one - a crop that is not the whole canvas, a caller that has no
    room - every stroke is drawn, which is what this always did.
    """
    width = int(round(crop["width"]))
    height = int(round(crop["height"]))
    canvas = np.ones((height, width, 3), dtype=np.float32) * 255.0

    for layer in snapshot.layers:
        if not layer.get("visible") or (layer.get("opacity") or 0) <= 0:
            continue
        if not layer.get("includeInAI"):
            continue
        if layer["kind"] == "reference" and layer.get("imageId"):
            raster = LayerRaster(width, height)
            stored = snapshot.images.get(layer["imageId"])
            if stored is None:
                continue
            img = _decode(stored.id, stored.data)
            scale = layer.get("scale") or 1
            target_w = max(1, int(round(stored.width * scale)))
            target_h = max(1, int(round(stored.height * scale)))
            left = int(round((layer.get("x") or 0) - crop["x"]))
            top = int(round((layer.get("y") or 0) - crop["y"]))
            placed = _reference_patch(img, target_w, target_h, left, top, width, height)
            if placed is None:
                continue
            arr, left, top = placed
            raster.source_over(left, top, arr[:, :, :3] * 255.0, arr[:, :, 3])
        else:
            dx = layer.get("offsetX") or 0
            dy = layer.get("offsetY") or 0
            # `strokes_for_crop` already drops undone strokes, so what it
            # returns IS what the layer shows - which is what makes it a
            # prefix the cache can be checked against.
            raster = LAYER_CACHE.render_layer(
                room_id,
                layer["id"],
                strokes_for_crop(snapshot, crop, layer["id"], {"x": dx, "y": dy}),
                width,
                height,
                offset_x=crop["x"] - dx,
                offset_y=crop["y"] - dy,
            )
        # Opacity and visibility are applied here, on the way into the canvas,
        # so changing either costs nothing and invalidates nothing.
        opacity = float(layer.get("opacity") or 0)
        a = (raster.alpha * opacity)[:, :, None]
        canvas = raster.rgb * opacity + canvas * (1.0 - a)

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    if size != width or size != height:
        image = image.resize((size, size), Image.LANCZOS)
    return to_png(image, AI_INPUT_COMPRESS_LEVEL)


#: The AI input never leaves this process, so it is encoded for speed, not for
#: size: level 1 is ~5x cheaper than level 6 and the bytes are handed straight
#: to the backend. Level 6 stays the default for anything a client downloads.
AI_INPUT_COMPRESS_LEVEL = 1


def to_png(image: Image.Image, compress_level: int = 6) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", compress_level=compress_level)
    return out.getvalue()


class BuiltMask:
    """`png` is what the backend receives (white = regenerate); `alpha` is the
    same shape kept for local compositing."""

    def __init__(self, png: bytes, alpha: Image.Image, empty: bool) -> None:
        self.png = png
        self.alpha = alpha
        self.empty = empty


@lru_cache(maxsize=8)
def _full_mask(size: int) -> BuiltMask:
    alpha = Image.new("L", (size, size), 255)
    return BuiltMask(to_png(alpha.convert("RGB"), AI_INPUT_COMPRESS_LEVEL), alpha, False)


def build_full_mask(size: int) -> BuiltMask:
    """Full-canvas mode: everything is regenerated, so the mask is opaque - and
    therefore identical for every run at a given size, so it is built once. The
    alpha is only ever read (`putalpha`, never drawn into), so sharing it is
    safe; the PNG is immutable bytes."""
    return _full_mask(size)


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

    def to_jpeg(self, quality: int = 90) -> bytes:
        """The canvas as a JPEG, for the saved history.

        JPEG has no alpha, and the canvas has plenty: anything the model has
        not painted yet is transparent, which would come out black. It is
        flattened onto white, the same background the drawing is composited on.
        """
        flat = Image.new("RGB", self.image.size, (255, 255, 255))
        flat.paste(self.image, (0, 0), self.image)
        out = io.BytesIO()
        flat.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()

    def clear(self) -> None:
        self.image = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
