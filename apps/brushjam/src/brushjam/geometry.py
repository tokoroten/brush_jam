"""Mirror of packages/shared/src/geometry.ts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, TypedDict


class Rect(TypedDict):
    x: float
    y: float
    width: float
    height: float


class Point(TypedDict, total=False):
    x: float
    y: float
    p: float


def rect(x: float, y: float, width: float, height: float) -> Rect:
    return {"x": x, "y": y, "width": width, "height": height}


def rect_right(r: Rect) -> float:
    return r["x"] + r["width"]


def rect_bottom(r: Rect) -> float:
    return r["y"] + r["height"]


def translate_rect(r: Rect, dx: float, dy: float) -> Rect:
    return {"x": r["x"] + dx, "y": r["y"] + dy, "width": r["width"], "height": r["height"]}


def expand_rect(r: Rect, by: float) -> Rect:
    return {"x": r["x"] - by, "y": r["y"] - by, "width": r["width"] + by * 2, "height": r["height"] + by * 2}


def rects_intersect(a: Rect, b: Rect) -> bool:
    return (
        a["x"] < rect_right(b)
        and b["x"] < rect_right(a)
        and a["y"] < rect_bottom(b)
        and b["y"] < rect_bottom(a)
    )


def union_rect(a: Rect, b: Rect) -> Rect:
    x = min(a["x"], b["x"])
    y = min(a["y"], b["y"])
    return {
        "x": x,
        "y": y,
        "width": max(rect_right(a), rect_right(b)) - x,
        "height": max(rect_bottom(a), rect_bottom(b)) - y,
    }


def union_rects(rects: Sequence[Rect]) -> Optional[Rect]:
    if not rects:
        return None
    acc = rects[0]
    for r in rects[1:]:
        acc = union_rect(acc, r)
    return acc


def intersect_rect(a: Rect, b: Rect) -> Optional[Rect]:
    x = max(a["x"], b["x"])
    y = max(a["y"], b["y"])
    r = min(rect_right(a), rect_right(b))
    bo = min(rect_bottom(a), rect_bottom(b))
    if r <= x or bo <= y:
        return None
    return {"x": x, "y": y, "width": r - x, "height": bo - y}


def clamp_rect_inside(r: Rect, bounds_width: float, bounds_height: float) -> Rect:
    width = min(r["width"], bounds_width)
    height = min(r["height"], bounds_height)
    x = min(max(r["x"], 0), bounds_width - width)
    y = min(max(r["y"], 0), bounds_height - height)
    return {"x": x, "y": y, "width": width, "height": height}


def round_rect_values(r: Rect) -> Rect:
    x = math.floor(r["x"])
    y = math.floor(r["y"])
    return {"x": x, "y": y, "width": math.ceil(rect_right(r)) - x, "height": math.ceil(rect_bottom(r)) - y}


def stroke_bbox(points: Sequence[Point], width: float) -> Rect:
    """Bounding box of a stroke, padded by half its width (plus 1 for AA)."""
    if not points:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    min_x = min(p["x"] for p in points)
    min_y = min(p["y"] for p in points)
    max_x = max(p["x"] for p in points)
    max_y = max(p["y"] for p in points)
    pad = width / 2 + 1
    return {
        "x": min_x - pad,
        "y": min_y - pad,
        "width": max_x - min_x + pad * 2,
        "height": max_y - min_y + pad * 2,
    }


def subtract_rect(a: Rect, b: Rect) -> List[Rect]:
    """`a` minus `b`, as up to four rects."""
    overlap = intersect_rect(a, b)
    if not overlap:
        return [a]
    out: List[Rect] = []
    if overlap["y"] > a["y"]:
        out.append({"x": a["x"], "y": a["y"], "width": a["width"], "height": overlap["y"] - a["y"]})
    if rect_bottom(overlap) < rect_bottom(a):
        out.append(
            {
                "x": a["x"],
                "y": rect_bottom(overlap),
                "width": a["width"],
                "height": rect_bottom(a) - rect_bottom(overlap),
            }
        )
    if overlap["x"] > a["x"]:
        out.append({"x": a["x"], "y": overlap["y"], "width": overlap["x"] - a["x"], "height": overlap["height"]})
    if rect_right(overlap) < rect_right(a):
        out.append(
            {
                "x": rect_right(overlap),
                "y": overlap["y"],
                "width": rect_right(a) - rect_right(overlap),
                "height": overlap["height"],
            }
        )
    return out
