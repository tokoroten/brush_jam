"""Hand port of apps/server/src/validate.ts.

The reducer is allowed to assume well-formed input; anything that fails here is
answered with an `error` frame. Error strings are copied verbatim so a client
(or a fixture exported from the Node server) sees exactly the same text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .constants import (
    AI_PROFILES,
    MAX_AI_RESOLUTION,
    MAX_DENOISE,
    MAX_NEGATIVE_PROMPT,
    MAX_STROKE_ALPHA,
    MIN_AI_RESOLUTION,
    MIN_DENOISE,
    MIN_STROKE_ALPHA,
)
from .geometry import Point

MAX_POINTS_PER_MESSAGE = 20_000


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    msg: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _bad(error: str) -> ValidationResult:
    return ValidationResult(ok=False, error=error)


def _is_obj(v: Any) -> bool:
    return isinstance(v, dict)


def _is_num(v: Any) -> bool:
    # JS: typeof v === 'number' && Number.isFinite(v). A JSON bool is never a
    # number, even though Python's bool subclasses int.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def _is_id(v: Any) -> bool:
    return _is_str(v) and 0 < len(v) <= 64


def _points(value: Any) -> Optional[List[Point]]:
    if not isinstance(value, list):
        return None
    if len(value) > MAX_POINTS_PER_MESSAGE:
        return None
    out: List[Point] = []
    for raw in value:
        if not _is_obj(raw) or not _is_num(raw.get("x")) or not _is_num(raw.get("y")):
            return None
        p: Point = {"x": raw["x"], "y": raw["y"]}
        # `p: undefined` and an absent `p` are the same thing in JS; `p: null`
        # is not a finite number and is rejected there too.
        if "p" in raw and raw["p"] is not None:
            if not _is_num(raw["p"]):
                return None
            p["p"] = raw["p"]
        elif "p" in raw:
            return None
        out.append(p)
    return out


LAYER_PATCH_KEYS = {
    "name",
    "visible",
    "locked",
    "opacity",
    "includeInAI",
    "x",
    "y",
    "scale",
    "offsetX",
    "offsetY",
}
_BOOL_KEYS = {"visible", "locked", "includeInAI"}
_NUM_KEYS = {"opacity", "x", "y", "scale", "offsetX", "offsetY"}


def _layer_patch(value: Any) -> Optional[Dict[str, Any]]:
    if not _is_obj(value):
        return None
    out: Dict[str, Any] = {}
    for key, v in value.items():
        if key not in LAYER_PATCH_KEYS:
            continue
        if key == "name" and not _is_str(v):
            return None
        if key in _BOOL_KEYS and not isinstance(v, bool):
            return None
        if key in _NUM_KEYS and not _is_num(v):
            return None
        out[key] = v
    return out


def validate_client_message(raw: Any) -> ValidationResult:
    if not _is_obj(raw):
        return _bad("message must be an object")
    t = raw.get("t")
    if not _is_str(t):
        return _bad("missing message type")

    if t == "cursor":
        if not _is_num(raw.get("x")) or not _is_num(raw.get("y")):
            return _bad("cursor needs finite x and y")
        return ValidationResult(True, {"t": "cursor", "x": raw["x"], "y": raw["y"]})

    if t == "stroke_start":
        s = raw.get("stroke")
        if not _is_obj(s):
            return _bad("stroke_start needs a stroke object")
        if not _is_id(s.get("id")):
            return _bad("stroke needs an id")
        if not _is_id(s.get("layerId")):
            return _bad("stroke needs a layerId")
        if s.get("tool") not in ("pen", "eraser", "noise"):
            return _bad("stroke tool must be pen, eraser or noise")
        if not _is_str(s.get("color")):
            return _bad("stroke needs a color")
        if not _is_num(s.get("width")):
            return _bad("stroke needs a finite width")
        pts = _points(s.get("points"))
        if pts is None:
            return _bad("stroke points are malformed or too many")
        has_alpha = "alpha" in s and s["alpha"] is not None
        if "alpha" in s and not has_alpha:
            return _bad("stroke alpha must be a finite number")
        if has_alpha and not _is_num(s["alpha"]):
            return _bad("stroke alpha must be a finite number")
        stroke: Dict[str, Any] = {
            "id": s["id"],
            "layerId": s["layerId"],
            "tool": s["tool"],
            "color": s["color"],
            "width": s["width"],
        }
        if has_alpha:
            stroke["alpha"] = max(MIN_STROKE_ALPHA, min(MAX_STROKE_ALPHA, s["alpha"]))
        stroke["points"] = pts
        return ValidationResult(True, {"t": "stroke_start", "stroke": stroke})

    if t in ("stroke_chunk", "stroke_end"):
        if not _is_id(raw.get("strokeId")):
            return _bad(f"{t} needs a strokeId")
        pts = _points(raw.get("points"))
        if pts is None:
            return _bad(f"{t} points are malformed or too many")
        return ValidationResult(True, {"t": t, "strokeId": raw["strokeId"], "points": pts})

    if t == "undo":
        return ValidationResult(True, {"t": "undo"})

    if t == "clear_layer":
        if not _is_id(raw.get("layerId")):
            return _bad("clear_layer needs a layerId")
        return ValidationResult(True, {"t": "clear_layer", "layerId": raw["layerId"]})

    if t == "layer_create":
        layer_raw = raw.get("layer")
        if not _is_obj(layer_raw):
            return _bad("layer_create needs a layer object")
        if layer_raw.get("kind") not in ("draw", "reference"):
            return _bad("layer kind must be draw or reference")
        patch = _layer_patch(layer_raw)
        if patch is None:
            return _bad("layer fields are malformed")
        image_id = layer_raw.get("imageId")
        # Presence, not non-None: Node tests `!== undefined`, so an explicit
        # `"imageId": null` is malformed there and must be malformed here.
        if "imageId" in layer_raw and not _is_id(image_id):
            return _bad("imageId must be a string")
        layer: Dict[str, Any] = dict(patch)
        layer["kind"] = layer_raw["kind"]
        if _is_id(image_id):
            layer["imageId"] = image_id
        return ValidationResult(True, {"t": "layer_create", "layer": layer})

    if t == "layer_update":
        if not _is_id(raw.get("id")):
            return _bad("layer_update needs an id")
        patch = _layer_patch(raw.get("patch"))
        if patch is None:
            return _bad("layer patch is malformed")
        return ValidationResult(True, {"t": "layer_update", "id": raw["id"], "patch": patch})

    if t == "layer_delete":
        if not _is_id(raw.get("id")):
            return _bad("layer_delete needs an id")
        return ValidationResult(True, {"t": "layer_delete", "id": raw["id"]})

    if t == "layer_reorder":
        ids = raw.get("ids")
        if (
            not isinstance(ids, list)
            or len(ids) == 0
            or len(ids) > 64
            or not all(_is_id(i) for i in ids)
        ):
            return _bad("layer_reorder needs an id list")
        if len(set(ids)) != len(ids):
            return _bad("layer_reorder ids must be unique")
        return ValidationResult(True, {"t": "layer_reorder", "ids": list(ids)})

    if t == "set_prompt":
        if not _is_str(raw.get("prompt")):
            return _bad("set_prompt needs a string prompt")
        return ValidationResult(True, {"t": "set_prompt", "prompt": raw["prompt"]})

    if t == "set_ai_settings":
        msg: Dict[str, Any] = {"t": "set_ai_settings"}
        if "denoise" in raw:
            if not _is_num(raw["denoise"]):
                return _bad("set_ai_settings denoise must be a finite number")
            if raw["denoise"] < MIN_DENOISE or raw["denoise"] > MAX_DENOISE:
                return _bad(
                    "set_ai_settings denoise must be between "
                    f"{_num_str(MIN_DENOISE)} and {_num_str(MAX_DENOISE)}"
                )
            msg["denoise"] = raw["denoise"]
        if "negativePrompt" in raw:
            if not _is_str(raw["negativePrompt"]):
                return _bad("set_ai_settings negativePrompt must be a string")
            if len(raw["negativePrompt"]) > MAX_NEGATIVE_PROMPT:
                return _bad("set_ai_settings negativePrompt is too long")
            msg["negativePrompt"] = raw["negativePrompt"]
        if "aiResolution" in raw:
            if not _is_num(raw["aiResolution"]):
                return _bad("set_ai_settings aiResolution must be a finite number")
            if raw["aiResolution"] < MIN_AI_RESOLUTION or raw["aiResolution"] > MAX_AI_RESOLUTION:
                return _bad(
                    "set_ai_settings aiResolution must be between "
                    f"{_num_str(MIN_AI_RESOLUTION)} and {_num_str(MAX_AI_RESOLUTION)}"
                )
            if raw["aiResolution"] % 64 != 0:
                return _bad("set_ai_settings aiResolution must be a multiple of 64")
            msg["aiResolution"] = raw["aiResolution"]
        if "aiProfile" in raw:
            if not _is_str(raw["aiProfile"]) or raw["aiProfile"] not in AI_PROFILES:
                return _bad(
                    f"set_ai_settings aiProfile must be one of {', '.join(AI_PROFILES)}"
                )
            msg["aiProfile"] = raw["aiProfile"]
        if len(msg) == 1:
            return _bad("set_ai_settings needs denoise, negativePrompt, aiResolution or aiProfile")
        return ValidationResult(True, msg)

    return _bad(f"unknown message type: {t[:32]}")


def _num_str(value: float) -> str:
    """Format a number the way JS string interpolation does (0.2, not 0.20)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)
