"""Replay what the Node implementation actually did.

The fixtures were written by the retired Node server's export-fixtures script,
driving its real modules - the validator, the reducer, the noise hash and the
shared renderer through a canvas. That server is gone, so these are no longer a
parity oracle; they are the record of the behaviour this one was built to
match, and every run replays all of it. They are frozen: nothing can regenerate
them, which is the point.
"""

from __future__ import annotations

import math

import json
import re
from typing import Any, Dict, List

import pytest

from brushjam.noise import fnv1a, noise_rgb
from brushjam.room import (
    RoomLimits,
    apply_client_message,
    create_room,
    join_member,
    snapshot,
    sorted_layers,
)
from brushjam.validate import validate_client_message

from conftest import load_fixture


# ------------------------------------------------------------------ protocol


def _protocol_cases():
    data = load_fixture("protocol-samples.json")
    return [(s["type"], s["input"], s["result"]) for s in data["samples"]]


@pytest.mark.parametrize("kind,payload,expected", _protocol_cases())
def test_validator_matches_node(kind: str, payload: Any, expected: Dict[str, Any]) -> None:
    result = validate_client_message(payload)
    if expected["ok"]:
        assert result.ok, f"{kind}: expected accept, got {result.error}"
        assert result.msg == expected["msg"]
    else:
        assert not result.ok, f"{kind}: expected reject, got {result.msg}"
        assert result.error == expected["error"]


# --------------------------------------------------------------------- noise


def test_fnv1a_matches_node() -> None:
    data = load_fixture("noise-samples.json")
    for case in data["fnv1a"]:
        assert fnv1a(case["text"]) == case["seed"], case["text"]


def test_noise_rgb_matches_node() -> None:
    data = load_fixture("noise-samples.json")
    for case in data["rgb"]:
        assert list(noise_rgb(case["seed"], case["x"], case["y"])) == case["rgb"], case


# ------------------------------------------------------------------- reducer


def _normalise(value: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for real, placeholder in mapping.items():
            out = out.replace(real, placeholder)
        return out
    if isinstance(value, list):
        return [_normalise(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v, mapping) for k, v in value.items()}
    return value


def _resolve(msg: Any, created: List[str]) -> Any:
    """`@layerN` in the script means "the Nth layer ever created in this room"."""
    text = json.dumps(msg)
    text = re.sub(
        r'"@layer(\d+)"',
        lambda m: json.dumps(created[int(m.group(1))] if int(m.group(1)) < len(created) else "missing"),
        text,
    )
    return json.loads(text)


def test_reducer_trace_matches_node() -> None:
    data = load_fixture("reducer-trace.json")
    spec = data["room"]
    room = create_room(
        spec["id"],
        spec["denoise"],
        spec["canvasSize"],
        spec["resolution"],
        spec["adjustable"],
        spec["profile"],
        RoomLimits(
            profiles=["fast", "quality"],
            max_denoise=0.95,
            max_resolution=1024,
            negative_prompt_active={"fast": True, "quality": True},
        ),
        seed=spec["seed"],
    )
    users = [
        join_member(room, "Alice", "tok-alice-0001")["userId"],
        join_member(room, "Bob", "tok-bob-0001")["userId"],
    ]
    created: List[str] = [l["id"] for l in sorted_layers(room)]

    for index, step in enumerate(data["steps"]):
        for layer in sorted_layers(room):
            if layer["id"] not in created:
                created.append(layer["id"])
        resolved = _resolve(step["msg"], created)
        # The fixture keeps the placeholders, so resolve them the same way.
        validated = validate_client_message(resolved)
        mapping = {users[i]: f"U{i}" for i in range(len(users))}
        mapping.update({layer_id: f"L{i}" for i, layer_id in enumerate(created)})

        if "rejected" in step:
            assert not validated.ok, f"step {index}: expected a rejection"
            assert validated.error == step["rejected"]
            continue

        assert validated.ok, f"step {index}: {validated.error}"
        result = apply_client_message(room, users[step["by"]], validated.msg)
        after = snapshot(
            room, users[step["by"]], "idle", {"window": 768, "apply": 1024, "canvasSize": 1024}
        )
        for layer in sorted_layers(room):
            if layer["id"] not in created:
                created.append(layer["id"])
        mapping = {users[i]: f"U{i}" for i in range(len(users))}
        mapping.update({layer_id: f"L{i}" for i, layer_id in enumerate(created)})

        actual = _normalise(
            {
                "broadcast": result.broadcast,
                "relay": result.relay,
                "toSender": result.to_sender,
                "dirty": result.dirty,
                "promptChanged": result.prompt_changed,
            },
            mapping,
        )
        assert json.loads(json.dumps(actual)) == step["result"], f"step {index} ({step['msg']})"
        assert json.loads(json.dumps(_normalise(after, mapping))) == step["snapshot"], (
            f"step {index} snapshot ({step['msg']})"
        )


# ----------------------------------------------------------- noise placement


def test_noise_hashes_from_the_same_world_pixel_as_the_browser() -> None:
    """Where a noise stroke's texture starts, for fractional layer offsets.

    The fixture is produced by driving the real shared renderer, so this is the
    origin the browser uses, not a restatement of it. Two things it pins down:
    the temp raster's origin must stay unrounded for hashing (flooring it first
    shifts the texture onto the previous world pixel), and the rounding must be
    `Math.round`, not Python's ties-to-even.
    """
    from brushjam.raster import _js_round, _temp_box

    cases: List[Dict[str, Any]] = load_fixture("noise-placement.json")["cases"]
    assert len(cases) > 100
    ties = 0
    for case in cases:
        stroke = {
            "id": case["strokeId"],
            "tool": "noise",
            "color": "#000000",
            "width": case["width"],
            "points": case["points"],
        }
        box = _temp_box(stroke, case["offsetX"], case["offsetY"], case["bounds"])
        assert box is not None, case
        assert box.logical_left == case["logicalLeft"], case
        assert box.logical_top == case["logicalTop"], case

        world_x = _js_round(box.logical_left + case["offsetX"])
        world_y = _js_round(box.logical_top + case["offsetY"])
        assert list(noise_rgb(fnv1a(case["strokeId"]), world_x, world_y)) == case["originRGB"], case

        if (world_x, world_y) != (
            round(box.logical_left + case["offsetX"]),
            round(box.logical_top + case["offsetY"]),
        ):
            ties += 1
    # If the fixture stopped covering exact .5 sums, the rounding rule would be
    # untested and this would silently pass.
    assert ties > 0, "no case exercises the Math.round / round() difference"


def test_a_fractionally_placed_stroke_matches_the_browsers_pixels() -> None:
    """Not just where the noise starts - what the target actually holds.

    The shared renderer draws its temp canvas at a *fractional* origin, so the
    canvas resamples it: every interior pixel of a noise stroke is a bilinear
    blend of neighbouring random values. Compositing at the floored origin put
    a completely different colour in each one - (227,29,100) where the browser
    has (166,63,55) - which no origin-only fixture could see. These samples
    come from a real canvas, so this is the browser's answer, not a restatement
    of ours.
    """
    from brushjam.raster import LayerRaster, render_strokes

    cases: List[Dict[str, Any]] = load_fixture("noise-placement.json")["pixels"]
    assert cases and all(case["samples"] for case in cases)
    checked = 0
    for case in cases:
        target = LayerRaster(case["bounds"]["width"], case["bounds"]["height"])
        render_strokes(target, [case["stroke"]], offset_x=case["offsetX"], offset_y=case["offsetY"])
        alpha8 = case["alpha8"]
        for sample in case["samples"]:
            x, y = sample["x"], sample["y"]
            alpha = target.alpha[y, x]
            assert round(float(alpha) * 255) == alpha8, (case["stroke"]["id"], x, y, alpha)
            # The raster is premultiplied; getImageData is not. Half rounds up
            # here, as it does in the canvas - Python's round() would send an
            # exact .5 to the nearest even instead.
            actual = [math.floor(float(v) * 255.0 / alpha8 + 0.5) for v in target.rgb[y, x]]
            assert actual == sample["rgb"], (case["stroke"]["id"], x, y, actual, sample["rgb"])
            checked += 1
    assert checked > 200, checked
