"""Rendering: determinism, the noise pen's world addressing, layer semantics."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from brushjam.noise import fnv1a, noise_rgb
from brushjam.raster import AICanvas, build_full_mask, render_crop_input, to_png
from brushjam.room import (
    RenderSnapshot,
    apply_client_message,
    capture_render_snapshot,
    create_room,
    add_member,
)


def _room_with(strokes, canvas=256):
    state = create_room("r", 0.7, canvas, canvas, True, "fast")
    user = add_member(state, "A")["userId"]
    layer = state.layers[0]["id"]
    for stroke_id, tool, points, extra in strokes:
        init = {
            "id": stroke_id,
            "layerId": layer,
            "tool": tool,
            "color": "#204080",
            "width": 20,
            "points": points[:1],
        }
        init.update(extra)
        apply_client_message(state, user, {"t": "stroke_start", "stroke": init})
        apply_client_message(
            state, user, {"t": "stroke_end", "strokeId": stroke_id, "points": points[1:]}
        )
    return state, user, layer


def _pixels(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as img:
        return np.asarray(img.convert("RGB"))


FULL = {"x": 0, "y": 0, "width": 256, "height": 256}


def test_render_is_deterministic() -> None:
    state, _, _ = _room_with(
        [
            ("s1", "pen", [{"x": 20, "y": 20}, {"x": 200, "y": 120}], {}),
            ("s2", "noise", [{"x": 40, "y": 180}, {"x": 200, "y": 220}], {"width": 40}),
        ]
    )
    snap = capture_render_snapshot(state)
    assert render_crop_input(snap, FULL, 256) == render_crop_input(snap, FULL, 256)


def test_a_noise_stroke_is_addressed_in_world_coordinates() -> None:
    """A crop rendered with the crop origin subtracted must produce the same
    pixels as the same region of the full-canvas render."""
    state, _, _ = _room_with(
        [("s1", "noise", [{"x": 100, "y": 100}, {"x": 180, "y": 160}], {"width": 40})]
    )
    snap = capture_render_snapshot(state)
    full = _pixels(render_crop_input(snap, FULL, 256))
    crop = {"x": 64, "y": 64, "width": 128, "height": 128}
    cropped = _pixels(render_crop_input(snap, crop, 128))
    # Compare the interior only: coverage at the very edge of a crop depends on
    # geometry the crop cannot see.
    a = full[64 + 4 : 192 - 4, 64 + 4 : 192 - 4]
    b = cropped[4:-4, 4:-4]
    assert np.array_equal(a, b)


def test_noise_pixels_follow_the_documented_hash() -> None:
    state, _, _ = _room_with(
        [("s1", "noise", [{"x": 100, "y": 100}, {"x": 160, "y": 100}], {"width": 40})]
    )
    snap = capture_render_snapshot(state)
    pixels = _pixels(render_crop_input(snap, FULL, 256))
    stroke_id = state.strokes[0]["id"]
    seed = fnv1a(stroke_id)
    # A point well inside the stroke body, where coverage is 1.
    assert tuple(pixels[100, 130]) == noise_rgb(seed, 130, 100)


def test_an_eraser_cuts_the_layer_but_not_the_background() -> None:
    state, user, layer = _room_with(
        [("s1", "pen", [{"x": 20, "y": 128}, {"x": 236, "y": 128}], {"width": 60})]
    )
    before = _pixels(render_crop_input(capture_render_snapshot(state), FULL, 256))
    assert tuple(before[128, 128]) == (0x20, 0x40, 0x80)
    apply_client_message(
        state,
        user,
        {
            "t": "stroke_start",
            "stroke": {
                "id": "e1",
                "layerId": layer,
                "tool": "eraser",
                "color": "#000000",
                "width": 40,
                "points": [{"x": 128, "y": 100}],
            },
        },
    )
    apply_client_message(state, user, {"t": "stroke_end", "strokeId": "e1", "points": [{"x": 128, "y": 160}]})
    after = _pixels(render_crop_input(capture_render_snapshot(state), FULL, 256))
    # White paper shows through where the eraser went.
    assert tuple(after[128, 128]) == (255, 255, 255)


def test_a_translucent_stroke_does_not_darken_where_it_crosses_itself() -> None:
    state, _, _ = _room_with(
        [
            (
                "s1",
                "pen",
                [{"x": 60, "y": 60}, {"x": 200, "y": 200}, {"x": 200, "y": 60}, {"x": 60, "y": 200}],
                {"alpha": 0.5, "width": 24},
            )
        ]
    )
    pixels = _pixels(render_crop_input(capture_render_snapshot(state), FULL, 256))
    crossing = pixels[130, 130]
    single = pixels[80, 80]
    assert np.abs(crossing.astype(int) - single.astype(int)).max() <= 2


def test_layer_opacity_and_visibility() -> None:
    state, user, layer = _room_with(
        [("s1", "pen", [{"x": 20, "y": 128}, {"x": 236, "y": 128}], {"width": 60})]
    )
    apply_client_message(
        state, user, {"t": "layer_update", "id": layer, "patch": {"opacity": 0.5}}
    )
    half = _pixels(render_crop_input(capture_render_snapshot(state), FULL, 256))[128, 128]
    assert 100 < int(half[0]) < 160  # halfway between 0x20 and white
    apply_client_message(
        state, user, {"t": "layer_update", "id": layer, "patch": {"visible": False}}
    )
    hidden = _pixels(render_crop_input(capture_render_snapshot(state), FULL, 256))[128, 128]
    assert tuple(hidden) == (255, 255, 255)


def test_a_moved_layer_translates_its_strokes() -> None:
    state, user, layer = _room_with(
        [("s1", "pen", [{"x": 50, "y": 50}, {"x": 50, "y": 200}], {"width": 20})]
    )
    apply_client_message(
        state, user, {"t": "layer_update", "id": layer, "patch": {"offsetX": 60}}
    )
    pixels = _pixels(render_crop_input(capture_render_snapshot(state), FULL, 256))
    assert tuple(pixels[128, 110]) == (0x20, 0x40, 0x80)
    assert tuple(pixels[128, 50]) == (255, 255, 255)


def test_the_render_is_resampled_to_the_generation_size() -> None:
    state, _, _ = _room_with([("s1", "pen", [{"x": 20, "y": 20}, {"x": 200, "y": 200}], {})])
    png = render_crop_input(capture_render_snapshot(state), FULL, 128)
    with Image.open(io.BytesIO(png)) as img:
        assert img.size == (128, 128)


def test_full_mask_is_opaque_and_composites_the_whole_crop() -> None:
    mask = build_full_mask(64)
    assert not mask.empty
    assert np.asarray(mask.alpha).min() == 255
    canvas = AICanvas(128)
    assert canvas.to_png()  # transparent until something lands
    patch = to_png(Image.new("RGB", (64, 64), (10, 20, 30)))
    out = canvas.composite(patch, {"x": 0, "y": 0, "width": 128, "height": 128}, mask.alpha)
    assert _pixels(out)[64, 64].tolist() == [10, 20, 30]
