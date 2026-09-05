"""Rendering: determinism, the noise pen's world addressing, layer semantics."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from brushjam.noise import fnv1a, noise_rgb
from brushjam.raster import (
    AICanvas,
    build_full_mask,
    forget_images,
    render_crop_input,
    to_png,
)
from brushjam.room import (
    RenderSnapshot,
    RoomImage,
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


# -------------------------------------------------- enlarged reference images


def _reference_snapshot(image: Image.Image, scale: float, x: float = 0, y: float = 0):
    """A snapshot with one reference layer holding `image` at `scale`."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    stored = RoomImage(
        id="img1",
        mime="image/png",
        data=buf.getvalue(),
        width=image.width,
        height=image.height,
        created_at=0,
    )
    layer = {
        "id": "L1",
        "name": "ref",
        "kind": "reference",
        "visible": True,
        "opacity": 1.0,
        "locked": False,
        "includeInAI": True,
        "imageId": "img1",
        "scale": scale,
        "x": x,
        "y": y,
        "offsetX": 0,
        "offsetY": 0,
    }
    forget_images(["img1"])
    return RenderSnapshot(
        revision=1,
        prompt="",
        denoise=0.7,
        negative_prompt="",
        ai_resolution=256,
        ai_profile="fast",
        seed=1234,
        layers=[layer],
        strokes=[],
        undone=set(),
        images={"img1": stored},
    )


def test_an_enlarged_reference_only_resamples_what_the_crop_can_see(monkeypatch) -> None:
    """A legal upload must not become an illegal allocation.

    4096x4096 at the accepted scale of 8 is a 32768x32768 image - 16 GiB as
    float32 RGBA - for a crop that can show a megapixel of it. Nothing that
    large may ever be asked for, so the resize is intercepted rather than
    survived.
    """
    source = Image.new("RGBA", (4096, 4096), (10, 200, 30, 255))
    snap = _reference_snapshot(source, scale=8)

    asked: list = []
    real_resize = Image.Image.resize

    def spy(self, size, *args, **kwargs):
        asked.append(size)
        return real_resize(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", spy)
    png = render_crop_input(snap, FULL, 256)

    assert asked, "the reference was never resampled"
    for width, height in asked:
        assert width * height <= (256 + 64) * (256 + 64), asked
    pixels = _pixels(png)
    assert tuple(pixels[128, 128]) == (10, 200, 30)


def test_a_reference_fully_outside_the_crop_is_skipped(monkeypatch) -> None:
    snap = _reference_snapshot(Image.new("RGBA", (64, 64), (255, 0, 0, 255)), scale=4, x=1000, y=1000)
    monkeypatch.setattr(
        Image.Image, "resize", lambda *a, **k: pytest.fail("an invisible reference was resampled")
    )
    assert (_pixels(render_crop_input(snap, FULL, 256)) == 255).all()


@pytest.mark.parametrize("scale,x,y", [(4, -100, -60), (4, 200, 30), (0.5, 10, 10), (3, 0, 0), (8, -700, -700)])
def test_the_patch_matches_scaling_the_whole_image(scale, x, y) -> None:
    """The pixels a crop shows must not depend on how much of the image it saw.

    This is the property the filter-support padding buys: resampling only the
    visible box has to give the same pixels as resampling everything and
    throwing most of it away, which is what the code used to do.
    """
    from brushjam.raster import _reference_patch

    rng = np.random.default_rng(7)
    source = Image.fromarray(rng.integers(0, 256, (200, 200, 4), dtype=np.uint8), "RGBA")
    target_w = max(1, round(200 * scale))
    target_h = max(1, round(200 * scale))

    placed = _reference_patch(source, target_w, target_h, x, y, 256, 256)
    assert placed is not None
    patch, left, top = placed
    whole = np.asarray(source.resize((target_w, target_h), Image.LANCZOS), dtype=np.float32) / 255.0
    expected = whole[top - y : top - y + patch.shape[0], left - x : left - x + patch.shape[1]]
    assert patch.shape == expected.shape
    # Only the part the crop shows has to agree; the padding exists to feed the
    # filter, not to be drawn, and the compositor clips it.
    vy0, vx0 = max(0, -top), max(0, -left)
    vy1, vx1 = min(patch.shape[0], 256 - top), min(patch.shape[1], 256 - left)
    # Not bit-exact: Pillow derives its filter coefficients from the box, and a
    # box whose edges are not whole source pixels rounds a few of them
    # differently. On a random-noise image - the worst case there is - that is
    # a handful of pixels off by at most 6/255.
    assert np.abs(patch[vy0:vy1, vx0:vx1] - expected[vy0:vy1, vx0:vx1]).max() * 255 <= 6
