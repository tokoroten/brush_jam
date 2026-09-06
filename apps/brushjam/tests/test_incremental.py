"""Incremental layer rasters must be indistinguishable from full renders.

The AI input is rendered from the whole stroke log every time, which at a few
thousand strokes dominated a generation (the soak that prompted this measured
throughput falling from 41 to 11 generations a minute against an instant
backend). Layers are now kept between renders and only new strokes are drawn
onto them.

That is only allowed to be a speed-up. Compositing is a sequence of float32
operations on one buffer, so continuing a saved buffer must produce the same
bytes as replaying the whole sequence - and "the same bytes" is what these
tests assert, over random sequences of the three tools, with alpha, undo,
clear and layer moves mixed in.
"""

from __future__ import annotations

import io
import random
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

import brushjam.raster as raster_module
from brushjam.raster import (
    LAYER_CACHE,
    LayerCacheStore,
    render_crop_input,
)
from brushjam.room import (
    RoomLimits,
    apply_client_message,
    capture_render_snapshot,
    create_room,
    join_member,
    sorted_layers,
)
from brushjam.validate import validate_client_message

CANVAS = 256
CROP = {"x": 0, "y": 0, "width": CANVAS, "height": CANVAS}
ROOM_ID = "incr01"


@pytest.fixture(autouse=True)
def _empty_cache():
    """Each test starts with nothing cached and leaves nothing behind."""
    LAYER_CACHE.clear()
    yield
    LAYER_CACHE.clear()


def room():
    state = create_room(
        ROOM_ID,
        0.55,
        CANVAS,
        CANVAS,
        True,
        "fast",
        RoomLimits(profiles=["fast", "quality"], max_denoise=0.95, max_resolution=1024),
        seed=7,
    )
    return state, join_member(state, "Alice", "tok-alice-0001")["userId"]


def send(state, user_id: str, msg: Dict[str, Any]) -> None:
    validated = validate_client_message(msg)
    assert validated.ok, validated.error
    apply_client_message(state, user_id, validated.msg)


def stroke(rng: random.Random, layer_id: str, index: int) -> Dict[str, Any]:
    tool = rng.choice(["pen", "pen", "noise", "eraser"])
    points = [
        {"x": rng.uniform(0, CANVAS), "y": rng.uniform(0, CANVAS), "p": rng.uniform(0.2, 1.0)}
        for _ in range(rng.randint(2, 6))
    ]
    return {
        "id": f"s{index:04d}",
        "layerId": layer_id,
        "tool": tool,
        "color": rng.choice(["#1b1b1b", "#cc3344", "#2277aa"]),
        "width": rng.randint(2, 40),
        # Alpha sends the stroke through the temp-raster path, which is where a
        # sub-pixel difference would show up first.
        "alpha": rng.choice([1.0, 0.35, 0.8]),
        "points": points,
    }


def commit(state, user_id: str, s: Dict[str, Any]) -> None:
    send(state, user_id, {"t": "stroke_start", "stroke": s})
    send(state, user_id, {"t": "stroke_end", "strokeId": s["id"], "points": []})


def rendered(state, room_id) -> bytes:
    """One render of the whole canvas, cached or not."""
    return render_crop_input(capture_render_snapshot(state), CROP, CANVAS, room_id=room_id)


def pixels(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as img:
        return np.asarray(img.convert("RGB"))


def assert_same(state) -> None:
    """The incremental render and the from-scratch one, byte for byte."""
    incremental = rendered(state, ROOM_ID)
    scratch = rendered(state, None)
    assert incremental == scratch, "incremental render differs from a full one"
    assert np.array_equal(pixels(incremental), pixels(scratch))


# --------------------------------------------------------------- the basics


def test_a_cached_layer_grows_stroke_by_stroke() -> None:
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(1)
    for i in range(12):
        commit(state, user, stroke(rng, layer_id, i))
        assert_same(state)
    assert LAYER_CACHE.stats()["layers"] == 1


def test_every_tool_survives_the_cache() -> None:
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    base = {"layerId": layer_id, "color": "#112233", "width": 24, "alpha": 0.6}
    line = [{"x": 20, "y": 20, "p": 1}, {"x": 200, "y": 180, "p": 1}]
    for i, tool in enumerate(("pen", "noise", "eraser", "noise", "pen")):
        commit(state, user, {**base, "id": f"t{i}", "tool": tool, "points": line})
        assert_same(state)


def test_undo_inside_the_cached_prefix_rebuilds_it() -> None:
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(2)
    for i in range(6):
        commit(state, user, stroke(rng, layer_id, i))
    assert_same(state)  # everything cached
    send(state, user, {"t": "undo"})  # removes the last stroke, which IS cached
    assert_same(state)
    for i in range(6, 9):
        commit(state, user, stroke(rng, layer_id, i))
    assert_same(state)


def test_clearing_a_layer_rebuilds_it() -> None:
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(3)
    for i in range(5):
        commit(state, user, stroke(rng, layer_id, i))
    assert_same(state)
    send(state, user, {"t": "clear_layer", "layerId": layer_id})
    assert_same(state)
    commit(state, user, stroke(rng, layer_id, 99))
    assert_same(state)


def test_moving_a_layer_rebuilds_it_at_the_new_offset() -> None:
    """A fractional offset rasterises every stroke at a different sub-pixel
    phase, so the cached pixels cannot simply be shifted."""
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(4)
    for i in range(5):
        commit(state, user, stroke(rng, layer_id, i))
    assert_same(state)
    for offset in ({"offsetX": 12, "offsetY": -7}, {"offsetX": 3.25, "offsetY": 0.5}):
        send(state, user, {"t": "layer_update", "id": layer_id, "patch": offset})
        assert_same(state)
    commit(state, user, stroke(rng, layer_id, 50))
    assert_same(state)


def test_opacity_and_visibility_do_not_invalidate_anything() -> None:
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(5)
    for i in range(4):
        commit(state, user, stroke(rng, layer_id, i))
    assert_same(state)
    before = LAYER_CACHE.stats()

    send(state, user, {"t": "layer_update", "id": layer_id, "patch": {"opacity": 0.4}})
    assert_same(state)
    send(state, user, {"t": "layer_update", "id": layer_id, "patch": {"visible": False}})
    rendered(state, ROOM_ID)  # a hidden layer is not rendered at all
    send(state, user, {"t": "layer_update", "id": layer_id, "patch": {"visible": True}})
    assert_same(state)
    # Nothing was thrown away and nothing was rebuilt: same entry, same bytes.
    assert LAYER_CACHE.stats() == before


def test_several_layers_are_cached_independently() -> None:
    state, user = room()
    send(state, user, {"t": "layer_create", "layer": {"kind": "draw"}})
    layers = [l["id"] for l in sorted_layers(state) if l["kind"] == "draw"]
    assert len(layers) == 2
    rng = random.Random(6)
    for i in range(8):
        commit(state, user, stroke(rng, layers[i % 2], i))
        assert_same(state)
    assert LAYER_CACHE.stats()["layers"] == 2

    send(state, user, {"t": "layer_delete", "id": layers[1]})
    assert_same(state)


# ------------------------------------------------------------- the fuzz test


@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15])
def test_random_sequences_render_identically(seed: int) -> None:
    """Pen, noise and eraser strokes with alpha, plus undo, clear, moves,
    opacity changes and a second layer - compared after every single step."""
    rng = random.Random(seed)
    state, user = room()
    send(state, user, {"t": "layer_create", "layer": {"kind": "draw"}})
    layers = [l["id"] for l in sorted_layers(state) if l["kind"] == "draw"]
    index = 0

    for _ in range(24):
        action = rng.choices(
            ["stroke", "stroke", "stroke", "stroke", "undo", "clear", "move", "opacity"],
            weights=[5, 5, 5, 5, 2, 1, 2, 1],
        )[0]
        target = rng.choice(layers)
        if action == "stroke":
            commit(state, user, stroke(rng, target, index))
            index += 1
        elif action == "undo":
            send(state, user, {"t": "undo"})
        elif action == "clear":
            send(state, user, {"t": "clear_layer", "layerId": target})
        elif action == "move":
            send(
                state,
                user,
                {
                    "t": "layer_update",
                    "id": target,
                    "patch": {
                        "offsetX": round(rng.uniform(-20, 20), 2),
                        "offsetY": round(rng.uniform(-20, 20), 2),
                    },
                },
            )
        else:
            send(
                state,
                user,
                {"t": "layer_update", "id": target, "patch": {"opacity": rng.choice([0.3, 1.0])}},
            )
        assert_same(state)


# ------------------------------------------------------------ housekeeping


def test_the_budget_bounds_what_is_held() -> None:
    """Past the budget the least recently used layer goes, and rendering it
    again is a rebuild rather than a wrong picture."""
    store = LayerCacheStore(budget_bytes=CANVAS * CANVAS * 16 * 2)  # two layers
    state, user = room()
    send(state, user, {"t": "layer_create", "layer": {"kind": "draw"}})
    send(state, user, {"t": "layer_create", "layer": {"kind": "draw"}})
    layers = [l["id"] for l in sorted_layers(state) if l["kind"] == "draw"]
    rng = random.Random(9)
    for i, layer_id in enumerate(layers):
        commit(state, user, stroke(rng, layer_id, i))

    snapshot = capture_render_snapshot(state)
    from brushjam.room import strokes_for_crop

    for layer_id in layers:
        store.render_layer(
            ROOM_ID, layer_id, strokes_for_crop(snapshot, CROP, layer_id), CANVAS, CANVAS, 0, 0
        )
    assert store.stats()["layers"] == 2  # three asked for, two fit
    assert store.stats()["bytes"] <= store.budget_bytes
    # And the room's own render still matches a full one.
    assert_same(state)


def test_a_room_can_hand_its_rasters_back() -> None:
    from brushjam.raster import forget_layer_rasters

    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    commit(state, user, stroke(random.Random(10), layer_id, 0))
    rendered(state, ROOM_ID)
    assert LAYER_CACHE.stats()["layers"] == 1

    forget_layer_rasters(ROOM_ID)
    assert LAYER_CACHE.stats()["layers"] == 0 and LAYER_CACHE.stats()["bytes"] == 0
    # ... and the next render is correct, just slower.
    assert_same(state)


def test_two_rooms_do_not_share_a_layer() -> None:
    """The cache key carries the room, not just the layer.

    Layer ids happen to be random, so a collision is unlikely rather than
    impossible - and "unlikely" is not what should stand between one room and
    another room's drawing.
    """
    first, first_user = room()
    second = create_room(
        "incr02", 0.55, CANVAS, CANVAS, True, "fast", RoomLimits(), seed=7
    )
    second_user = join_member(second, "Bob", "tok-bob-0001")["userId"]
    layer_a = sorted_layers(first)[0]["id"]
    layer_b = sorted_layers(second)[0]["id"]

    commit(first, first_user, stroke(random.Random(20), layer_a, 0))
    commit(second, second_user, stroke(random.Random(21), layer_b, 1))
    a = render_crop_input(capture_render_snapshot(first), CROP, CANVAS, room_id="incr01")
    b = render_crop_input(capture_render_snapshot(second), CROP, CANVAS, room_id="incr02")
    assert a != b
    assert a == render_crop_input(capture_render_snapshot(first), CROP, CANVAS, room_id=None)
    assert b == render_crop_input(capture_render_snapshot(second), CROP, CANVAS, room_id=None)
    assert LAYER_CACHE.stats()["layers"] == 2
    # Dropping one room's rasters leaves the other's alone.
    from brushjam.raster import forget_layer_rasters

    forget_layer_rasters("incr01")
    assert LAYER_CACHE.stats()["layers"] == 1
    assert b == render_crop_input(capture_render_snapshot(second), CROP, CANVAS, room_id="incr02")


# ------------------------------------------------- review 3: stroke identity


def test_a_reused_stroke_id_does_not_return_the_old_drawing() -> None:
    """`clear_layer` takes a layer's strokes out of the log, which frees their
    ids for reuse. The cache keyed on those ids, so the same id carrying a
    different drawing looked like the same prefix, and the render came back as
    the picture that had just been cleared."""
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(11)
    for i in range(4):
        commit(state, user, stroke(rng, layer_id, i))
    before = rendered(state, ROOM_ID)  # the cache now holds s0000..s0003

    send(state, user, {"t": "clear_layer", "layerId": layer_id})
    # Nothing is rendered in between: the next render is the first one that
    # sees the same four ids again, drawn differently.
    other = random.Random(77)
    for i in range(4):
        commit(state, user, stroke(other, layer_id, i))

    incremental = rendered(state, ROOM_ID)
    assert incremental != before, "the test drew the same picture twice"
    assert_same(state)


def test_a_clear_while_a_render_is_running_is_not_undone_by_it(monkeypatch) -> None:
    """A render reads the strokes, then spends 100 ms drawing them. A clear
    that lands in that window must not be reverted by the render installing
    the raster it started before the clear."""
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(5)
    for i in range(3):
        commit(state, user, stroke(rng, layer_id, i))
    strokes = list(capture_render_snapshot(state).strokes)

    store = LayerCacheStore()
    real = raster_module.render_strokes

    def clear_midway(target, drawn, *, offset_x=0.0, offset_y=0.0):
        real(target, drawn, offset_x=offset_x, offset_y=offset_y)
        store.forget_layer(ROOM_ID, layer_id)  # the clear arrives here

    monkeypatch.setattr(raster_module, "render_strokes", clear_midway)
    out = store.render_layer(ROOM_ID, layer_id, strokes, CANVAS, CANVAS, 0.0, 0.0)
    assert out is not None  # the caller still gets its pixels
    assert store.stats()["layers"] == 0, "a stale raster was installed over a clear"
    assert store.stats()["bytes"] == 0


def test_dropping_a_layer_leaves_no_accounting_behind() -> None:
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(6)
    for i in range(3):
        commit(state, user, stroke(rng, layer_id, i))
    strokes = list(capture_render_snapshot(state).strokes)

    store = LayerCacheStore()
    store.render_layer(ROOM_ID, layer_id, strokes, CANVAS, CANVAS, 0.0, 0.0)
    assert store.stats()["bytes"] > 0
    store.forget_room(ROOM_ID)
    assert store.stats()["layers"] == 0 and store.stats()["bytes"] == 0
    # ...and a render after the drop starts a fresh entry rather than doubling
    # the accounting.
    store.render_layer(ROOM_ID, layer_id, strokes, CANVAS, CANVAS, 0.0, 0.0)
    assert store.stats()["layers"] == 1


# --------------------------------------------- review 4: nothing grows for ever


def test_generation_records_do_not_accumulate() -> None:
    """The invalidation counters are the one thing here with no byte budget:
    rooms and layers churn, and a record that protects nothing is retired."""
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(21)
    for i in range(2):
        commit(state, user, stroke(rng, layer_id, i))
    strokes = list(capture_render_snapshot(state).strokes)

    store = LayerCacheStore()
    for n in range(50):
        key_room = "room%03d" % n
        store.render_layer(key_room, layer_id, strokes, CANVAS, CANVAS, 0.0, 0.0)
        store.forget_room(key_room)
    assert store.stats()["layers"] == 0
    assert store.stats()["generations"] == 0, "invalidation records outlived their layers"

    # A cached layer keeps its record, because that is what a later clear
    # compares against.
    store.render_layer(ROOM_ID, layer_id, strokes, CANVAS, CANVAS, 0.0, 0.0)
    store.forget_layer(ROOM_ID, layer_id)
    assert store.stats()["generations"] == 0
    store.clear()
    assert store.stats()["generations"] == 0


def test_a_running_render_keeps_its_invalidation_record(monkeypatch) -> None:
    """Retiring must not lose the protection: a clear during a render still
    stops that render from installing its raster afterwards."""
    state, user = room()
    layer_id = sorted_layers(state)[0]["id"]
    rng = random.Random(22)
    for i in range(2):
        commit(state, user, stroke(rng, layer_id, i))
    strokes = list(capture_render_snapshot(state).strokes)

    store = LayerCacheStore()
    real = raster_module.render_strokes
    seen = {}

    def clear_midway(target, drawn, *, offset_x=0.0, offset_y=0.0):
        real(target, drawn, offset_x=offset_x, offset_y=offset_y)
        # Nothing is cached yet, so this invalidation exists only to protect
        # the render that is running - exactly the record that must not be
        # retired while it runs.
        store.forget_layer(ROOM_ID, layer_id)
        seen["generations"] = store.stats()["generations"]

    monkeypatch.setattr(raster_module, "render_strokes", clear_midway)
    store.render_layer(ROOM_ID, layer_id, strokes, CANVAS, CANVAS, 0.0, 0.0)
    assert seen["generations"] == 1, "the record was dropped while a render held it"
    assert store.stats()["layers"] == 0, "a stale raster was installed over a clear"
    assert store.stats()["generations"] == 0, "and the record was retired afterwards"
