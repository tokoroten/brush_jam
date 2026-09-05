"""Reducer behaviour the Node suites cover (apps/server/test/room*.test.ts)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from brushjam.room import (
    MAX_PENDING_PER_USER,
    MAX_SESSIONS,
    PENDING_STROKE_IDLE_MS,
    RoomLimits,
    add_member,
    apply_client_message,
    clamp_denoise,
    clamp_resolution,
    create_room,
    expire_pending_strokes,
    join_member,
    now_ms,
    remove_member,
    snapshot,
)


def room(**kwargs):
    return create_room("testroom", 0.7, 1024, 768, True, "fast", kwargs.get("limits"))


def draw(state, user, stroke_id, layer_id, points, tool="pen", **extra):
    init: Dict[str, Any] = {
        "id": stroke_id,
        "layerId": layer_id,
        "tool": tool,
        "color": "#000000",
        "width": 10,
        "points": points[:1],
    }
    init.update(extra)
    apply_client_message(state, user, {"t": "stroke_start", "stroke": init})
    return apply_client_message(
        state, user, {"t": "stroke_end", "strokeId": stroke_id, "points": points[1:]}
    )


def test_undo_takes_the_senders_own_latest_stroke() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    bob = add_member(state, "Bob")["userId"]
    layer = state.layers[0]["id"]
    draw(state, alice, "a1", layer, [{"x": 1, "y": 1}, {"x": 2, "y": 2}])
    draw(state, bob, "b1", layer, [{"x": 3, "y": 3}, {"x": 4, "y": 4}])
    draw(state, alice, "a2", layer, [{"x": 5, "y": 5}, {"x": 6, "y": 6}])

    result = apply_client_message(state, alice, {"t": "undo"})
    assert result.broadcast[0]["strokeId"] == f"{alice}:a2"
    # Bob's undo takes Bob's stroke, not Alice's older one.
    result = apply_client_message(state, bob, {"t": "undo"})
    assert result.broadcast[0]["strokeId"] == f"{bob}:b1"
    # Nothing left for Bob.
    assert apply_client_message(state, bob, {"t": "undo"}).broadcast == []


def test_undone_order_is_preserved_in_the_snapshot() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    bob = add_member(state, "Bob")["userId"]
    layer = state.layers[0]["id"]
    draw(state, alice, "a1", layer, [{"x": 1, "y": 1}, {"x": 2, "y": 2}])
    draw(state, bob, "b1", layer, [{"x": 3, "y": 3}, {"x": 4, "y": 4}])
    apply_client_message(state, bob, {"t": "undo"})
    apply_client_message(state, alice, {"t": "undo"})
    snap = snapshot(state, alice, "idle", {"window": 768, "apply": 1024, "canvasSize": 1024})
    assert snap["undone"] == [f"{bob}:b1", f"{alice}:a1"]


def test_clear_layer_forgets_the_undone_ids_too() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    draw(state, alice, "a1", layer, [{"x": 1, "y": 1}, {"x": 2, "y": 2}])
    apply_client_message(state, alice, {"t": "undo"})
    apply_client_message(state, alice, {"t": "clear_layer", "layerId": layer})
    assert state.strokes == []
    assert list(state.undone) == []


def test_a_pending_stroke_is_cancelled_when_its_layer_moves() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    apply_client_message(
        state,
        alice,
        {
            "t": "stroke_start",
            "stroke": {
                "id": "a1",
                "layerId": layer,
                "tool": "pen",
                "color": "#000000",
                "width": 4,
                "points": [{"x": 1, "y": 1}],
            },
        },
    )
    result = apply_client_message(state, alice, {"t": "layer_update", "id": layer, "patch": {"offsetX": 20}})
    cancels = [m for m in result.broadcast if m["t"] == "stroke_cancel"]
    assert [c["reason"] for c in cancels] == ["layer moved"]
    assert state.pending == {}


def test_a_locked_layer_refuses_a_transform_but_not_a_rename() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    apply_client_message(state, alice, {"t": "layer_update", "id": layer, "patch": {"locked": True}})
    refused = apply_client_message(
        state, alice, {"t": "layer_update", "id": layer, "patch": {"offsetX": 5}}
    )
    assert refused.to_sender[0]["message"] == "layer is locked"
    renamed = apply_client_message(
        state, alice, {"t": "layer_update", "id": layer, "patch": {"name": "Sky"}}
    )
    assert renamed.broadcast[0]["layer"]["name"] == "Sky"
    # A rename changes no pixels, so it must not spend a generation.
    assert renamed.dirty == []


def test_the_last_draw_layer_cannot_be_deleted() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    refused = apply_client_message(state, alice, {"t": "layer_delete", "id": state.layers[0]["id"]})
    assert refused.to_sender[0]["message"] == "cannot delete the last draw layer"


def test_a_user_cannot_hold_more_than_four_strokes_in_progress() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    for i in range(MAX_PENDING_PER_USER):
        result = apply_client_message(
            state,
            alice,
            {
                "t": "stroke_start",
                "stroke": {
                    "id": f"s{i}",
                    "layerId": layer,
                    "tool": "pen",
                    "color": "#000000",
                    "width": 4,
                    "points": [],
                },
            },
        )
        assert result.relay
    refused = apply_client_message(
        state,
        alice,
        {
            "t": "stroke_start",
            "stroke": {
                "id": "s9",
                "layerId": layer,
                "tool": "pen",
                "color": "#000000",
                "width": 4,
                "points": [],
            },
        },
    )
    assert refused.to_sender[0]["t"] == "stroke_cancel"
    assert refused.to_sender[1]["message"] == "too many strokes in progress"


def test_an_idle_pending_stroke_is_abandoned() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    apply_client_message(
        state,
        alice,
        {
            "t": "stroke_start",
            "stroke": {
                "id": "s1",
                "layerId": layer,
                "tool": "pen",
                "color": "#000000",
                "width": 4,
                "points": [],
            },
        },
    )
    assert expire_pending_strokes(state, now_ms()) == []
    cancels = expire_pending_strokes(state, now_ms() + PENDING_STROKE_IDLE_MS + 1)
    assert [c["reason"] for c in cancels] == ["stroke abandoned"]


def test_a_token_resumes_the_same_identity() -> None:
    state = room()
    first = join_member(state, "Alice", "tok-alice-0001")
    remove_member(state, first["userId"])
    again = join_member(state, "Alice", "tok-alice-0001")
    assert again["userId"] == first["userId"]


def test_a_full_session_table_never_evicts_a_connected_member() -> None:
    state = room()
    live = [join_member(state, f"user{i}", f"token-{i:04d}") for i in range(MAX_SESSIONS)]
    assert len(state.sessions) == MAX_SESSIONS
    newcomer = join_member(state, "late", "token-late-0001")
    # Everyone recorded is still connected, so the newcomer simply gets a
    # working but non-resumable identity.
    assert len(state.sessions) == MAX_SESSIONS
    assert "token-late-0001" not in state.sessions
    assert newcomer["userId"] in state.members
    # Once somebody disconnects, the newcomer's token fits.
    remove_member(state, live[0]["userId"])
    other = join_member(state, "later", "token-later-001")
    assert state.sessions["token-later-001"]["userId"] == other["userId"]


def test_settings_are_clamped_to_the_backend_limits() -> None:
    state = create_room(
        "r",
        0.7,
        1024,
        768,
        True,
        "fast",
        RoomLimits(profiles=["fast"], max_denoise=0.9, max_resolution=768),
    )
    alice = add_member(state, "Alice")["userId"]
    assert state.ai_profiles == ["fast"]
    refused = apply_client_message(state, alice, {"t": "set_ai_settings", "aiProfile": "quality"})
    assert refused.to_sender[0]["message"] == "this backend only supports the fast profile"
    refused = apply_client_message(state, alice, {"t": "set_ai_settings", "denoise": 0.95})
    assert refused.to_sender[0]["message"] == "this backend supports denoise up to 0.9"
    refused = apply_client_message(state, alice, {"t": "set_ai_settings", "aiResolution": 1024})
    assert refused.to_sender[0]["message"] == "this room can generate at up to 768"


def test_switching_profile_moves_the_resolution_to_its_default() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    assert state.ai_resolution == 768
    result = apply_client_message(state, alice, {"t": "set_ai_settings", "aiProfile": "quality"})
    # capped by the room's ceiling (768 here, from AI_WINDOW)
    assert result.broadcast[0]["aiResolution"] == 768
    assert result.prompt_changed is True


def test_an_unchanged_setting_is_not_broadcast() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    assert apply_client_message(state, alice, {"t": "set_ai_settings", "denoise": 0.7}).broadcast == []
    assert apply_client_message(state, alice, {"t": "set_prompt", "prompt": state.prompt}).broadcast == []


@pytest.mark.parametrize(
    "value,expected", [(0.7, 0.7), (0.73, 0.75), (0.01, 0.2), (5, 0.95), (float("nan"), 0.55)]
)
def test_clamp_denoise(value: float, expected: float) -> None:
    assert clamp_denoise(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,maximum,expected", [(768, 1024, 768), (700, 1024, 704), (100, 1024, 512), (4096, 1024, 1024)]
)
def test_clamp_resolution(value: float, maximum: int, expected: int) -> None:
    assert clamp_resolution(value, maximum) == expected


def test_a_refused_locked_layer_update_changes_nothing() -> None:
    """`{locked: true, offsetX: 10}` used to lock the layer and then refuse.

    The refusal carries no revision and no broadcast, so every client kept
    believing the layer was unlocked while the server had locked it.
    """
    room = create_room("r")
    join_member(room, "Alice")
    user = next(iter(room.members))
    layer = room.layers[0]
    before_revision = room.human_revision

    result = apply_client_message(
        room,
        user,
        {"t": "layer_update", "id": layer["id"], "patch": {"locked": True, "offsetX": 10}},
    )
    assert result.to_sender and result.to_sender[0]["message"] == "layer is locked"
    assert layer["locked"] is False, "a refused patch must not lock the layer"
    assert (layer.get("offsetX") or 0) == 0
    assert room.human_revision == before_revision
    assert result.broadcast == []


def test_unlocking_and_moving_in_one_update_is_still_allowed() -> None:
    room = create_room("r")
    join_member(room, "Alice")
    user = next(iter(room.members))
    layer = room.layers[0]
    layer["locked"] = True

    apply_client_message(
        room,
        user,
        {"t": "layer_update", "id": layer["id"], "patch": {"locked": False, "offsetX": 12}},
    )
    assert layer["locked"] is False
    assert layer["offsetX"] == 12


def test_a_locked_layer_still_refuses_a_transform_on_its_own() -> None:
    room = create_room("r")
    join_member(room, "Alice")
    user = next(iter(room.members))
    layer = room.layers[0]
    layer["locked"] = True

    result = apply_client_message(
        room, user, {"t": "layer_update", "id": layer["id"], "patch": {"offsetX": 12}}
    )
    assert result.to_sender[0]["message"] == "layer is locked"
    assert (layer.get("offsetX") or 0) == 0


# ------------------------------------------------- the clamp is in world space


def test_a_moved_layer_keeps_the_point_the_player_drew() -> None:
    """Points are layer-space; the limit is a world-space one.

    Move a layer 1500 to the right and draw at world x=100: the browser sends
    local x=-1400. Clamping the local number to -canvas pinned it at -1024,
    which is world 476 - the mark jumped half a canvas away from the cursor the
    moment it was committed, and the AI saw it there too.
    """
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    apply_client_message(state, alice, {"t": "layer_update", "id": layer, "patch": {"offsetX": 1500}})

    draw(state, alice, "a1", layer, [{"x": -1400, "y": 10}, {"x": -1390, "y": 20}])
    committed = state.strokes[-1]
    assert [p["x"] for p in committed["points"]] == [-1400, -1390]


def test_the_world_clamp_still_bounds_a_moved_layer() -> None:
    """Moving a layer must not extend how far outside the canvas anyone can draw."""
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    apply_client_message(state, alice, {"t": "layer_update", "id": layer, "patch": {"offsetX": 1500}})

    # World -5000 and world 9000, both far outside the allowed -1024..2048.
    draw(state, alice, "a1", layer, [{"x": -6500, "y": 0}, {"x": 7500, "y": 0}])
    committed = state.strokes[-1]
    world = [p["x"] + 1500 for p in committed["points"]]
    assert world == [-1024, 2048]


def test_a_chunk_is_clamped_against_its_own_layers_offset() -> None:
    """The middle of a stroke used to be clamped with no offset at all."""
    state = room()
    alice = add_member(state, "Alice")["userId"]
    layer = state.layers[0]["id"]
    apply_client_message(state, alice, {"t": "layer_update", "id": layer, "patch": {"offsetY": -1500}})
    apply_client_message(
        state,
        alice,
        {
            "t": "stroke_start",
            "stroke": {
                "id": "a1",
                "layerId": layer,
                "tool": "pen",
                "color": "#000000",
                "width": 4,
                "points": [{"x": 0, "y": 1600}],
            },
        },
    )
    relayed = apply_client_message(
        state, alice, {"t": "stroke_chunk", "strokeId": "a1", "points": [{"x": 0, "y": 1700}]}
    )
    assert [p["y"] for p in relayed.relay[0]["points"]] == [1700]


# ------------------------------------------------------------------- the seed


def test_the_room_owns_its_seed() -> None:
    """Fixed per room, not drawn per generation.

    With a fresh seed every time, adding one stroke reshuffled the entire
    picture: nobody could tell what their own change had done.
    """
    from brushjam.constants import MAX_SEED
    from brushjam.room import capture_render_snapshot

    state = room()
    assert 0 <= state.seed <= MAX_SEED
    assert capture_render_snapshot(state).seed == state.seed
    assert capture_render_snapshot(state).seed == state.seed

    # Two rooms do not share one, or every room would draw the same picture.
    seeds = {create_room(f"r{i}").seed for i in range(20)}
    assert len(seeds) > 1


def test_setting_the_seed_re_runs_the_ai() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    result = apply_client_message(state, alice, {"t": "set_ai_settings", "seed": 4242})
    assert state.seed == 4242
    assert result.prompt_changed is True
    assert result.broadcast[0]["seed"] == 4242
    assert snapshot(state, alice, "idle", {"window": 768, "apply": 1024})["seed"] == 4242


def test_setting_the_same_seed_does_nothing() -> None:
    state = room()
    alice = add_member(state, "Alice")["userId"]
    apply_client_message(state, alice, {"t": "set_ai_settings", "seed": 7})
    again = apply_client_message(state, alice, {"t": "set_ai_settings", "seed": 7})
    assert again.broadcast == [] and again.prompt_changed is False


def test_a_seed_out_of_range_is_refused() -> None:
    from brushjam.constants import MAX_SEED

    state = room()
    alice = add_member(state, "Alice")["userId"]
    before = state.seed
    for bad in (-1, MAX_SEED + 1, 1.5):
        refused = apply_client_message(state, alice, {"t": "set_ai_settings", "seed": bad})
        assert refused.to_sender and "seed" in refused.to_sender[0]["message"], bad
        assert state.seed == before


def test_the_seed_reaches_the_generation() -> None:
    """The scheduler must use the room's seed, not one of its own."""
    from brushjam.scheduler import RenderJob, _default_seed

    job = RenderJob(
        revision=1,
        prompt="",
        denoise=None,
        negative_prompt=None,
        resolution=768,
        profile="fast",
        render=lambda crop, size: b"",
        seed=99,
    )
    assert job.seed == 99
    # A job without one still gets a random seed rather than a constant.
    assert len({_default_seed() for _ in range(10)}) > 1
