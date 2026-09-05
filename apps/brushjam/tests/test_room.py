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
