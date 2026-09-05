"""Capacity, admission and identity: the ways a shared server gets exhausted.

Every case here is something an unauthenticated client can do on purpose, so
the assertions are about what the server refuses, not about what it manages.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from brushjam.ai.backends.mock import MockBackend
from brushjam.app import create_app
from brushjam.config import load_config
from brushjam.room import apply_client_message, create_room, join_member
from brushjam.runtime import RoomRegistry, RoomRuntime


def config(**env):
    settings = {
        "AI_BACKEND": "mock",
        "CANVAS_SIZE": "512",
        "AI_WINDOW": "512",
        "AI_DEBOUNCE_MS": "10",
    }
    settings.update(env)
    return load_config(settings)


def client(**env) -> TestClient:
    return TestClient(create_app(config(**env), MockBackend(latency_ms=0)))


class FakeSocket:
    """A Connection that records what it was sent and whether it was cut off."""

    def __init__(self) -> None:
        self.sent: List[str] = []
        self._open = True
        self.revoked = False
        self.buffered = 0

    def send_text(self, data: str) -> None:
        self.sent.append(data)

    def close_now(self) -> None:
        self._open = False

    def revoke(self) -> None:
        self._open = False
        self.revoked = True

    @property
    def open(self) -> bool:
        return self._open

    @property
    def buffered_bytes(self) -> int:
        return self.buffered

    def types(self) -> List[str]:
        return [json.loads(s)["t"] for s in self.sent]


# ------------------------------------------------------------- room creation


def test_room_creation_is_rate_limited_per_client() -> None:
    with client(ROOM_CREATE_PER_MIN="3") as c:
        codes = [c.post("/api/rooms").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]


def test_the_rate_limit_bucket_refills() -> None:
    registry = RoomRegistry(MockBackend(), config(ROOM_CREATE_PER_MIN="60"))
    now = 1000.0
    assert all(registry.allow_create("1.2.3.4", now) for _ in range(60))
    assert not registry.allow_create("1.2.3.4", now)
    # 60 a minute is one a second.
    assert registry.allow_create("1.2.3.4", now + 1.01)


def test_one_client_cannot_spend_the_budget_of_another() -> None:
    registry = RoomRegistry(MockBackend(), config(ROOM_CREATE_PER_MIN="1"))
    assert registry.allow_create("1.1.1.1", 0.0)
    assert not registry.allow_create("1.1.1.1", 0.0)
    assert registry.allow_create("2.2.2.2", 0.0)


def test_a_room_nobody_joined_is_reclaimed_quickly() -> None:
    cfg = config(UNJOINED_ROOM_TTL_MS="1000", ROOM_IDLE_MS="600000")
    registry = RoomRegistry(MockBackend(), cfg)
    room = registry.create()
    assert room is not None
    created = room.state.created_at
    # Still inside the grace period.
    assert registry.sweep(created + 500) == 0
    assert registry.size == 1
    assert registry.sweep(created + 2000) == 1
    assert registry.size == 0


def test_a_room_someone_joined_keeps_the_longer_idle_life() -> None:
    cfg = config(UNJOINED_ROOM_TTL_MS="1000", ROOM_IDLE_MS="600000")
    registry = RoomRegistry(MockBackend(), cfg)
    room = registry.create()
    assert room is not None
    room.join(FakeSocket(), "Alice")
    assert registry.sweep(room.state.created_at + 2000) == 0


# ---------------------------------------------------------------- socket caps


def test_a_room_refuses_sockets_past_its_member_cap() -> None:
    registry = RoomRegistry(MockBackend(), config(MAX_ROOM_SOCKETS="2"))
    room = registry.ensure("aaaa")
    assert room is not None
    assert registry.can_accept_socket(room) is None
    room.join(FakeSocket(), "A")
    room.join(FakeSocket(), "B")
    assert registry.can_accept_socket(room) == "this room is full"


def test_the_process_refuses_sockets_past_the_global_cap() -> None:
    registry = RoomRegistry(MockBackend(), config(MAX_TOTAL_SOCKETS="2"))
    room = registry.ensure("aaaa")
    assert room is not None
    registry.note_socket_open()
    registry.note_socket_open()
    assert registry.can_accept_socket(room) == "the server is holding too many connections"
    registry.note_socket_closed()
    assert registry.can_accept_socket(room) is None


def test_the_socket_cap_is_enforced_over_a_real_connection() -> None:
    with client(MAX_ROOM_SOCKETS="1") as c:
        with c.websocket_connect("/ws/rooms/caproom?name=A") as first:
            assert json.loads(first.receive_text())["t"] == "snapshot"
            with pytest.raises(Exception):
                with c.websocket_connect("/ws/rooms/caproom?name=B") as second:
                    second.receive_text()


# -------------------------------------------------------------- stroke memory


def draw(room, user: str, stroke_id: str, points: List[Dict[str, float]]) -> Any:
    layer = room.layers[0]["id"]
    apply_client_message(
        room,
        user,
        {
            "t": "stroke_start",
            "stroke": {
                "id": stroke_id,
                "layerId": layer,
                "tool": "pen",
                "color": "#000000",
                "width": 4,
                "points": [points[0]],
            },
        },
    )
    return apply_client_message(
        room, user, {"t": "stroke_end", "strokeId": stroke_id, "points": points[1:]}
    )


def test_a_room_refuses_strokes_past_its_aggregate_point_quota() -> None:
    room = create_room("r", max_points=40)
    join_member(room, "Alice")
    user = next(iter(room.members))
    points = [{"x": float(i), "y": 1.0} for i in range(30)]

    first = draw(room, user, "s1", points)
    assert first.broadcast[0]["t"] == "stroke_committed"
    assert room.committed_points == 30

    second = draw(room, user, "s2", points)
    assert second.broadcast[0]["t"] == "stroke_cancel"
    assert second.broadcast[0]["reason"] == "room stroke limit reached"
    # Refused means refused: nothing committed, no revision spent.
    assert len(room.strokes) == 1
    assert room.committed_points == 30


def test_clearing_a_layer_gives_the_quota_back() -> None:
    room = create_room("r", max_points=100)
    join_member(room, "Alice")
    user = next(iter(room.members))
    draw(room, user, "s1", [{"x": float(i), "y": 1.0} for i in range(30)])
    assert room.committed_points == 30
    apply_client_message(room, user, {"t": "clear_layer", "layerId": room.layers[0]["id"]})
    assert room.committed_points == 0


def test_a_snapshot_too_large_to_send_refuses_the_join() -> None:
    from brushjam import runtime as runtime_module

    room = RoomRuntime("big", MockBackend(), config())
    socket = FakeSocket()
    original = runtime_module.MAX_BUFFERED_BYTES
    try:
        runtime_module.MAX_BUFFERED_BYTES = 10  # any real snapshot is larger
        room.join(socket, "Alice")
    finally:
        runtime_module.MAX_BUFFERED_BYTES = original
    assert socket.revoked is True
    assert socket.sent == []
    assert room.member_count == 0


async def test_the_join_snapshot_is_serialised_off_the_event_loop() -> None:
    room = RoomRuntime("async", MockBackend(), config())
    socket = FakeSocket()
    user_id = await room.join_async(socket, "Alice")
    assert socket.types()[0] == "snapshot"
    assert room.member_count == 1
    assert json.loads(socket.sent[0])["snapshot"]["youUserId"] == user_id


# ------------------------------------------------------------------- identity


def a_stroke(room, stroke_id: str) -> str:
    return json.dumps(
        {
            "t": "stroke_start",
            "stroke": {
                "id": stroke_id,
                "layerId": room.state.layers[0]["id"],
                "tool": "pen",
                "color": "#000000",
                "width": 4,
                "points": [{"x": 1, "y": 1}],
            },
        }
    )


async def test_a_superseded_socket_loses_its_identity_immediately() -> None:
    # Async because committing a stroke schedules a generation, which needs a
    # running loop - as it always has in production.
    room = RoomRuntime("resume", MockBackend(), config())
    old = FakeSocket()
    # The client chooses the reconnect token; presenting it again is a resume.
    user_id = room.join(old, "Alice", "tok-alice-0001")

    new = FakeSocket()
    assert room.join(new, "Alice", "tok-alice-0001") == user_id
    # Not "queued behind whatever it had buffered": gone now.
    assert old.revoked is True

    # The old socket is still draining its receive buffer and sends a stroke.
    before = len(room.state.strokes)
    room.handle(user_id, a_stroke(room, "ghost"), old)
    room.handle(
        user_id,
        json.dumps({"t": "stroke_end", "strokeId": "ghost", "points": [{"x": 2, "y": 2}]}),
        old,
    )
    assert len(room.state.strokes) == before
    assert room.state.pending == {}

    # The socket that actually holds the identity is unaffected.
    room.handle(user_id, a_stroke(room, "real"), new)
    room.handle(
        user_id,
        json.dumps({"t": "stroke_end", "strokeId": "real", "points": [{"x": 2, "y": 2}]}),
        new,
    )
    assert len(room.state.strokes) == before + 1
