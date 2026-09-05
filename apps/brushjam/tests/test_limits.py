"""Capacity, admission and identity: the ways a shared server gets exhausted.

Every case here is something an unauthenticated client can do on purpose, so
the assertions are about what the server refuses, not about what it manages.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
from starlette.testclient import TestClient

from brushjam.ai.backends.mock import MockBackend
from brushjam.app import create_app
from brushjam.config import load_config
from brushjam.room import apply_client_message, create_room, join_member, now_ms
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


class HoldingSocket(FakeSocket):
    """A FakeSocket with the real connection's hold/release behaviour."""

    def __init__(self) -> None:
        super().__init__()
        self._held = True
        self._buffer: List[str] = []

    def send_text(self, data: str) -> None:
        if self._held:
            self._buffer.append(data)
            return
        self.sent.append(data)

    def release(self, first: Optional[str] = None) -> None:
        if not self._held:
            if first is not None:
                self.sent.append(first)
            return
        self._held = False
        if first is not None:
            self.sent.append(first)
        self.sent.extend(self._buffer)
        self._buffer = []


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
    room = registry.create("test")
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
    room = registry.create("test")
    assert room is not None
    room.join(FakeSocket(), "Alice")
    assert registry.sweep(room.state.created_at + 2000) == 0


# ---------------------------------------------------------------- socket caps


def test_a_room_refuses_sockets_past_its_member_cap() -> None:
    registry = RoomRegistry(MockBackend(), config(MAX_ROOM_SOCKETS="2"))
    room = registry.get_or_create("aaaa", "test")
    assert room is not None
    # A live socket holds its reservation for its whole life, so two joined
    # members are two reservations.
    assert registry.reserve_socket(room) is not None
    room.join(FakeSocket(), "A")
    assert registry.reserve_socket(room) is not None
    room.join(FakeSocket(), "B")
    assert registry.reserve_socket(room) is None
    assert registry.can_accept_socket(room) == "this room is full"


def test_pending_handshakes_count_against_the_room_cap() -> None:
    """Reservations are taken before the first await, so N concurrent
    handshakes at the cap admit exactly the cap."""
    registry = RoomRegistry(MockBackend(), config(MAX_ROOM_SOCKETS="3"))
    room = registry.get_or_create("aaaa", "test")
    assert room is not None
    taken = [registry.reserve_socket(room) for _ in range(8)]
    assert sum(1 for t in taken if t is not None) == 3
    assert room.reserved_sockets == 3
    # Idempotent release: the route finally, a revoke and a failed join all
    # want to give the same slot back.
    first = next(t for t in taken if t is not None)
    first.release()
    first.release()
    assert room.reserved_sockets == 2
    assert registry.sockets_open == 2


def test_a_resume_takes_over_the_lease_it_replaces() -> None:
    registry = RoomRegistry(MockBackend(), config(MAX_ROOM_SOCKETS="1"))
    room = registry.get_or_create("aaaa", "test")
    assert room is not None
    first = registry.reserve_socket(room)
    assert first is not None
    room.join(FakeSocket(), "A", "tok-a-0001", first)

    # A newcomer is refused...
    assert registry.reserve_socket(room) is None
    # ...but the reconnect takes over the slot the socket it replaces holds.
    resumed = registry.reserve_socket(room, "tok-a-0001")
    assert resumed is not None
    assert registry.sockets_open == 1 and room.reserved_sockets == 1
    # The old route's release finds nothing to give back.
    first.release()
    assert registry.sockets_open == 1 and room.reserved_sockets == 1
    resumed.release()
    assert registry.sockets_open == 0 and room.reserved_sockets == 0


def test_many_reconnects_with_one_token_cannot_raise_the_counters() -> None:
    """One valid token used to raise a limit of 1 to 21."""
    registry = RoomRegistry(MockBackend(), config(MAX_ROOM_SOCKETS="1", MAX_TOTAL_SOCKETS="1"))
    room = registry.get_or_create("aaaa", "test")
    assert room is not None
    lease = registry.reserve_socket(room)
    assert lease is not None
    socket = FakeSocket()
    room.join(socket, "A", "tok-a-0001", lease)

    held = [lease]
    for _ in range(20):
        nxt = registry.reserve_socket(room, "tok-a-0001")
        if nxt is None:
            continue
        # Each reconnect must install itself the way the route does, or the
        # next one has nothing to take over.
        socket = FakeSocket()
        room.join(socket, "A", "tok-a-0001", nxt)
        held.append(nxt)
        assert registry.sockets_open <= 1
        assert room.reserved_sockets <= 1

    assert registry.sockets_open == 1
    assert room.reserved_sockets == 1
    for reservation in held:
        reservation.release()
    assert registry.sockets_open == 0
    assert room.reserved_sockets == 0


def test_the_process_refuses_sockets_past_the_global_cap() -> None:
    registry = RoomRegistry(MockBackend(), config(MAX_TOTAL_SOCKETS="2"))
    room = registry.get_or_create("aaaa", "test")
    assert room is not None
    first = registry.reserve_socket(room)
    second = registry.reserve_socket(room)
    assert first is not None and second is not None
    assert registry.reserve_socket(room) is None
    assert registry.can_accept_socket(room) == "the server is holding too many connections"
    second.release()
    assert registry.reserve_socket(room) is not None


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
    assert second.broadcast[0]["reason"] == "quota"
    # Refused means refused: nothing committed, no revision spent.
    assert len(room.strokes) == 1
    assert room.committed_points == 30


def test_a_room_refuses_strokes_past_its_snapshot_byte_budget() -> None:
    """The budget that decides whether the room stays joinable at all.

    A room may not accumulate a log that serialises past the outbound frame
    cap: every later join and every reconnect would be refused.
    """
    from brushjam.room import stroke_snapshot_bytes

    points = [{"x": float(i), "y": 1.0} for i in range(100)]
    probe = create_room("probe", max_points=10_000_000)
    join_member(probe, "Probe")
    draw(probe, next(iter(probe.members)), "p1", points)
    one_stroke = probe.snapshot_bytes
    assert one_stroke == stroke_snapshot_bytes(probe.strokes[0])

    room = create_room("r", max_points=10_000_000, max_snapshot_bytes=one_stroke + 10)
    join_member(room, "Alice")
    user = next(iter(room.members))

    assert draw(room, user, "s1", points).broadcast[0]["t"] == "stroke_committed"
    assert room.snapshot_bytes == one_stroke
    refused = draw(room, user, "s2", points)
    assert refused.broadcast[0]["t"] == "stroke_cancel"
    assert refused.broadcast[0]["reason"] == "quota"
    assert len(room.strokes) == 1


def test_a_full_room_still_serialises_under_the_outbound_cap() -> None:
    """The estimate has to be an over-estimate, or the budget is decorative."""
    import json

    from brushjam.room import snapshot
    from brushjam.runtime import MAX_BUFFERED_BYTES

    room = create_room("r", max_points=5_000_000)
    join_member(room, "Alice")
    user = next(iter(room.members))
    # Fill to the byte budget with realistic strokes.
    points = [{"x": 1234.5, "y": 6789.25, "p": 0.5} for _ in range(2000)]
    n = 0
    while draw(room, user, f"s{n}", points).broadcast[0]["t"] == "stroke_committed":
        n += 1
        assert n < 500, "the budget never refused a stroke"
    assert n > 10
    data = json.dumps(
        snapshot(room, user, "idle", {"window": 768, "apply": 768, "canvasSize": 1024}),
        separators=(",", ":"),
    )
    assert len(data) <= room.max_snapshot_bytes, "the accounting under-counts"
    assert len(data) < MAX_BUFFERED_BYTES


def test_deleting_a_layer_gives_the_budget_back() -> None:
    room = create_room("r")
    join_member(room, "Alice")
    user = next(iter(room.members))
    apply_client_message(
        room, user, {"t": "layer_create", "layer": {"kind": "draw", "name": "Two"}}
    )
    second = room.layers[-1]["id"]
    layer = room.layers[0]["id"]
    draw(room, user, "s1", [{"x": float(i), "y": 1.0} for i in range(30)])
    assert room.snapshot_bytes > 0
    assert second != layer
    apply_client_message(room, user, {"t": "layer_delete", "id": layer})
    assert room.committed_points == 0
    assert room.snapshot_bytes == 0


def test_clearing_a_layer_gives_the_quota_back() -> None:
    room = create_room("r", max_points=100)
    join_member(room, "Alice")
    user = next(iter(room.members))
    draw(room, user, "s1", [{"x": float(i), "y": 1.0} for i in range(30)])
    assert room.committed_points == 30
    apply_client_message(room, user, {"t": "clear_layer", "layerId": room.layers[0]["id"]})
    assert room.committed_points == 0
    assert room.snapshot_bytes == 0


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


# ------------------------------------------------ creation, every entry point


def test_a_websocket_to_an_unknown_room_is_rate_limited_too() -> None:
    """The POST is not the only way to make a room; the link is the usual way."""
    with client(ROOM_CREATE_PER_MIN="2") as c:
        with c.websocket_connect("/ws/rooms/wsroom01?name=A") as first:
            assert json.loads(first.receive_text())["t"] == "snapshot"
        with c.websocket_connect("/ws/rooms/wsroom02?name=A") as second:
            assert json.loads(second.receive_text())["t"] == "snapshot"
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/rooms/wsroom03?name=A") as third:
                third.receive_text()


def test_an_upload_to_an_unknown_room_is_rate_limited_too() -> None:
    from PIL import Image

    from brushjam.raster import to_png

    body = to_png(Image.new("RGB", (8, 8), (1, 2, 3)))
    headers = {"content-type": "image/png"}
    with client(ROOM_CREATE_PER_MIN="1") as c:
        first = c.post("/rooms/uproom001/images", content=body, headers=headers)
        second = c.post("/rooms/uproom002/images", content=body, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 429


def test_get_never_creates_a_room() -> None:
    registry = RoomRegistry(MockBackend(), config())
    assert registry.get("nothing1") is None
    assert registry.size == 0


def test_a_refused_connection_does_not_leave_a_room_behind() -> None:
    """Capacity is decided before the room is made, so a rejected socket
    cannot have spent one of the 64 room slots."""
    with client(ROOM_CREATE_PER_MIN="1") as c:
        with c.websocket_connect("/ws/rooms/allowed1?name=A") as ok:
            ok.receive_text()
            with pytest.raises(Exception):
                with c.websocket_connect("/ws/rooms/refused1?name=B") as no:
                    no.receive_text()
        rooms = c.get("/healthz").json()["rooms"]
    assert rooms == 1


def test_a_connection_that_never_joined_does_not_extend_the_room_lifetime() -> None:
    cfg = config(UNJOINED_ROOM_TTL_MS="1000", ROOM_IDLE_MS="600000")
    registry = RoomRegistry(MockBackend(), cfg)
    room = registry.create_named("aaaa", "test")
    assert room is not None
    # A socket that took a reservation but never completed a join.
    reservation = registry.reserve_socket(room)
    assert reservation is not None
    reservation.release()
    assert room.ever_joined is False
    assert registry.sweep(room.state.created_at + 2000) == 1


def test_the_rate_limiter_is_bounded_by_address_count() -> None:
    from brushjam.runtime import MAX_RATE_BUCKETS

    registry = RoomRegistry(MockBackend(), config(ROOM_CREATE_PER_MIN="1"))
    now = 1000.0
    for i in range(5000):
        address = "10.0.%d.%d" % (i // 256, i % 256)
        # Twice each: the second is refused, and the refused path has to prune
        # as well or the table grows on exactly the traffic that matters.
        registry.allow_create(address, now)
        registry.allow_create(address, now)
    assert len(registry._create_buckets) <= MAX_RATE_BUCKETS


def test_a_refilled_bucket_is_dropped_rather_than_kept_forever() -> None:
    registry = RoomRegistry(MockBackend(), config(ROOM_CREATE_PER_MIN="10"))
    for i in range(100):
        registry.allow_create("1.0.0.%d" % i, 0.0)
    assert len(registry._create_buckets) == 100
    # A minute later every one of them has refilled, and a full bucket says
    # nothing a fresh one would not.
    registry.allow_create("2.2.2.2", 120.0)
    assert len(registry._create_buckets) == 1


# ------------------------------------------- join ordering and join failures


async def test_a_broadcast_during_serialisation_arrives_after_the_snapshot() -> None:
    """The snapshot is first, and nothing sent while it was being built is
    lost - it is delivered behind it, in order."""
    room = RoomRuntime("order", MockBackend(), config())
    room.join(FakeSocket(), "Bob")

    socket = HoldingSocket()
    join = asyncio.ensure_future(room.join_async(socket, "Alice"))
    await asyncio.sleep(0)  # the join is now inside its threaded dumps
    room.broadcast({"t": "prompt_changed", "prompt": "during", "humanRevision": 1})
    room.broadcast({"t": "clear_applied", "layerId": "L0", "humanRevision": 2})
    await join

    types = socket.types()
    assert types[0] == "snapshot"
    assert types[1:3] == ["prompt_changed", "clear_applied"]


async def test_a_join_that_fails_leaves_no_member_and_no_reservation() -> None:
    registry = RoomRegistry(MockBackend(), config())
    room = registry.create_named("failjoin", "test")
    assert room is not None
    reservation = registry.reserve_socket(room)
    assert reservation is not None

    class Exploding(FakeSocket):
        def send_text(self, data: str) -> None:
            raise RuntimeError("transport gone")

    socket = Exploding()
    user_id = None
    try:
        user_id = await room.join_async(socket, "Alice")
    except Exception:
        pass
    finally:
        room.rollback_join(user_id, socket)
        reservation.release()

    assert room.member_count == 0
    assert room.state.members == {}
    assert registry.sockets_open == 0
    assert room.reserved_sockets == 0


async def test_an_old_socket_leaving_during_a_resume_keeps_the_member() -> None:
    """The gap this closes: the old route reaching leave() used to find no
    mapping and remove the member the new socket had just resumed."""
    room = RoomRuntime("gap", MockBackend(), config())
    old = FakeSocket()
    user_id = room.join(old, "Alice", "tok-gap-0001")
    room.handle(user_id, a_stroke(room, "mid"), old)
    assert room.state.pending

    new = HoldingSocket()
    join = asyncio.ensure_future(room.join_async(new, "Alice", "tok-gap-0001"))
    await asyncio.sleep(0)
    # The old route notices its socket closed, mid-resume.
    room.leave(user_id, old)
    await join

    assert user_id in room.state.members, "the resumed member was removed"
    assert room.member_count == 1
    assert room.state.pending, "the resumed user's in-progress stroke was cancelled"


# ------------------------------------------------------------ pending strokes


def start(room, user: str, stroke_id: str, points: List[Dict[str, float]]) -> Any:
    return apply_client_message(
        room,
        user,
        {
            "t": "stroke_start",
            "stroke": {
                "id": stroke_id,
                "layerId": room.layers[0]["id"],
                "tool": "pen",
                "color": "#000000",
                "width": 4,
                "points": points,
            },
        },
    )


def chunk(room, user: str, stroke_id: str, points: List[Dict[str, float]]) -> Any:
    return apply_client_message(
        room, user, {"t": "stroke_chunk", "strokeId": stroke_id, "points": points}
    )


def test_strokes_in_progress_are_charged_to_the_room_budget() -> None:
    """A stroke nobody finishes occupies the room exactly as much as one
    somebody does, and the commit-time check never saw it."""
    room = create_room("r", max_points=50)
    join_member(room, "Alice")
    user = next(iter(room.members))
    points = [{"x": float(i), "y": 1.0} for i in range(30)]

    assert start(room, user, "s1", points).relay
    assert room.pending_points == 30
    refused = chunk(room, user, "s1", points)
    assert refused.broadcast[0]["t"] == "stroke_cancel"
    assert refused.broadcast[0]["reason"] == "quota"
    # Refused and dropped, so the reservation went with it.
    assert room.pending_points == 0
    assert room.pending == {}


def test_a_new_stroke_is_refused_when_the_room_is_already_full_of_pending() -> None:
    room = create_room("r", max_points=40)
    join_member(room, "Alice")
    user = next(iter(room.members))
    points = [{"x": float(i), "y": 1.0} for i in range(30)]
    assert start(room, user, "s1", points).relay
    refused = start(room, user, "s2", points)
    assert refused.to_sender[0]["t"] == "stroke_cancel"
    assert refused.to_sender[0]["reason"] == "quota"
    assert room.pending_points == 30


def test_many_members_cannot_hold_more_pending_than_the_room_allows() -> None:
    """Sixteen members times four strokes of fifty thousand points was 3.2
    million point dicts that no committed budget ever counted."""
    from brushjam.room import MAX_PENDING_PER_USER

    room = create_room("r", max_points=5_000)
    users = []
    for i in range(16):
        join_member(room, "U%d" % i)
    users = list(room.members)
    points = [{"x": float(i), "y": 1.0} for i in range(200)]

    for user in users:
        for n in range(MAX_PENDING_PER_USER):
            start(room, user, "s%s%d" % (user, n), points)
            for _ in range(5):
                chunk(room, user, "s%s%d" % (user, n), points)

    assert room.committed_points + room.pending_points <= room.max_points


def test_every_way_a_pending_stroke_ends_gives_its_points_back() -> None:
    from brushjam.room import expire_pending_strokes, remove_member

    points = [{"x": float(i), "y": 1.0} for i in range(20)]

    # committed
    room = create_room("r")
    join_member(room, "Alice")
    user = next(iter(room.members))
    start(room, user, "s1", points)
    apply_client_message(room, user, {"t": "stroke_end", "strokeId": "s1", "points": []})
    assert room.pending_points == 0
    assert room.committed_points == 20

    # the author leaves
    start(room, user, "s2", points)
    assert room.pending_points == 20
    remove_member(room, user)
    assert room.pending_points == 0

    # the layer is cleared
    join_member(room, "Bob")
    bob = next(u for u in room.members)
    start(room, bob, "s3", points)
    assert room.pending_points == 20
    apply_client_message(room, bob, {"t": "clear_layer", "layerId": room.layers[0]["id"]})
    assert room.pending_points == 0

    # it is abandoned
    start(room, bob, "s4", points)
    assert room.pending_points == 20
    expire_pending_strokes(room, now_ms() + 10 * 60_000)
    assert room.pending_points == 0


# ------------------------------------------------------- sweeping and buffers


def test_a_room_with_a_handshake_in_flight_is_never_swept() -> None:
    """The reservation is taken before the socket is accepted; sweeping in
    that instant left the route joining a room the registry had forgotten."""
    cfg = config(UNJOINED_ROOM_TTL_MS="1000", ROOM_IDLE_MS="10000")
    registry = RoomRegistry(MockBackend(), cfg)
    room = registry.create_named("pinned01", "test")
    assert room is not None
    reservation = registry.reserve_socket(room)
    assert reservation is not None

    assert registry.sweep(room.state.created_at + 60_000) == 0
    assert registry.get("pinned01") is room

    reservation.release()
    assert registry.sweep(room.state.created_at + 60_000) == 1


def test_a_frame_that_would_cross_the_cap_is_refused_before_it_is_queued() -> None:
    from brushjam.runtime import MAX_BUFFERED_BYTES

    room = RoomRuntime("cap", MockBackend(), config())
    socket = FakeSocket()
    room.join(socket, "Alice")
    socket.buffered = MAX_BUFFERED_BYTES - 10
    assert socket.sent, "the join itself should have fitted"
    socket.sent.clear()

    room.broadcast({"t": "prompt_changed", "prompt": "x" * 100, "humanRevision": 1})
    assert socket.sent == [], "a frame that crosses the cap was queued anyway"
    assert socket.open is False


async def test_a_snapshot_that_does_not_fit_beside_its_held_frames_is_refused() -> None:
    """They are released together, so they have to fit together."""
    from brushjam import runtime as runtime_module

    room = RoomRuntime("fit", MockBackend(), config())
    socket = HoldingSocket()
    original = runtime_module.MAX_BUFFERED_BYTES
    try:
        runtime_module.MAX_BUFFERED_BYTES = 4096
        join = asyncio.ensure_future(room.join_async(socket, "Alice"))
        await asyncio.sleep(0)
        # Held frames that leave no room for the snapshot behind them.
        socket.buffered = 4000
        await join
    finally:
        runtime_module.MAX_BUFFERED_BYTES = original

    assert socket.revoked is True
    assert socket.sent == []
    assert room.member_count == 0


# -------------------------------------------------------------------- uploads


def test_upload_bodies_have_a_process_wide_budget() -> None:
    from brushjam.runtime import MAX_IMAGE_BYTES_PER_SLOT, UploadGate

    gate = UploadGate(max_concurrent=4, max_bytes=3 * MAX_IMAGE_BYTES_PER_SLOT)
    slots = []
    for i in range(4):
        slot = gate.acquire("10.0.0.%d" % i, MAX_IMAGE_BYTES_PER_SLOT)
        if slot is not None:
            slots.append(slot)
    # Three fit in the byte budget; the fourth does not, whatever the count says.
    assert len(slots) == 3
    assert gate.bytes_in_flight == 3 * MAX_IMAGE_BYTES_PER_SLOT
    slots[0].release()
    slots[0].release()
    assert gate.in_flight == 2
    assert gate.acquire("10.0.0.9", MAX_IMAGE_BYTES_PER_SLOT) is not None


def test_one_address_gets_one_upload_at_a_time() -> None:
    from brushjam.runtime import UploadGate

    gate = UploadGate()
    first = gate.acquire("1.1.1.1", 1024)
    assert first is not None
    assert gate.acquire("1.1.1.1", 1024) is None
    assert gate.acquire("2.2.2.2", 1024) is not None
    first.release()
    assert gate.acquire("1.1.1.1", 1024) is not None


def test_an_upload_slot_is_released_even_when_the_body_is_rejected() -> None:
    from PIL import Image

    from brushjam.raster import to_png

    body = to_png(Image.new("RGB", (8, 8), (1, 2, 3)))
    with client() as c:
        registry = c.app.state.registry
        for _ in range(3):
            c.post("/rooms/uphold01/images", content=body, headers={"content-type": "image/png"})
            # Rejected or not, the slot is given back.
            assert registry.uploads.in_flight == 0
        bad = c.post(
            "/rooms/uphold01/images", content=b"not an image", headers={"content-type": "image/png"}
        )
        assert bad.status_code == 400
        assert registry.uploads.in_flight == 0
        assert registry.uploads.bytes_in_flight == 0
