"""Runtime: image quotas, room eviction, slow clients, capability refresh."""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any, Dict, List

import pytest
from PIL import Image

from brushjam.ai.backends.base import BackendCapabilities
from brushjam.ai.backends.mock import MockBackend
from brushjam.config import load_config
from brushjam.raster import to_png
from brushjam.room import RoomLimits, now_ms
from brushjam.runtime import (
    IMAGE_GRACE_MS,
    MAX_BUFFERED_BYTES,
    MAX_IMAGES_PER_ROOM,
    MAX_ROOMS,
    RoomRegistry,
    RoomRuntime,
    looks_like_limit_error,
)


class FakeSocket:
    def __init__(self, buffered: int = 0) -> None:
        self.sent: List[Dict[str, Any]] = []
        self.buffered_bytes = buffered
        self.open = True
        self.closed = False

    def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def close_now(self) -> None:
        self.closed = True
        self.open = False

    def of_type(self, t: str) -> List[Dict[str, Any]]:
        return [m for m in self.sent if m["t"] == t]


def config(**env):
    settings = {"AI_BACKEND": "mock", "CANVAS_SIZE": "512", "AI_WINDOW": "512"}
    settings.update(env)
    return load_config(settings)


def png(width: int = 8, height: int = 8) -> bytes:
    return to_png(Image.new("RGB", (width, height), (1, 2, 3)))


def test_join_sends_a_snapshot_and_presence() -> None:
    room = RoomRuntime("r1", MockBackend(0), config())
    socket = FakeSocket()
    user = room.join(socket, "Alice")
    assert socket.of_type("snapshot")[0]["snapshot"]["youUserId"] == user
    assert socket.of_type("presence")[0]["members"][0]["name"] == "Alice"


def test_a_superseded_socket_does_not_evict_the_live_one() -> None:
    room = RoomRuntime("r1", MockBackend(0), config())
    first = FakeSocket()
    user = room.join(first, "Alice", "tok-alice-0001")
    second = FakeSocket()
    assert room.join(second, "Alice", "tok-alice-0001") == user
    assert first.closed  # the newest connection wins
    # The old socket's close event arrives afterwards and must be ignored.
    room.leave(user, first)
    assert room.member_count == 1


def test_a_slow_client_is_dropped() -> None:
    room = RoomRuntime("r1", MockBackend(0), config())
    socket = FakeSocket(buffered=MAX_BUFFERED_BYTES + 1)
    room.join(socket, "Alice")
    assert socket.closed
    assert socket.sent == []


async def test_image_quota_and_sweeping() -> None:
    room = RoomRuntime("r1", MockBackend(0), config())
    stored = await room.add_image(png(), "image/png")
    assert "imageId" in stored
    # Not referenced by any layer, but inside the grace period.
    assert room.prune_images(now_ms()) == 0
    assert room.prune_images(now_ms() + IMAGE_GRACE_MS + 1) == 1
    assert room.state.images == {}

    for _ in range(MAX_IMAGES_PER_ROOM):
        assert "imageId" in await room.add_image(png(), "image/png")
    full = await room.add_image(png(), "image/png")
    assert full["error"] == "this room already holds the maximum number of images"


async def test_an_undecodable_upload_is_refused() -> None:
    room = RoomRuntime("r1", MockBackend(0), config())
    assert "error" in await room.add_image(b"garbage", "image/png")


async def test_dirty_regions_are_clipped_to_the_canvas() -> None:
    room = RoomRuntime("r1", MockBackend(0), config())
    socket = FakeSocket()
    user = room.join(socket, "Alice")
    layer = room.state.layers[0]["id"]
    room.handle(
        user,
        json.dumps(
            {
                "t": "stroke_start",
                "stroke": {
                    "id": "s1",
                    "layerId": layer,
                    "tool": "pen",
                    "color": "#000000",
                    "width": 8,
                    "points": [{"x": -400, "y": -400}],
                },
            }
        ),
    )
    room.handle(user, json.dumps({"t": "stroke_end", "strokeId": "s1", "points": [{"x": 40, "y": 40}]}))
    # It committed, and the scheduler was told to run (full mode has no regions).
    assert socket.of_type("stroke_committed")
    assert room.scheduler.state == "queued"
    room.dispose()


def test_a_room_is_reclaimed_once_it_has_been_empty_long_enough() -> None:
    registry = RoomRegistry(MockBackend(0), config(ROOM_IDLE_MS="10000"))
    room = registry.create()
    assert room is not None
    assert registry.sweep(now_ms()) == 0
    assert registry.sweep(now_ms() + 10_001) == 1
    assert registry.size == 0


def test_rooms_are_created_on_demand_and_capped() -> None:
    registry = RoomRegistry(MockBackend(0), config())
    same = registry.ensure("shared01")
    assert same is registry.ensure("shared01")
    for i in range(MAX_ROOMS - 1):
        assert registry.create() is not None
    assert registry.size == MAX_ROOMS
    # Every room is fresh (lastActiveAt is now), so nothing can be swept.
    assert registry.create() is None
    assert registry.ensure("brandnew") is None


async def test_capabilities_are_pushed_into_live_rooms() -> None:
    class Shrinking(MockBackend):
        def __init__(self) -> None:
            super().__init__(0)
            self.limit = 2048

        async def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities(["fast"], self.limit, 0.8, {"fast": True, "quality": True})

    backend = Shrinking()
    registry = RoomRegistry(
        backend,
        config(),
        RoomLimits(profiles=["fast", "quality"], max_denoise=0.95, max_resolution=1024),
    )
    room = registry.ensure("live0001")
    assert room is not None
    socket = FakeSocket()
    room.join(socket, "Alice")
    backend.limit = 512
    await registry.refresh_capabilities()
    caps = socket.of_type("ai_capabilities")[-1]
    assert caps["aiProfiles"] == ["fast"]
    assert caps["maxDenoise"] == 0.8
    assert caps["aiResolutionMax"] == 512
    assert room.state.ai_resolution == 512
    registry.dispose()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("size 512 out of range", True),
        ("status 400 from worker", True),
        ("connection refused", False),
        ("model not supported", True),
    ],
)
def test_limit_error_detection(message: str, expected: bool) -> None:
    assert looks_like_limit_error(message) is expected
