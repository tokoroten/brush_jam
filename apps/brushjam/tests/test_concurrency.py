"""Two rooms at once, and one canvas touched from two places.

The single-room tests cannot see any of this: everything here only goes wrong
when a second room, a second socket or a second thread exists.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image
from starlette.testclient import TestClient

from brushjam.ai.backends.mock import MockBackend
from brushjam.app import _contained, create_app, prebuild_full_masks
from brushjam.config import load_config
from brushjam.raster import (
    _decode,
    _full_mask,
    build_full_mask,
    decoded_cache_stats,
    forget_images,
    to_png,
)
from brushjam.runtime import RoomRuntime
from brushjam.scheduler import GenerationAdmission


def config(**env):
    settings = {
        "AI_BACKEND": "mock",
        "CANVAS_SIZE": "512",
        "AI_WINDOW": "512",
        "AI_DEBOUNCE_MS": "10",
    }
    settings.update(env)
    return load_config(settings)


# ------------------------------------------------------------------ admission


async def test_admission_is_one_at_a_time_and_first_come_first_served() -> None:
    admission = GenerationAdmission()
    order: List[int] = []
    inside: List[int] = []
    release = asyncio.Event()

    async def worker(n: int) -> None:
        async with admission.admit():
            inside.append(n)
            order.append(n)
            await release.wait()
            inside.remove(n)

    tasks = [asyncio.create_task(worker(n)) for n in range(4)]
    # Let each task reach the semaphore in creation order.
    for _ in range(8):
        await asyncio.sleep(0)
    assert inside == [0], "more than one holder at a time"
    assert admission.waiting == 3

    release.set()
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2, 3], "admission was not first come, first served"


async def test_nothing_is_rendered_before_the_room_is_admitted() -> None:
    """The whole point: a queued room must not have rasterised anything yet."""
    admission = GenerationAdmission()
    rendered: List[str] = []

    async def room(tag: str, hold: asyncio.Event) -> None:
        async with admission.admit():
            rendered.append(tag)
            await hold.wait()

    hold = asyncio.Event()
    first = asyncio.create_task(room("a", hold))
    for _ in range(4):
        await asyncio.sleep(0)
    second = asyncio.create_task(room("b", hold))
    for _ in range(4):
        await asyncio.sleep(0)

    assert rendered == ["a"], "the queued room rendered while waiting for the GPU"
    hold.set()
    await asyncio.gather(first, second)
    assert rendered == ["a", "b"]


async def test_every_room_in_a_registry_shares_one_admission() -> None:
    from brushjam.runtime import RoomRegistry

    registry = RoomRegistry(MockBackend(), config())
    one = registry.get_or_create("aaaa", "test")
    two = registry.get_or_create("bbbb", "test")
    assert one is not None and two is not None
    assert one.scheduler.opts.admission is two.scheduler.opts.admission
    assert one.scheduler.opts.admission is registry.admission


# ------------------------------------------------------------------ AI canvas


async def test_the_ai_canvas_is_not_composited_and_encoded_at_once() -> None:
    room = RoomRuntime("canvas", MockBackend(), config())
    size = room.config.canvas_size
    patch = to_png(Image.new("RGB", (size, size), (10, 20, 30)))
    mask = build_full_mask(size).alpha
    rect = {"x": 0, "y": 0, "width": size, "height": size}

    overlap = 0
    depth = 0
    lock = threading.Lock()
    canvas_composite = room._composite

    def watched(*args: Any, **kwargs: Any) -> bytes:
        nonlocal overlap, depth
        with lock:
            depth += 1
            if depth > 1:
                overlap += 1
        try:
            return canvas_composite(*args, **kwargs)
        finally:
            with lock:
                depth -= 1

    room._composite = watched  # type: ignore[assignment]

    async def composite() -> None:
        async with room._ai_lock:
            await asyncio.to_thread(room._composite, patch, rect, mask)

    await asyncio.gather(*(composite() for _ in range(4)), room.ai_png(), room.ai_png())
    assert overlap == 0


async def test_ai_png_is_awaited_rather_than_encoded_on_the_loop() -> None:
    room = RoomRuntime("png", MockBackend(), config())
    body = await room.ai_png()
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


# ------------------------------------------------------------- decoded images


def test_the_decoded_image_cache_counts_a_racing_decode_once() -> None:
    forget_images(["shared"])
    before = decoded_cache_stats()
    data = to_png(Image.new("RGBA", (64, 64), (1, 2, 3, 255)))

    results: List[Image.Image] = []
    barrier = threading.Barrier(4)

    def decode() -> None:
        barrier.wait()
        results.append(_decode("shared", data))

    threads = [threading.Thread(target=decode) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    after = decoded_cache_stats()
    assert after["entries"] == before["entries"] + 1
    # 64*64 counted once, however many threads decoded it.
    assert after["pixels"] == before["pixels"] + 64 * 64
    assert len(results) == 4
    forget_images(["shared"])


def test_cached_images_are_not_shared_mutable_objects() -> None:
    forget_images(["immutable"])
    data = to_png(Image.new("RGBA", (8, 8), (9, 9, 9, 255)))
    first = _decode("immutable", data)
    second = _decode("immutable", data)
    assert first is not second, "two renders must not share one PIL image"
    # Backed by the same read-only pixels, so a write cannot corrupt the cache.
    import numpy as np

    with pytest.raises(ValueError):
        np.asarray(first)[0, 0, 0] = 255
    forget_images(["immutable"])


def test_forgetting_an_image_gives_the_budget_back() -> None:
    forget_images(["temp"])
    before = decoded_cache_stats()
    _decode("temp", to_png(Image.new("RGBA", (32, 32), (0, 0, 0, 255))))
    assert decoded_cache_stats()["pixels"] == before["pixels"] + 32 * 32
    forget_images(["temp"])
    assert decoded_cache_stats() == before


# -------------------------------------------------------------------- masks


def test_masks_for_every_allowed_size_are_built_at_startup() -> None:
    _full_mask.cache_clear()
    cfg = config(CANVAS_SIZE="1024", AI_WINDOW="768")
    prebuild_full_masks(cfg)
    hits_before = _full_mask.cache_info().currsize
    assert hits_before >= 3  # 512, 768, 1024
    misses = _full_mask.cache_info().misses
    for size in (512, 768, 1024):
        build_full_mask(size)
    # Every size a room can pick was already built: no first-use stall.
    assert _full_mask.cache_info().misses == misses


def test_the_same_mask_is_returned_rather_than_rebuilt() -> None:
    assert build_full_mask(256) is build_full_mask(256)


# ------------------------------------------------------------ static serving


def test_static_paths_are_contained_by_resolved_path_not_by_prefix(tmp_path: Path) -> None:
    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("<html></html>", encoding="utf-8")
    # A sibling whose name merely starts with the root's string.
    sibling = tmp_path / "dist-secrets"
    sibling.mkdir()
    (sibling / "key.txt").write_text("secret", encoding="utf-8")

    assert _contained(root / "index.html", root) is True
    assert _contained(root, root) is True
    assert _contained(sibling / "key.txt", root) is False
    assert str(sibling / "key.txt").startswith(str(root)), "the old check would have passed this"


def test_a_traversal_path_falls_back_to_the_spa(tmp_path: Path) -> None:
    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    cfg = config(WEB_DIST=str(root))
    with TestClient(create_app(cfg, MockBackend(latency_ms=0))) as c:
        body = c.get("/../secret.txt").text
    assert "secret" not in body
