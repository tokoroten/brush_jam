"""Full-mode scheduler: debounce, one in flight, stale results, error policy."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import pytest

from brushjam.ai.backends.base import BackendCapabilities, BackendHttpError, GenerateRequest
from brushjam.scheduler import (
    AIScheduler,
    GenerationAdmission,
    RenderJob,
    SchedulerOptions,
    is_permanent_error,
)


class FakeBackend:
    name = "fake"

    def __init__(self, fail: Optional[Exception] = None, latency_ms: float = 0) -> None:
        self.fail = fail
        self.latency_ms = latency_ms
        self.requests: List[GenerateRequest] = []
        #: Set as soon as a generation is under way, so a test can act on a run
        #: that has definitely already captured the room state.
        self.started = asyncio.Event()

    async def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(["fast", "quality"], 2048, 0.95)

    async def generate(self, req: GenerateRequest) -> bytes:
        self.requests.append(req)
        self.started.set()
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if self.fail is not None:
            raise self.fail
        return b"patch"


class FakeHost:
    def __init__(self) -> None:
        self.revision = 0
        self.messages: List[Dict[str, Any]] = []
        self.applied: List[int] = []
        self.errors: List[tuple] = []
        self.renders = 0
        self.profile = "fast"

    def get_revision(self) -> int:
        return self.revision

    def begin_job(self) -> RenderJob:
        revision = self.revision

        def render(crop, size):
            self.renders += 1
            return b"input"

        return RenderJob(
            revision=revision,
            prompt="a town",
            denoise=0.7,
            negative_prompt="",
            resolution=512,
            profile=self.profile,
            render=render,
        )

    def build_full_mask(self, size: int):
        class _Mask:
            png = b"mask"
            alpha = object()
            empty = False

        return _Mask()

    async def apply_result(self, patch, crop, apply, mask, for_revision):
        self.applied.append(for_revision)
        return {"rect": crop, "url": f"/patch/{for_revision}.png", "aiGeneration": len(self.applied)}

    def emit(self, msg) -> None:
        self.messages.append(msg)

    def on_error(self, message: str, repeated: int) -> None:
        self.errors.append((message, repeated))

    def states(self) -> List[str]:
        return [m["state"] for m in self.messages if m["t"] == "ai_status"]

    def results(self) -> List[Dict[str, Any]]:
        return [m for m in self.messages if m["t"] == "ai_result"]


def make(host: FakeHost, backend: FakeBackend, **opts) -> AIScheduler:
    options = SchedulerOptions(
        window=512,
        apply=1024,
        steps=14,
        fast_steps=4,
        denoise=0.7,
        debounce_ms=20,
        canvas_size=1024,
        error_backoff_ms=20,
        seed=lambda: 7,
    )
    for key, value in opts.items():
        setattr(options, key, value)
    return AIScheduler(host, backend, options)


async def settle(ms: float = 200) -> None:
    await asyncio.sleep(ms / 1000)


async def test_activity_is_debounced_into_one_generation() -> None:
    host, backend = FakeHost(), FakeBackend()
    scheduler = make(host, backend)
    for _ in range(5):
        host.revision += 1
        scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
        await asyncio.sleep(0.005)
    await settle()
    assert len(backend.requests) == 1
    assert host.applied == [5]
    assert host.results()[0]["url"] == "/patch/5.png"
    scheduler.stop()


async def test_a_change_during_a_run_queues_exactly_one_more() -> None:
    host, backend = FakeHost(), FakeBackend(latency_ms=60)
    scheduler = make(host, backend)
    host.revision = 1
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await backend.started.wait()  # definitely inside the generation
    for _ in range(3):
        host.revision += 1
        scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await settle(400)
    assert len(backend.requests) == 2
    assert host.applied == [1, 4]
    scheduler.stop()


async def test_the_fast_profile_runs_fewer_steps() -> None:
    host, backend = FakeHost(), FakeBackend()
    scheduler = make(host, backend)
    host.profile = "fast"
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await settle()
    assert backend.requests[0].steps == 4
    host.profile = "quality"
    scheduler.nudge()
    await settle()
    assert backend.requests[1].steps == 14
    scheduler.stop()


async def test_an_empty_negative_prompt_falls_back_to_the_built_in_list() -> None:
    host, backend = FakeHost(), FakeBackend()
    scheduler = make(host, backend)
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await settle()
    assert "worst quality" in backend.requests[0].negative_prompt
    scheduler.stop()


async def test_a_prompt_change_mid_flight_runs_again() -> None:
    host, backend = FakeHost(), FakeBackend(latency_ms=60)
    scheduler = make(host, backend)
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    # Only a nudge that lands after the job was captured is "mid-flight": one
    # that arrives before it is simply picked up by the run itself.
    await backend.started.wait()
    scheduler.nudge()  # someone typed a new prompt while it was generating
    await settle(400)
    assert len(backend.requests) == 2
    scheduler.stop()


async def test_a_refusal_stops_the_retries_until_something_changes() -> None:
    host, backend = FakeHost(), FakeBackend(fail=BackendHttpError("size out of range", 400))
    scheduler = make(host, backend, max_repeated_errors=2)
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await settle(300)
    assert len(backend.requests) == 2  # tried twice, then gave up
    assert host.states()[-1] == "error"
    assert "not retrying" in host.messages[-1]["message"]
    # A new edit is a different request, so it is tried again.
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await settle(300)
    assert len(backend.requests) > 2
    scheduler.stop()


async def test_a_transient_failure_keeps_retrying() -> None:
    host, backend = FakeHost(), FakeBackend(fail=ConnectionRefusedError("ECONNREFUSED"))
    scheduler = make(host, backend, max_repeated_errors=2)
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await settle(300)
    assert len(backend.requests) >= 3
    assert host.states()[-1] == "error"
    scheduler.stop()


async def test_the_watchdog_abandons_a_stuck_generation() -> None:
    host, backend = FakeHost(), FakeBackend(latency_ms=5000)
    scheduler = make(host, backend, watchdog_ms=40, error_backoff_ms=5000)
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await settle(200)
    assert host.states()[-1] == "error"
    assert host.messages[-1]["message"].startswith("generation timed out")
    scheduler.stop()


async def test_stop_cancels_the_pending_timer() -> None:
    host, backend = FakeHost(), FakeBackend()
    scheduler = make(host, backend, debounce_ms=100)
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    scheduler.stop()
    await settle()
    assert backend.requests == []


@pytest.mark.parametrize(
    "err,permanent",
    [
        (BackendHttpError("nope", 400), True),
        (BackendHttpError("nope", 503), False),
        ("size 512 out of range", True),
        ("connection timed out", False),
        ("fetch failed", False),
        ("failed: 404 not found", True),
    ],
)
def test_permanent_error_classification(err, permanent: bool) -> None:
    assert is_permanent_error(err) is permanent


# ------------------------------------------------------------------ admission


async def test_a_queued_room_does_not_render_until_it_is_admitted() -> None:
    """Rendering before the GPU queue is what made N rooms cost N rasters."""
    admission = GenerationAdmission()
    slow = FakeBackend(latency_ms=120)
    first_host, second_host = FakeHost(), FakeHost()
    first = make(first_host, slow, admission=admission)
    second = make(second_host, FakeBackend(), admission=admission)

    first_host.revision += 1
    first.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await asyncio.sleep(0.05)
    second_host.revision += 1
    second.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await asyncio.sleep(0.05)

    assert first_host.renders == 1
    assert second_host.renders == 0, "the queued room rasterised while waiting"

    await settle(400)
    assert second_host.renders == 1
    first.stop()
    second.stop()


async def test_the_watchdog_does_not_count_time_spent_queueing() -> None:
    """A run that timed out while waiting would retry, doubling the backlog."""
    admission = GenerationAdmission()
    # The watchdog is shorter than the run ahead of it in the queue.
    slow = FakeBackend(latency_ms=150)
    first_host, second_host = FakeHost(), FakeHost()
    first = make(first_host, slow, admission=admission, watchdog_ms=100_000)
    second = make(second_host, FakeBackend(latency_ms=10), admission=admission, watchdog_ms=80)

    first_host.revision += 1
    first.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await asyncio.sleep(0.05)
    second_host.revision += 1
    second.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])

    await settle(600)
    assert second_host.results(), "the queued room timed out before it started"
    assert "error" not in second_host.states()
    first.stop()
    second.stop()


class ThreadBackend:
    """A backend whose work is on a thread, so cancelling the caller does not
    stop it - the in-process pipeline's situation."""

    name = "thread"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.running = False
        self.finished = False
        self.started = asyncio.Event()

    async def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(["fast", "quality"], 2048, 0.95)

    def _work(self) -> bytes:
        import time as _time

        self.running = True
        try:
            _time.sleep(self.seconds)
            return b"patch"
        finally:
            self.running = False
            self.finished = True

    async def generate(self, req: GenerateRequest) -> bytes:
        self.started.set()
        future = asyncio.ensure_future(asyncio.to_thread(self._work))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # What InprocBackend does: wait for the thread before letting go.
            from contextlib import suppress

            with suppress(BaseException):
                await asyncio.shield(future)
            raise


async def test_a_timed_out_run_holds_admission_until_its_thread_finishes() -> None:
    """Releasing the slot early hands the next room a device that is still
    busy - and it would rasterise and start its own watchdog while queued."""
    admission = GenerationAdmission()
    slow = ThreadBackend(seconds=0.4)
    host = FakeHost()
    scheduler = make(host, slow, admission=admission, watchdog_ms=60, error_backoff_ms=10_000)

    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])
    await slow.started.wait()
    await asyncio.sleep(0.15)  # past the watchdog, inside the thread's sleep

    # The watchdog has fired, but the thread has not finished, so the slot is
    # still taken.
    assert slow.running is True
    admitted = asyncio.ensure_future(_take(admission))
    await asyncio.sleep(0.05)
    assert not admitted.done(), "admission was released while the thread ran"

    await asyncio.sleep(0.4)
    assert slow.finished is True
    await asyncio.wait_for(admitted, 1)
    scheduler.stop()


async def _take(admission: GenerationAdmission) -> None:
    async with admission.admit():
        return None


async def test_the_watchdog_covers_the_render_not_only_the_backend() -> None:
    """A render that never returns used to hold admission unwatched."""
    host, backend = FakeHost(), FakeBackend()
    stuck = threading.Event()

    def render(crop, size):
        host.renders += 1
        stuck.wait(2.0)
        return b"input"

    def begin_job() -> RenderJob:
        return RenderJob(
            revision=host.revision,
            prompt="a town",
            denoise=0.7,
            negative_prompt="",
            resolution=512,
            profile="fast",
            render=render,
        )

    host.begin_job = begin_job  # type: ignore[assignment]
    scheduler = make(host, backend, watchdog_ms=50, error_backoff_ms=10_000)
    scheduler.mark_dirty([{"x": 0, "y": 0, "width": 1, "height": 1}])

    await settle(200)
    # The watchdog has fired, but the render thread is still going, so nothing
    # is reported and nothing is released until it stops.
    assert host.states()[-1] == "generating"
    assert backend.requests == [], "a timed-out render still reached the backend"

    stuck.set()
    await settle(200)
    assert host.states()[-1] == "error"
    assert "timed out" in (host.messages[-1].get("message") or "")
    scheduler.stop()
