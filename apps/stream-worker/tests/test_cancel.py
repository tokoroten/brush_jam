"""Cancellation, busy/409 and request_id contract. No GPU required.

These exercise the reason cancellation exists at all: aborting the HTTP request
cannot stop GPU work, because the pipeline call is already running in a worker
thread. Without cooperative cancellation an abandoned request keeps the GPU and
the next one queues behind it.
"""

from __future__ import annotations

import asyncio
import base64
import io

import httpx
import pytest
from PIL import Image

from brushjam.ai.pipeline import GenerateResult, GenerationCancelled

from stream_worker.app import create_app
from stream_worker.config import Settings


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


REQUEST = {
    "image_b64": png_b64(Image.new("RGB", (256, 256), "white")),
    "prompt": "anime style, fantasy town",
    "size": 256,
}


class SlowPipeline:
    """Runs `steps` fake steps, polling should_cancel between them like the
    real `callback_on_step_end` does."""

    backend = "slow-fake"

    def __init__(self, steps: int = 20, step_seconds: float = 0.01) -> None:
        self.warm = True
        self.steps = steps
        self.step_seconds = step_seconds
        self.started = asyncio.Event()
        self.steps_run = 0
        self.loop: asyncio.AbstractEventLoop | None = None
        self.out_size: tuple[int, int] | None = None

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def model_name(self) -> str:
        return "slow-fake"

    def memory(self) -> dict:
        return {}

    def generate(self, *, width, height, should_cancel=None, request_id=None, **_kwargs):
        import time

        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.started.set)
        self.steps_run = 0
        for _ in range(self.steps):
            if should_cancel is not None and should_cancel():
                raise GenerationCancelled(request_id)
            self.steps_run += 1
            time.sleep(self.step_seconds)
        size = self.out_size or (width, height)
        return GenerateResult(image=Image.new("RGB", size, (1, 2, 3)), timings={"diffusion_ms": 1.0})


async def client_for(pipe) -> httpx.AsyncClient:
    app = create_app(Settings(), pipeline=pipe)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://worker")


@pytest.mark.anyio
async def test_generate_echoes_the_request_id():
    pipe = SlowPipeline(steps=1, step_seconds=0)
    async with await client_for(pipe) as client:
        res = await client.post("/generate", json={**REQUEST, "request_id": "abc-123"})
        assert res.status_code == 200
        assert res.json()["request_id"] == "abc-123"


@pytest.mark.anyio
async def test_generate_invents_a_request_id_when_absent():
    pipe = SlowPipeline(steps=1, step_seconds=0)
    async with await client_for(pipe) as client:
        res = await client.post("/generate", json=REQUEST)
        assert res.status_code == 200
        assert len(res.json()["request_id"]) >= 8


@pytest.mark.anyio
async def test_second_request_is_refused_with_409_while_busy():
    pipe = SlowPipeline(steps=40, step_seconds=0.01)
    pipe.loop = asyncio.get_running_loop()
    async with await client_for(pipe) as client:
        first = asyncio.create_task(client.post("/generate", json={**REQUEST, "request_id": "one"}))
        await asyncio.wait_for(pipe.started.wait(), timeout=5)

        busy = await client.post("/generate", json={**REQUEST, "request_id": "two"})
        assert busy.status_code == 409
        assert "one" in busy.json()["detail"]
        assert busy.headers.get("retry-after") == "1"

        assert (await first).status_code == 200


@pytest.mark.anyio
async def test_queue_true_waits_instead_of_409():
    pipe = SlowPipeline(steps=10, step_seconds=0.01)
    pipe.loop = asyncio.get_running_loop()
    async with await client_for(pipe) as client:
        first = asyncio.create_task(client.post("/generate", json={**REQUEST, "request_id": "one"}))
        await asyncio.wait_for(pipe.started.wait(), timeout=5)
        second = await client.post("/generate", json={**REQUEST, "request_id": "two", "queue": True})
        assert second.status_code == 200
        assert second.json()["timings"]["wait_ms"] > 0
        assert (await first).status_code == 200


@pytest.mark.anyio
async def test_cancel_stops_a_running_job_before_it_finishes():
    pipe = SlowPipeline(steps=500, step_seconds=0.01)
    pipe.loop = asyncio.get_running_loop()
    async with await client_for(pipe) as client:
        running = asyncio.create_task(client.post("/generate", json={**REQUEST, "request_id": "kill-me"}))
        await asyncio.wait_for(pipe.started.wait(), timeout=5)

        cancelled = await client.post("/cancel", json={"request_id": "kill-me"})
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "running"

        res = await asyncio.wait_for(running, timeout=10)
        assert res.status_code == 499
        # The point of cooperative cancellation: it stopped early.
        assert pipe.steps_run < 500


@pytest.mark.anyio
async def test_cancel_before_the_job_starts_means_it_never_runs():
    pipe = SlowPipeline(steps=30, step_seconds=0.01)
    pipe.loop = asyncio.get_running_loop()
    async with await client_for(pipe) as client:
        first = asyncio.create_task(client.post("/generate", json={**REQUEST, "request_id": "one"}))
        await asyncio.wait_for(pipe.started.wait(), timeout=5)

        queued = asyncio.create_task(
            client.post("/generate", json={**REQUEST, "request_id": "two", "queue": True})
        )
        await asyncio.sleep(0.02)
        assert (await client.post("/cancel", json={"request_id": "two"})).json()["state"] == "pending"

        assert (await first).status_code == 200
        assert (await asyncio.wait_for(queued, timeout=10)).status_code == 499


@pytest.mark.anyio
async def test_healthz_reports_busy_and_current_request_id():
    pipe = SlowPipeline(steps=40, step_seconds=0.01)
    pipe.loop = asyncio.get_running_loop()
    async with await client_for(pipe) as client:
        idle = (await client.get("/healthz")).json()
        assert idle["busy"] is False and idle["current_request_id"] is None

        running = asyncio.create_task(client.post("/generate", json={**REQUEST, "request_id": "live"}))
        await asyncio.wait_for(pipe.started.wait(), timeout=5)
        busy = (await client.get("/healthz")).json()
        assert busy["busy"] is True
        assert busy["current_request_id"] == "live"

        await running
        after = (await client.get("/healthz")).json()
        assert after["busy"] is False and after["current_request_id"] is None


@pytest.mark.anyio
async def test_cancelling_an_unknown_id_is_not_an_error():
    """The server may cancel a request the worker already finished."""
    pipe = SlowPipeline(steps=1, step_seconds=0)
    async with await client_for(pipe) as client:
        res = await client.post("/cancel", json={"request_id": "never-existed"})
        assert res.status_code == 200
        assert res.json() == {"ok": True, "request_id": "never-existed", "state": "pending"}


@pytest.mark.anyio
async def test_a_wrongly_sized_result_is_a_500_not_a_bad_png():
    """The contract promises exactly width x height; returning anything else
    would be composited at the wrong scale by the server."""
    pipe = SlowPipeline(steps=1, step_seconds=0)
    pipe.out_size = (128, 128)
    async with await client_for(pipe) as client:
        res = await client.post("/generate", json=REQUEST)
        assert res.status_code == 500
        assert "expected (256, 256)" in res.json()["detail"]


@pytest.mark.anyio
async def test_timings_carry_the_phase_breakdown():
    pipe = SlowPipeline(steps=1, step_seconds=0)
    async with await client_for(pipe) as client:
        timings = (await client.post("/generate", json=REQUEST)).json()["timings"]
        for key in ("wait_ms", "png_encode_ms", "total_ms", "diffusion_ms"):
            assert key in timings, key
