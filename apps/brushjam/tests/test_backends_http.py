"""The three HTTP backends, driven through an httpx transport instead of a
network: workflow shape, capability semantics, decoding and cleanup.

Mirrors the retired Node server's test/{comfyui,comfyui-cleanup,stream,runpod}.test.ts.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest
from PIL import Image

from brushjam.ai.backends.base import BackendHttpError, GenerateRequest
from brushjam.ai.backends.comfyui import (
    DEFAULT_FAST_LORA,
    ComfyUIBackend,
    build_workflow,
    fast_profile,
    negative_prompt_active,
)
from brushjam.ai.backends.runpod import RunpodBackend, decode_image
from brushjam.ai.backends.stream import (
    DEFAULT_STREAM_MAX_DENOISE,
    DEFAULT_STREAM_MAX_RESOLUTION,
    StreamBackend,
    png_size,
)
from brushjam.constants import QUALITY_SUFFIX
from brushjam.raster import to_png


def a_png(size: int = 64) -> bytes:
    return to_png(Image.new("RGB", (size, size), (10, 20, 30)))


def a_request(size: int = 64, profile: str = "fast", **kwargs) -> GenerateRequest:
    defaults: Dict[str, Any] = dict(
        profile=profile,
        prompt="a town",
        negative_prompt="blurry",
        image_png=a_png(size),
        mask_png=a_png(size),
        size=size,
        denoise=0.7,
        steps=4,
        seed=42,
        tag="room_r1",
    )
    defaults.update(kwargs)
    return GenerateRequest(**defaults)


class Recorder:
    """An httpx transport that answers from a routing table and records calls."""

    def __init__(self, routes: Callable[[httpx.Request], httpx.Response]) -> None:
        self.routes = routes
        self.calls: List[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            return self.routes(request)

        return httpx.MockTransport(handler)

    def paths(self) -> List[str]:
        return [c.url.path for c in self.calls]

    def tails(self) -> List[str]:
        """Paths with the RunPod endpoint prefix removed."""
        return [c.url.path.split("/ep1", 1)[-1] for c in self.calls]

    def body(self, index: int) -> Any:
        return json.loads(self.calls[index].content)


# ------------------------------------------------------------------ workflow


def test_the_quality_workflow_has_no_lora_node() -> None:
    workflow = build_workflow(
        checkpoint="ckpt.safetensors",
        prompt="a town",
        negative_prompt="blurry",
        image_name="img.png",
        mask_name="mask.png",
        seed=1,
        steps=14,
        cfg=5.5,
        denoise=0.7,
        filename_prefix="brushjam/x",
    )
    assert "12" not in workflow
    sampler = workflow["9"]["inputs"]
    assert sampler["model"] == ["1", 0]
    assert (sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (5.5, "euler_ancestral", "normal")
    assert sampler["steps"] == 14 and sampler["denoise"] == 0.7
    assert workflow["2"]["inputs"]["text"] == "a town" + QUALITY_SUFFIX
    assert workflow["8"]["class_type"] == "SetLatentNoiseMask"
    assert workflow["10"]["class_type"] == "VAEDecode"


def test_the_fast_workflow_inserts_the_lora_between_checkpoint_and_consumers() -> None:
    workflow = build_workflow(
        checkpoint="ckpt.safetensors",
        prompt="a town",
        negative_prompt="",
        image_name="img.png",
        mask_name="mask.png",
        seed=1,
        steps=4,
        cfg=5.5,
        denoise=0.8,
        filename_prefix="brushjam/x",
        fast_lora=DEFAULT_FAST_LORA,
        vae_tile=512,
    )
    assert workflow["12"]["class_type"] == "LoraLoader"
    assert workflow["12"]["inputs"]["lora_name"] == DEFAULT_FAST_LORA
    # Both text encoders and the sampler read from the LoRA; the VAE does not.
    assert workflow["2"]["inputs"]["clip"] == ["12", 1]
    assert workflow["9"]["inputs"]["model"] == ["12", 0]
    assert workflow["7"]["inputs"]["vae"] == ["1", 2]
    # DMD2 is guidance-distilled: cfg 1.0, not the LCM 1.5.
    assert workflow["9"]["inputs"]["cfg"] == 1.0
    assert workflow["9"]["inputs"]["sampler_name"] == "lcm"
    assert workflow["10"]["class_type"] == "VAEDecodeTiled"
    assert workflow["10"]["inputs"]["tile_size"] == 512


def test_the_lora_name_picks_the_profile() -> None:
    assert fast_profile("dmd2_sdxl_4step_lora_fp16.safetensors").cfg == 1.0
    assert fast_profile("lcm-lora-sdxl.safetensors").cfg == 1.5
    assert fast_profile("") is None
    assert negative_prompt_active(fast_profile(DEFAULT_FAST_LORA), 5.5) is False
    assert negative_prompt_active(None, 5.5) is True
    assert negative_prompt_active(None, 1.0) is False


async def test_comfyui_capabilities_follow_the_lora() -> None:
    with_lora = ComfyUIBackend("http://c", "ckpt", fast_lora=DEFAULT_FAST_LORA)
    caps = await with_lora.capabilities()
    assert caps.profiles == ["fast", "quality"]
    assert caps.negative_prompt_active == {"fast": False, "quality": True}
    without = await ComfyUIBackend("http://c", "ckpt", fast_lora="").capabilities()
    assert without.profiles == ["quality"]


# ------------------------------------------------------------------- comfyui


def _comfy_routes(history: Dict[str, Any], queue: Optional[Dict[str, Any]] = None):
    def routes(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/upload/image":
            return httpx.Response(200, json={"name": "up.png", "subfolder": "", "type": "input"})
        if path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if path.startswith("/history/"):
            return httpx.Response(200, json=history)
        if path == "/view":
            return httpx.Response(200, content=a_png(), headers={"content-type": "image/png"})
        if path == "/queue":
            return httpx.Response(200, json=queue or {})
        if path == "/interrupt":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    return routes


async def test_comfyui_uploads_queues_and_fetches_the_result() -> None:
    history = {"p1": {"status": {"status_str": "success"}, "outputs": {"11": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}}
    rec = Recorder(_comfy_routes(history))
    backend = ComfyUIBackend("http://c", "ckpt", fast_lora=DEFAULT_FAST_LORA, transport=rec.transport())
    out = await backend.generate(a_request())
    assert png_size(out) == (64, 64)
    assert rec.paths()[:3] == ["/upload/image", "/upload/image", "/prompt"]
    workflow = rec.body(2)["prompt"]
    assert workflow["12"]["inputs"]["lora_name"] == DEFAULT_FAST_LORA


async def test_comfyui_interrupts_a_running_job_when_the_generation_fails() -> None:
    history = {"p1": {"status": {"status_str": "error"}, "outputs": {}}}
    rec = Recorder(_comfy_routes(history, queue={"queue_running": [[0, "p1"]], "queue_pending": []}))
    backend = ComfyUIBackend("http://c", "ckpt", transport=rec.transport(), poll_interval_ms=1)
    with pytest.raises(RuntimeError, match="execution error"):
        await backend.generate(a_request(profile="quality"))
    # It must interrupt *its own* job rather than leaving it on the GPU.
    assert "/interrupt" in rec.paths()


async def test_comfyui_deletes_a_queued_job_rather_than_interrupting_someone_elses() -> None:
    history = {"p1": {"status": {"status_str": "error"}, "outputs": {}}}
    rec = Recorder(_comfy_routes(history, queue={"queue_running": [[0, "other"]], "queue_pending": [[1, "p1"]]}))
    backend = ComfyUIBackend("http://c", "ckpt", transport=rec.transport(), poll_interval_ms=1)
    with pytest.raises(RuntimeError):
        await backend.generate(a_request(profile="quality"))
    assert "/interrupt" not in rec.paths()
    delete = [c for c in rec.calls if c.url.path == "/queue" and c.method == "POST"]
    assert json.loads(delete[0].content) == {"delete": ["p1"]}


async def test_comfyui_gives_up_at_its_deadline() -> None:
    rec = Recorder(_comfy_routes({}))
    backend = ComfyUIBackend(
        "http://c", "ckpt", transport=rec.transport(), poll_interval_ms=1, timeout_ms=30
    )
    with pytest.raises(RuntimeError, match="timed out"):
        await backend.generate(a_request(profile="quality"))


# -------------------------------------------------------------------- stream


def _stream_health_body(**overrides) -> Dict[str, Any]:
    body = {
        "ok": True,
        "warm": True,
        "max_size": 1024,
        "max_denoise": 0.9,
        "negative_prompt_active": False,
        "busy": False,
        "backend": "diffusers-sdxl-lcm",
        "steps": 4,
        "guidance": 1.0,
        "vae": "fp16fix",
        "model": "wai+dmd2",
        "lora": "dmd2",
    }
    body.update(overrides)
    return body


async def test_stream_health_and_capabilities() -> None:
    rec = Recorder(lambda r: httpx.Response(200, json=_stream_health_body()))
    backend = StreamBackend("http://w", transport=rec.transport())
    health = await backend.health()
    assert (health.ok, health.warm, health.max_size, health.busy) == (True, True, 1024, False)
    assert health.sampling["lora"] == "dmd2"
    caps = await backend.capabilities()
    # One fused few-step LoRA: fast only, and the worker's own ceilings.
    assert caps.profiles == ["fast"]
    assert (caps.max_resolution, caps.max_denoise) == (1024, 0.9)
    assert caps.negative_prompt_active == {"fast": False, "quality": False}


async def test_stream_falls_back_to_conservative_limits_when_the_worker_is_silent() -> None:
    rec = Recorder(lambda r: httpx.Response(200, json={"ok": True}))
    caps = await StreamBackend("http://w", transport=rec.transport()).capabilities()
    assert caps.max_resolution == DEFAULT_STREAM_MAX_RESOLUTION
    assert caps.max_denoise == DEFAULT_STREAM_MAX_DENOISE
    assert caps.negative_prompt_active == {"fast": True, "quality": True}


@pytest.mark.parametrize(
    "response,reason",
    [
        (httpx.Response(500), "/healthz answered 500"),
        (httpx.Response(200, json={"ok": False}), "/healthz reported not ok"),
    ],
)
async def test_an_unusable_worker_is_reported_not_guessed(response, reason: str) -> None:
    rec = Recorder(lambda r: response)
    health = await StreamBackend("http://w", transport=rec.transport()).health()
    assert not health.ok and health.reason == reason


async def test_stream_generate_strips_the_quality_suffix_and_validates_the_png() -> None:
    payload = {"image_b64": base64.b64encode(a_png(64)).decode()}
    rec = Recorder(lambda r: httpx.Response(200, json=payload))
    backend = StreamBackend("http://w", transport=rec.transport())
    out = await backend.generate(a_request(prompt="a town" + QUALITY_SUFFIX))
    assert png_size(out) == (64, 64)
    body = rec.body(0)
    # The worker appends the suffix itself; sending it twice doubles it.
    assert body["prompt"] == "a town"
    assert body["queue"] is True
    assert (body["width"], body["height"]) == (64, 64)


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({}, "returned no image"),
        ({"image_b64": "!!!not base64!!!"}, "malformed base64"),
        ({"image_b64": base64.b64encode(b"hello").decode()}, "not a PNG"),
    ],
)
async def test_a_bad_stream_response_is_named_not_passed_on(payload, fragment: str) -> None:
    rec = Recorder(lambda r: httpx.Response(200, json=payload))
    backend = StreamBackend("http://w", transport=rec.transport())
    with pytest.raises(RuntimeError, match=fragment):
        await backend.generate(a_request())


async def test_a_wrong_sized_stream_result_is_refused() -> None:
    rec = Recorder(lambda r: httpx.Response(200, json={"image_b64": base64.b64encode(a_png(32)).decode()}))
    with pytest.raises(RuntimeError, match="returned 32x32, expected 64x64"):
        await StreamBackend("http://w", transport=rec.transport()).generate(a_request())


async def test_a_busy_worker_is_a_backend_http_error() -> None:
    rec = Recorder(lambda r: httpx.Response(409, json={"detail": "busy with abc"}))
    with pytest.raises(BackendHttpError) as err:
        await StreamBackend("http://w", transport=rec.transport()).generate(a_request())
    assert err.value.status == 409
    assert "busy with abc" in str(err.value)


async def test_a_499_is_the_worker_acknowledging_our_own_cancellation() -> None:
    rec = Recorder(lambda r: httpx.Response(499, json={"detail": "cancelled"}))
    with pytest.raises(asyncio.CancelledError):
        await StreamBackend("http://w", transport=rec.transport()).generate(a_request())


# -------------------------------------------------------------------- runpod


async def test_runpod_posts_the_same_workflow_and_decodes_the_result() -> None:
    job = {"status": "COMPLETED", "output": {"images": [{"type": "base64", "data": base64.b64encode(a_png()).decode()}]}}
    rec = Recorder(lambda r: httpx.Response(200, json=job))
    backend = RunpodBackend("ep1", "key", "ckpt", fast_lora=DEFAULT_FAST_LORA, transport=rec.transport())
    out = await backend.generate(a_request())
    assert png_size(out) == (64, 64)
    body = rec.body(0)
    assert body["input"]["workflow"]["12"]["inputs"]["lora_name"] == DEFAULT_FAST_LORA
    assert [i["name"] for i in body["input"]["images"]] == [
        body["input"]["workflow"]["4"]["inputs"]["image"],
        body["input"]["workflow"]["5"]["inputs"]["image"],
    ]
    assert rec.calls[0].headers["authorization"] == "Bearer key"


async def test_runpod_polls_when_runsync_hands_back_a_job_id() -> None:
    states = iter(
        [
            {"id": "j1", "status": "IN_PROGRESS"},
            {"id": "j1", "status": "IN_QUEUE"},
            {
                "id": "j1",
                "status": "COMPLETED",
                "output": {"images": [{"type": "base64", "data": base64.b64encode(a_png()).decode()}]},
            },
        ]
    )
    rec = Recorder(lambda r: httpx.Response(200, json=next(states)))
    backend = RunpodBackend("ep1", "key", "ckpt", transport=rec.transport(), poll_interval_ms=1)
    out = await backend.generate(a_request(profile="quality"))
    assert png_size(out) == (64, 64)
    assert rec.tails() == ["/runsync", "/status/j1", "/status/j1"]


async def test_runpod_cancels_a_job_it_gives_up_on() -> None:
    def routes(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/runsync"):
            return httpx.Response(200, json={"id": "j1", "status": "IN_PROGRESS"})
        if "/status/" in request.url.path:
            return httpx.Response(200, json={"id": "j1", "status": "IN_PROGRESS"})
        return httpx.Response(200, json={})

    rec = Recorder(routes)
    backend = RunpodBackend(
        "ep1", "key", "ckpt", transport=rec.transport(), poll_interval_ms=1, timeout_ms=30
    )
    with pytest.raises(RuntimeError, match="timed out"):
        await backend.generate(a_request(profile="quality"))
    assert "/cancel/j1" in rec.tails()


@pytest.mark.parametrize(
    "job,fragment",
    [
        ({"status": "FAILED", "error": "boom"}, "RunPod job FAILED: boom"),
        ({"status": "COMPLETED", "output": {"images": []}}, "returned no image"),
        (
            {"status": "COMPLETED", "output": {"images": [{"type": "s3_url"}]}},
            "expects base64 images",
        ),
        (
            {"status": "COMPLETED", "output": {"images": [{"type": "base64"}]}},
            "no data",
        ),
    ],
)
def test_runpod_decode_errors(job, fragment: str) -> None:
    with pytest.raises(RuntimeError, match=fragment):
        decode_image(job)
