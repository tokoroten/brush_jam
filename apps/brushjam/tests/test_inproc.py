"""The in-process pipeline and its backend, with no GPU anywhere.

The schedule maths and the profile switch are pure, so they are tested against
the real objects; everything that would need CUDA runs against the dry-run
pipeline, which honours the same contract (including cancellation).
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, Dict, List

import pytest
from PIL import Image

from brushjam.ai.backends.base import BackendHttpError, GenerateRequest
from brushjam.ai.backends.inproc import InprocBackend
from brushjam.ai.pipeline import (
    DryRunPipeline,
    GenerationCancelled,
    InprocPipeline,
    PipelineSettings,
    composite_through_mask,
    lcm_timesteps_for_strength,
    round_size,
    steps_for_strength,
)
from brushjam.ai.backends.stream import png_size
from brushjam.raster import to_png


def a_png(size: int = 256, colour=(10, 20, 30)) -> bytes:
    return to_png(Image.new("RGB", (size, size), colour))


def a_request(**kwargs) -> GenerateRequest:
    defaults: Dict[str, Any] = dict(
        profile="fast",
        prompt="a town",
        negative_prompt="blurry",
        image_png=a_png(),
        mask_png=to_png(Image.new("L", (256, 256), 255).convert("RGB")),
        size=256,
        denoise=0.7,
        steps=4,
        seed=1,
        tag="room_r1",
    )
    defaults.update(kwargs)
    return GenerateRequest(**defaults)


def settings(**kwargs) -> PipelineSettings:
    s = PipelineSettings()
    s.max_size = 1024
    s.max_denoise = 0.9
    for key, value in kwargs.items():
        setattr(s, key, value)
    return s


# ------------------------------------------------------------------ schedules


def test_the_lcm_schedule_keeps_nearby_strengths_distinct() -> None:
    """The bug this replaced: two integer roundings made 0.8 and 0.9 produce
    byte-identical images at 4 steps."""
    at = {s: lcm_timesteps_for_strength(4, s) for s in (0.5, 0.65, 0.8, 0.9)}
    assert [v[0] for v in at.values()] == [499, 639, 799, 899]
    assert len({tuple(v) for v in at.values()}) == 4
    for schedule in at.values():
        assert len(schedule) == 4
        assert schedule == sorted(schedule, reverse=True)


def test_the_lcm_schedule_cannot_run_more_steps_than_exist_below_the_start() -> None:
    assert lcm_timesteps_for_strength(4, 0.02) == [19]
    assert len(lcm_timesteps_for_strength(8, 0.06)) == 3


def test_quality_steps_survive_the_strength_slice() -> None:
    # diffusers keeps int(n * strength) timesteps, so ask for enough that the
    # requested count survives - the ComfyUI semantic.
    for steps, strength in ((14, 0.7), (14, 0.5), (20, 0.8)):
        scheduled = steps_for_strength(steps, strength)
        assert int(scheduled * strength) >= steps - 1


@pytest.mark.parametrize("value,expected", [(768, 768), (1023, 1016), (256, 256)])
def test_round_size(value: int, expected: int) -> None:
    assert round_size(value, 1024) == expected


@pytest.mark.parametrize("value", [64, 2048])
def test_round_size_refuses_what_the_model_cannot_do(value: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        round_size(value, 1024)


def test_the_mask_decides_what_the_result_replaces() -> None:
    base = Image.new("RGB", (4, 4), (0, 0, 0))
    generated = Image.new("RGB", (4, 4), (255, 255, 255))
    assert composite_through_mask(base, generated, None).getpixel((0, 0)) == (255, 255, 255)
    black = Image.new("L", (4, 4), 0)
    assert composite_through_mask(base, generated, black).getpixel((0, 0)) == (0, 0, 0)
    half = Image.new("L", (4, 4), 128)
    assert composite_through_mask(base, generated, half).getpixel((0, 0))[0] == 128


# ------------------------------------------------------------- profile switch


class FakePipe:
    """Just enough of a diffusers pipeline to record the profile switch."""

    def __init__(self) -> None:
        self.scheduler: Any = "initial"
        self.calls: List[str] = []
        self.adapters: List[str] = []

    def enable_lora(self) -> None:
        self.calls.append("enable_lora")

    def disable_lora(self) -> None:
        self.calls.append("disable_lora")

    def set_adapters(self, names, adapter_weights=None) -> None:
        self.calls.append(f"set_adapters:{','.join(names)}")
        self.adapters = list(names)


def a_pipeline() -> InprocPipeline:
    pipeline = InprocPipeline(settings())
    pipeline.pipe = FakePipe()
    pipeline._schedulers = {"fast": "LCMScheduler", "quality": "EulerAncestralDiscreteScheduler"}
    pipeline._has_lora = True
    return pipeline


def test_switching_profile_attaches_and_detaches_the_lora() -> None:
    pipeline = a_pipeline()
    pipe = pipeline.pipe

    assert pipeline._select_profile("fast") == 1.0  # DMD2 is guidance-distilled
    assert pipe.scheduler == "LCMScheduler"
    assert pipe.calls == ["enable_lora", "set_adapters:fast"]

    assert pipeline._select_profile("quality") == 5.5
    assert pipe.scheduler == "EulerAncestralDiscreteScheduler"
    assert pipe.calls[-1] == "disable_lora"

    # Idempotent: staying on a profile costs nothing.
    before = list(pipe.calls)
    pipeline._select_profile("quality")
    assert pipe.calls == before


def test_the_lcm_lora_wants_a_little_guidance_and_dmd2_wants_none() -> None:
    assert settings(lora="dmd2").guidance_for("fast") == 1.0
    assert settings(lora="lcm").guidance_for("fast") == 1.5
    assert settings().guidance_for("quality") == 5.5
    # ...which is exactly what decides whether the negative prompt does anything.
    assert settings(lora="dmd2").negative_prompt_active() == {"fast": False, "quality": True}
    assert settings(lora="lcm").negative_prompt_active() == {"fast": True, "quality": True}


def test_steps_follow_the_profile() -> None:
    s = settings()
    assert (s.steps_for("fast"), s.steps_for("quality")) == (4, 14)


# ------------------------------------------------------------------- backend


async def test_the_dry_run_backend_runs_the_whole_contract() -> None:
    backend = InprocBackend(settings(), dry_run=True)
    await backend.load()
    caps = await backend.capabilities()
    # One resident model, both profiles: quality is the LoRA detached.
    assert caps.profiles == ["fast", "quality"]
    assert (caps.max_resolution, caps.max_denoise) == (1024, 0.9)
    out = await backend.generate(a_request())
    assert png_size(out) == (256, 256)
    call = backend.pipeline.calls[-1]
    assert call["profile"] == "fast" and call["guidance"] == 1.0
    assert call["has_mask"] is True


async def test_the_profile_travels_with_the_request() -> None:
    backend = InprocBackend(settings(), dry_run=True)
    await backend.generate(a_request(profile="fast", steps=4))
    await backend.generate(a_request(profile="quality", steps=14))
    fast, quality = backend.pipeline.calls[-2:]
    assert (fast["profile"], fast["steps"], fast["guidance"]) == ("fast", 4, 1.0)
    assert (quality["profile"], quality["steps"], quality["guidance"]) == ("quality", 14, 5.5)


async def test_denoise_is_clamped_to_what_the_model_will_honour() -> None:
    backend = InprocBackend(settings(max_denoise=0.9), dry_run=True)
    await backend.generate(a_request(denoise=0.99))
    assert backend.pipeline.calls[-1]["strength"] == 0.9


async def test_a_size_the_model_cannot_do_is_a_refusal_not_a_failure() -> None:
    backend = InprocBackend(settings(max_size=768), dry_run=True)
    with pytest.raises(BackendHttpError) as err:
        await backend.generate(a_request(size=1024))
    # 4xx, so the scheduler stops retrying instead of failing forever.
    assert err.value.status == 400


async def test_a_cancelled_generation_stops_at_the_next_step() -> None:
    backend = InprocBackend(settings(), pipeline=DryRunPipeline(settings(), latency_ms=2000))
    await backend.load()
    task = asyncio.ensure_future(backend.generate(a_request()))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The pipeline thread notices the flag and unwinds, so the next request is
    # not queued behind a run nobody is waiting for.
    for _ in range(200):
        if not backend._busy:
            break
        await asyncio.sleep(0.01)
    assert backend._busy is False


async def test_generations_are_serialised_onto_one_thread() -> None:
    backend = InprocBackend(settings(), pipeline=DryRunPipeline(settings(), latency_ms=60))
    await backend.load()
    started = []

    async def run(tag: str) -> None:
        started.append(tag)
        await backend.generate(a_request(tag=tag))

    await asyncio.gather(run("a"), run("b"), run("c"))
    # Three results, and the busy flag never overlapped: one GPU, one at a time.
    assert len(backend.pipeline.calls) == 3
    assert backend._busy is False


def test_status_publishes_what_the_tooling_reads() -> None:
    backend = InprocBackend(settings(), dry_run=True)
    status = backend.status()
    for key in ("model", "max_size", "max_denoise", "steps", "guidance", "lora", "vae", "warm"):
        assert key in status, key
    assert status["max_size"] == 1024
    assert status["lora"] == "dmd2"
