"""The in-process pipeline and its backend, with no GPU anywhere.

The schedule maths and the profile switch are pure, so they are tested against
the real objects; everything that would need CUDA runs against the dry-run
pipeline, which honours the same contract (including cancellation).
"""

from __future__ import annotations

import asyncio
import threading
import io
from typing import Any, Dict, List

import pytest
from PIL import Image

from brushjam.ai.backends.base import BackendHttpError, GenerateRequest
from brushjam.ai.backends.inproc import InprocBackend
from brushjam.ai.pipeline import (
    FP8_BELOW_BYTES,
    TILE_BELOW_BYTES,
    DryRunPipeline,
    GenerationCancelled,
    InprocPipeline,
    PipelineSettings,
    SMALL_CARD_VAE_TILE,
    choose_unet_storage,
    choose_vae_tile,
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
        self.fused = False

    def enable_lora(self) -> None:
        self.calls.append("enable_lora")

    def disable_lora(self) -> None:
        self.calls.append("disable_lora")

    def set_adapters(self, names, adapter_weights=None) -> None:
        self.calls.append(f"set_adapters:{','.join(names)}")
        self.adapters = list(names)

    def fuse_lora(self, components=None, adapter_names=None, lora_scale=1.0) -> None:
        self.calls.append(f"fuse_lora:{','.join(components or [])}:{lora_scale}")
        self.fused = True

    def unfuse_lora(self, components=None) -> None:
        self.calls.append(f"unfuse_lora:{','.join(components or [])}")
        self.fused = False


class NoFusePipe(FakePipe):
    """A build whose fuse_lora refuses - the fallback must still be correct."""

    def fuse_lora(self, components=None, adapter_names=None, lora_scale=1.0) -> None:
        raise RuntimeError("this build cannot fuse")


def a_pipeline(pipe: Any = None) -> InprocPipeline:
    pipeline = InprocPipeline(settings())
    pipeline.pipe = pipe or FakePipe()
    pipeline._schedulers = {"fast": "LCMScheduler", "quality": "EulerAncestralDiscreteScheduler"}
    pipeline._has_lora = True
    return pipeline


def test_switching_profile_fuses_and_unfuses_the_lora() -> None:
    pipeline = a_pipeline()
    pipe = pipeline.pipe

    assert pipeline._select_profile("fast") == 1.0  # DMD2 is guidance-distilled
    assert pipe.scheduler == "LCMScheduler"
    # Enable BEFORE fusing: PEFT unmerges a merged layer the moment adapters
    # are disabled, so the two must not be reordered.
    assert pipe.calls == ["enable_lora", "set_adapters:fast", "fuse_lora:unet:1.0"]
    assert pipe.fused is True and pipeline._fused is True

    assert pipeline._select_profile("quality") == 5.5
    assert pipe.scheduler == "EulerAncestralDiscreteScheduler"
    # ...and unfuse BEFORE disabling, for the same reason.
    assert pipe.calls[-2:] == ["unfuse_lora:unet", "disable_lora"]
    assert pipe.fused is False and pipeline._fused is False

    # Idempotent: staying on a profile costs nothing.
    before = list(pipe.calls)
    pipeline._select_profile("quality")
    assert pipe.calls == before


def test_the_adapter_is_never_fused_twice() -> None:
    pipeline = a_pipeline()
    pipe = pipeline.pipe
    pipeline._select_profile("fast")
    pipeline._select_profile("quality")
    pipeline._select_profile("fast")
    assert pipe.calls.count("fuse_lora:unet:1.0") == 2
    assert pipe.calls.count("unfuse_lora:unet") == 1
    # The adapter is never reloaded: switching is a weight operation on weights
    # that are already resident.
    assert pipe.adapters == ["fast"]


def test_a_build_that_cannot_fuse_falls_back_to_the_attached_adapter() -> None:
    pipeline = a_pipeline(NoFusePipe())
    pipe = pipeline.pipe

    pipeline._select_profile("fast")
    assert pipeline._fuse_unavailable is True
    assert pipeline._fused is False
    assert pipe.calls[-1] == "set_adapters:fast"

    pipeline._select_profile("quality")
    # No unfuse: nothing was ever merged, and unfusing would be a lie.
    assert "unfuse_lora:unet" not in pipe.calls
    assert pipe.calls[-1] == "disable_lora"


def test_the_switch_cost_is_recorded_and_zero_when_nothing_changes() -> None:
    pipeline = a_pipeline()
    pipeline._select_profile("fast")
    assert pipeline._last_switch_ms >= 0.0
    pipeline._last_switch_ms = 12.0
    pipeline._select_profile("fast")
    assert pipeline._last_switch_ms == 0.0


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


def test_unfusing_restores_the_base_weights_in_fp16() -> None:
    """The whole fuse/unfuse design rests on unfuse being lossless enough.

    PEFT merges by adding `B @ A * scale` to the base weight and unmerges by
    subtracting the same product, so the only question is fp16 rounding. This
    checks it on a layer the size of an SDXL attention projection, at a LoRA
    rank and magnitude in the range a 4-step distillation LoRA actually uses,
    and checks that repeated switching does not drift.
    """
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = torch.nn.Sequential()
    model.add_module("to_q", torch.nn.Linear(1280, 1280, bias=False))
    model = model.half()
    original = model.to_q.weight.detach().clone()

    wrapped = get_peft_model(
        model, LoraConfig(r=64, lora_alpha=64, target_modules=["to_q"], lora_dropout=0.0)
    )
    layer = wrapped.base_model.model.to_q
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(0, 0.02)
        layer.lora_B["default"].weight.normal_(0, 0.02)
    layer.half()

    layer.merge()
    fused = layer.base_layer.weight.detach().clone()
    applied = (fused.float() - original.float()).abs().max().item()
    assert applied > 1e-3, "the fixture LoRA must actually change the weights"

    layer.unmerge()
    residual = (layer.base_layer.weight.float() - original.float()).abs().max().item()
    # Under a thousandth of the change the fuse made: far below fp16's own
    # representable step at these magnitudes.
    assert residual < applied / 100

    for _ in range(9):
        layer.merge()
        layer.unmerge()
    drifted = (layer.base_layer.weight.float() - original.float()).abs().max().item()
    # Each cycle subtracts the same product it added, so the error does not
    # accumulate across a session's worth of profile switches.
    assert drifted <= residual * 2


async def test_a_cancelled_load_still_finishes_and_is_not_repeated() -> None:
    """Cancelling the awaiting task does not stop the thread.

    If `_loaded` were set by the awaiting task rather than on the thread, the
    retry would find it false and load the model - tens of seconds and several
    gigabytes - a second time.
    """
    loads = []
    started = threading.Event()
    proceed = threading.Event()

    class SlowLoad(DryRunPipeline):
        def load(self) -> None:
            started.set()
            proceed.wait(3.0)
            loads.append(1)
            super().load()

    backend = InprocBackend(settings(), pipeline=SlowLoad(settings()))
    first = asyncio.ensure_future(backend.load())
    await asyncio.get_running_loop().run_in_executor(None, started.wait, 3.0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    # The thread is still going; it finishes on its own.
    proceed.set()
    await backend.load()
    assert loads == [1], "the model was loaded twice"
    assert backend.loaded is True


# ---------------------------------------------------- the allocator cleanup
#
# Moved here with the pipeline itself (it used to live in
# apps/stream-worker/tests/test_contract.py, next to the copy of this file that
# the worker kept). No GPU: torch is faked, which is the point - what is being
# checked is that the `finally` runs on every exit path.

class FakeTorch:
    """Just enough torch for InprocPipeline.generate's bookkeeping."""

    def __init__(self) -> None:
        self.empty_cache_calls = 0
        outer = self

        class _Cuda:
            @staticmethod
            def synchronize() -> None:
                pass

            @staticmethod
            def empty_cache() -> None:
                outer.empty_cache_calls += 1

        class _Generator:
            def __init__(self, device: str | None = None) -> None:
                pass

            def manual_seed(self, seed: int) -> "_Generator":
                return self

        self.cuda = _Cuda()
        self.Generator = _Generator


def pipeline_that_fails(error: Exception):
    """An InprocPipeline whose diffusion call raises, with torch faked out."""
    p = InprocPipeline(PipelineSettings())
    p._torch = FakeTorch()
    p._embeds = lambda prompt, negative, cfg: (None, None, None, None)  # type: ignore[assignment]

    def boom(**kwargs):
        raise error

    p.pipe = boom
    return p


def run_generate(p, **overrides):
    return p.generate(
        image=Image.new("RGB", (64, 64), (255, 255, 255)),
        mask=None,
        prompt="x",
        negative_prompt="",
        strength=0.8,
        steps=4,
        seed=1,
        width=64,
        height=64,
        **overrides,
    )


def test_allocator_cache_is_returned_when_a_run_is_cancelled():
    # The cleanup used to sit after the return, so a cancelled run skipped it -
    # and cancellation is precisely when the next request is about to arrive.
    # Leaving the allocator holding the activations reintroduces the 2-8x spill.
    p = pipeline_that_fails(GenerationCancelled("req-1"))
    with pytest.raises(GenerationCancelled):
        run_generate(p)
    assert p._torch.empty_cache_calls == 1


def test_allocator_cache_is_returned_when_a_run_raises():
    p = pipeline_that_fails(RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError):
        run_generate(p)
    assert p._torch.empty_cache_calls == 1


def test_allocator_cleanup_can_be_switched_off():
    import dataclasses as dc

    p = InprocPipeline(dc.replace(PipelineSettings(), empty_cache_each_run=False))
    p._torch = FakeTorch()
    p._embeds = lambda prompt, negative, cfg: (None, None, None, None)  # type: ignore[assignment]

    def boom(**kwargs):
        raise RuntimeError("nope")

    p.pipe = boom
    with pytest.raises(RuntimeError):
        run_generate(p)
    assert p._torch.empty_cache_calls == 0


def test_cleanup_failure_does_not_mask_the_real_error():
    # A driver-level empty_cache failure inside `finally` must not replace the
    # exception the caller actually needs to see.
    p = pipeline_that_fails(RuntimeError("the real problem"))

    def explode() -> None:
        raise RuntimeError("cleanup blew up")

    p._torch.cuda.empty_cache = explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="the real problem"):
        run_generate(p)


# ------------------------------------------------------- how the UNet is kept


GIB = 1024**3


@pytest.mark.parametrize(
    "setting, memory, expected",
    [
        # `auto` is about the card. An 8 GB 3070 stays in fp16: with a tiled
        # decode the fp16 UNet fits and is quicker than fp8 by more than a
        # second an edit. Below that, fp8 is what makes the model fit at all.
        ("auto", 6 * GIB, "fp8"),
        ("auto", 8 * GIB, "fp16"),
        ("auto", 24 * GIB, "fp16"),
        ("", 6 * GIB, "fp8"),
        # A card whose size we could not read is the case that has always
        # worked: fp16.
        ("auto", None, "fp16"),
        ("auto", 0, "fp16"),
        # An explicit choice is an instruction, on any card.
        ("fp16", 6 * GIB, "fp16"),
        ("fp8", 24 * GIB, "fp8"),
        ("FP8", 24 * GIB, "fp8"),
    ],
)
def test_unet_storage_follows_the_card_unless_told(setting, memory, expected) -> None:
    assert choose_unet_storage(setting, memory) == expected


def test_the_thresholds_sit_where_the_measurements_put_them() -> None:
    # fp8 is for a card that cannot hold the fp16 UNet at all, which an 8 GB
    # one can - so the line is below 8 GB, not at it.
    assert 6 * GIB < FP8_BELOW_BYTES <= 8 * GIB
    # Tiling is for anything without room to decode a 1024 in one piece, which
    # includes the 8 GB card.
    assert 8 * GIB < TILE_BELOW_BYTES


def test_an_unknown_storage_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError) as err:
        choose_unet_storage("int4", 8 * GIB)
    assert "INPROC_UNET_STORAGE" in str(err.value)


def test_the_setting_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("INPROC_UNET_STORAGE", "fp8")
    assert PipelineSettings().unet_storage == "fp8"
    monkeypatch.delenv("INPROC_UNET_STORAGE")
    monkeypatch.setenv("STREAM_UNET_STORAGE", "fp16")  # the worker's spelling
    assert PipelineSettings().unet_storage == "fp16"
    monkeypatch.delenv("STREAM_UNET_STORAGE")
    assert PipelineSettings().unet_storage == "auto"


class FakeUnet:
    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.casting = None
        self.moved_to: list = []

    def to(self, device):
        self.moved_to.append(device)
        return self

    def enable_layerwise_casting(self, storage_dtype=None, compute_dtype=None):
        if not self.works:
            # Diffusers converts module by module with no rollback, so a
            # failure here has already changed some of the weights.
            self.casting = "half-converted"
            raise RuntimeError("out of memory converting block 14")
        self.casting = (storage_dtype, compute_dtype)


class DtypesOnlyTorch:
    """Just the two dtype objects `_store_unet_as_fp8` reaches for."""

    float16 = "fp16"
    float8_e4m3fn = "fp8"


def test_fp8_storage_attaches_the_lora_instead_of_fusing_it() -> None:
    """Fusing writes the merged weights back into the parameter, which fp8
    storage would quantise, and unfuse would then have nothing exact to take
    back out. So under fp8 the adapter stays attached."""
    pipeline = a_pipeline()
    pipeline._torch = DtypesOnlyTorch()
    pipeline.pipe.unet = FakeUnet()
    pipeline._store_unet_as_fp8(pipeline.pipe)

    assert pipeline.unet_storage == "fp8"
    assert pipeline.pipe.unet.casting == ("fp8", "fp16")
    assert pipeline._fuse_unavailable is True

    pipeline._select_profile("fast")
    assert pipeline._fused is False
    assert "fuse_lora:unet:1.0" not in pipeline.pipe.calls
    assert pipeline.pipe.calls[-1] == "set_adapters:fast"
    pipeline._select_profile("quality")
    assert pipeline.pipe.calls[-1] == "disable_lora"
    assert pipeline.describe_storage() == "unet fp8, lora off, vae tiles library default"


def test_a_build_without_layerwise_casting_stays_in_fp16() -> None:
    """Nothing has been touched in this case, so fp16 is the truth."""

    class NoCasting:
        pass

    pipeline = a_pipeline()
    pipeline._torch = DtypesOnlyTorch()
    pipeline.pipe.unet = NoCasting()
    pipeline._store_unet_as_fp8(pipeline.pipe)
    assert pipeline.unet_storage == "fp16"
    assert pipeline._fuse_unavailable is False  # fusing is still on the table


def test_casting_that_fails_part_way_refuses_to_load() -> None:
    """A failure mid-conversion leaves some blocks in fp8 and some in fp16.
    Calling that fp16 would let the LoRA be fused into quantised weights, and
    there is no rollback short of reloading the checkpoint - so it is fatal."""
    pipeline = a_pipeline()
    pipeline._torch = DtypesOnlyTorch()
    pipeline.pipe.unet = FakeUnet(works=False)
    with pytest.raises(RuntimeError) as err:
        pipeline._store_unet_as_fp8(pipeline.pipe)
    assert "INPROC_UNET_STORAGE=fp16" in str(err.value)
    assert "out of memory converting block 14" in str(err.value)


def test_the_unet_is_converted_before_it_is_moved_to_the_card() -> None:
    """A card small enough to want fp8 cannot hold the fp16 UNet on the way to
    it, so the conversion happens in system RAM."""
    pipeline = a_pipeline()
    pipeline._torch = DtypesOnlyTorch()
    unet = FakeUnet()
    pipeline.pipe.unet = unet
    pipeline._store_unet_as_fp8(pipeline.pipe)
    assert unet.casting == ("fp8", "fp16")
    assert unet.moved_to == [], "the fp16 UNet was sent to the card first"
    pipeline._to_device(pipeline.pipe)
    assert unet.moved_to == ["cuda"]


def test_parked_text_encoders_never_go_to_the_card() -> None:
    import dataclasses as dc

    class Part:
        def __init__(self) -> None:
            self.moved_to: list = []

        def to(self, device):
            self.moved_to.append(device)
            return self

    pipeline = a_pipeline()
    pipe = pipeline.pipe
    pipe.unet, pipe.vae = Part(), Part()
    pipe.text_encoder, pipe.text_encoder_2 = Part(), Part()
    pipeline._to_device(pipe)
    assert pipe.unet.moved_to == ["cuda"] and pipe.vae.moved_to == ["cuda"]
    assert pipe.text_encoder.moved_to == [], "1.8 GB across the bus and straight back"
    assert pipe.text_encoder_2.moved_to == []

    # ...and when they are not being offloaded, they go with everything else.
    pipeline = InprocPipeline(dc.replace(settings(), offload_text_encoders=False))
    pipeline.pipe = pipe = FakePipe()
    pipe.unet, pipe.vae = Part(), Part()
    pipe.text_encoder, pipe.text_encoder_2 = Part(), Part()
    pipeline._to_device(pipe)
    assert pipe.text_encoder.moved_to == ["cuda"]
    assert pipe.text_encoder_2.moved_to == ["cuda"]


@pytest.mark.parametrize(
    "setting, memory, expected",
    [
        ("auto", 8 * GIB, SMALL_CARD_VAE_TILE),
        ("auto", 24 * GIB, 0),  # a card with headroom decodes in one piece
        ("auto", None, 0),
        ("512", 8 * GIB, 512),
        ("0", 8 * GIB, 0),  # the library's own default, explicitly
    ],
)
def test_the_vae_tile_follows_the_card_unless_told(setting, memory, expected) -> None:
    assert choose_vae_tile(setting, memory) == expected


def test_a_tile_size_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(ValueError) as err:
        choose_vae_tile("small", 8 * GIB)
    assert "INPROC_VAE_TILE_SIZE" in str(err.value)


@pytest.mark.parametrize("tile", [1, 8, 100, 200, 64, 1088, 2048, -0 + 63])
def test_a_tile_the_decoder_cannot_use_is_refused_at_configuration_time(tile) -> None:
    """The decoder strides by the latent tile less a quarter of it, so a tile
    that is not a multiple of 64 lays out a grid that does not line up - and at
    64 or less the stride reaches 0 and the decode never terminates."""
    with pytest.raises(ValueError) as err:
        choose_vae_tile(str(tile), 8 * GIB)
    assert "multiple of 64" in str(err.value)


@pytest.mark.parametrize("tile", [128, 256, 512, 768, 1024])
def test_the_tiles_that_do_work_are_accepted(tile) -> None:
    assert choose_vae_tile(str(tile), 8 * GIB) == tile


def test_the_tile_is_pushed_into_the_vae_in_both_spaces() -> None:
    """diffusers keeps the sample-space tile and its latent-space twin as two
    separate attributes and uses the latent one to lay out the grid, so setting
    only the first would tile at the wrong size."""
    from brushjam.ai.pipeline import set_vae_tile_size, vae_latent_tile

    class Config:
        block_out_channels = [128, 256, 512, 512]

    class FakeVae:
        config = Config()
        tile_sample_min_size = 1024
        tile_latent_min_size = 128

    vae = FakeVae()
    assert set_vae_tile_size(vae, 256) is True
    assert vae.tile_sample_min_size == 256
    assert vae.tile_latent_min_size == 32  # 256 / 8, the SDXL VAE's ratio
    assert vae_latent_tile(768, 4) == 96

    assert set_vae_tile_size(vae, 0) is False  # nothing asked for, nothing done
    assert vae.tile_sample_min_size == 256

    class NoTiling:
        pass

    assert set_vae_tile_size(NoTiling(), 256) is False
