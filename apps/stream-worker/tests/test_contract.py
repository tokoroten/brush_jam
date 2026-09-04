"""GPU-free tests: the HTTP contract, mask compositing, and step scaling."""

from __future__ import annotations

import base64
import dataclasses
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from stream_worker.app import create_app
from stream_worker.config import Settings
from stream_worker.pipeline import (
    composite_through_mask,
    decode_png_b64,
    encode_png_b64,
    round_size,
    lcm_timesteps_for_strength,
    steps_for_strength,
)


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def base_request(size: int = 256) -> dict:
    """Smallest valid /generate body; callers override the fields they test."""
    return {"image_b64": png_b64(Image.new("RGB", (size, size), (255, 255, 255))), "size": size}


class FakePipeline:
    """Stands in for StreamPipeline: returns a flat colour, records the call."""

    backend = "fake"

    def __init__(self) -> None:
        self.warm = True
        self.calls: list[dict] = []
        self.loads = 0
        self.unloads = 0

    def __init_extra__(self) -> None:
        pass

    def load(self) -> None:
        self.loads += 1
        self.warm = True

    def unload(self) -> None:
        self.unloads += 1
        self.warm = False

    def model_name(self) -> str:
        return "fake-model"

    def memory(self) -> dict:
        return {}

    def generate(self, **kwargs):
        from stream_worker.pipeline import GenerateResult

        # should_cancel / request_id are part of the pipeline signature now;
        # the fake ignores them but must accept them.
        kwargs.pop("should_cancel", None)
        kwargs.pop("request_id", None)
        self.calls.append(kwargs)
        generated = Image.new("RGB", (kwargs["width"], kwargs["height"]), (255, 0, 0))
        out = composite_through_mask(kwargs["image"].convert("RGB"), generated, kwargs["mask"])
        return GenerateResult(image=out, timings={"diffusion_ms": 1.0, "total_ms": 2.0})


@pytest.fixture()
def client_and_pipe():
    pipe = FakePipeline()
    app = create_app(Settings(), pipeline=pipe)
    with TestClient(app) as client:
        yield client, pipe


def test_healthz_reports_backend_and_warm(client_and_pipe):
    client, _ = client_and_pipe
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["backend"] == "fake"
    assert body["model"] == "fake-model"
    assert body["warm"] is True
    assert body["size"] > 0


def test_healthz_publishes_the_sampling_settings_the_server_needs(client_and_pipe):
    # The grid script's results.json records nothing about sampling for this
    # backend, so /healthz is the only place these appear. The server greys out
    # its negative-prompt field on negative_prompt_active and caps its denoise
    # slider on max_denoise; both must survive refactors.
    client, _ = client_and_pipe
    body = client.get("/healthz").json()
    assert body["lora"] == "dmd2"
    assert body["guidance"] == 1.0
    assert body["negative_prompt_active"] is False
    assert body["max_denoise"] == 0.9
    assert body["steps"] == 4


def test_defaults_are_dmd2_at_cfg_one():
    # Changing either of these changes every room's output, so pin them.
    s = Settings()
    assert s.lora == "dmd2"
    assert s.guidance == 1.0
    assert s.lora_spec()[2] == "dmd2_sdxl_4step_lora_fp16.safetensors"


def test_negative_prompt_is_only_active_above_guidance_one():
    # CFG is what makes the negative prompt do anything: at 1.0 the pipeline
    # runs the conditional branch only and the negative embeddings are unused.
    assert Settings().negative_prompt_active() is False
    assert replace_guidance(1.0).negative_prompt_active() is False
    assert replace_guidance(1.5).negative_prompt_active() is True
    assert replace_guidance(1.01).negative_prompt_active() is True


def replace_guidance(value: float) -> Settings:
    return dataclasses.replace(Settings(), guidance=value)


def test_healthz_reflects_a_non_default_guidance(monkeypatch):
    monkeypatch.setenv("STREAM_GUIDANCE", "1.5")
    monkeypatch.setenv("STREAM_LORA", "lcm")
    app = create_app(Settings(), pipeline=FakePipeline())
    with TestClient(app) as client:
        body = client.get("/healthz").json()
    assert body["lora"] == "lcm"
    assert body["negative_prompt_active"] is True


def test_healthz_publishes_the_vae(client_and_pipe):
    # stream.ts records body.vae into its quality-grid reports; without this the
    # only sampling parameter that distinguishes two runs is missing from them.
    client, _ = client_and_pipe
    assert client.get("/healthz").json()["vae"] == "fp16fix"


def test_response_reports_the_steps_actually_run(client_and_pipe):
    # denoise 0.2 leaves 10 timesteps on a 50-point schedule, so 20 steps
    # cannot happen. The response must not claim they did.
    client, _ = client_and_pipe
    body = client.post("/generate", json={**base_request(), "denoise": 0.2, "steps": 20}).json()
    assert body["steps"] == 10
    assert len(lcm_timesteps_for_strength(20, 0.2)) == 10


def test_response_steps_match_the_request_when_they_fit(client_and_pipe):
    client, _ = client_and_pipe
    body = client.post("/generate", json={**base_request(), "denoise": 0.8, "steps": 4}).json()
    assert body["steps"] == 4


def test_strict_steps_refuses_an_impossible_combination(client_and_pipe):
    client, _ = client_and_pipe
    r = client.post("/generate", json={**base_request(), "denoise": 0.2, "steps": 20, "strict_steps": True})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "10 distinct timesteps" in detail
    assert "cannot run 20" in detail


def test_strict_steps_allows_a_combination_that_fits(client_and_pipe):
    client, _ = client_and_pipe
    r = client.post("/generate", json={**base_request(), "denoise": 0.8, "steps": 4, "strict_steps": True})
    assert r.status_code == 200
    assert r.json()["steps"] == 4


def test_strict_steps_is_off_by_default_so_the_server_contract_is_unchanged(client_and_pipe):
    client, _ = client_and_pipe
    r = client.post("/generate", json={**base_request(), "denoise": 0.2, "steps": 20})
    assert r.status_code == 200


def test_generate_returns_png_of_requested_size(client_and_pipe):
    client, pipe = client_and_pipe
    res = client.post(
        "/generate",
        json={
            "image_b64": png_b64(Image.new("RGB", (512, 512), "white")),
            "prompt": "anime style, fantasy town",
            "negative_prompt": "blurry",
            "denoise": 0.55,
            "steps": 4,
            "seed": 7,
            "width": 512,
            "height": 512,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    out = decode_png_b64(body["image_b64"])
    assert out.size == (512, 512)
    assert body["width"] == 512 and body["height"] == 512
    assert "total_ms" in body["timings"] and "wait_ms" in body["timings"]
    assert body["request_id"]
    # `denoise` must reach the pipeline as `strength`.
    assert pipe.calls[0]["strength"] == pytest.approx(0.55)
    assert pipe.calls[0]["steps"] == 4


def test_black_mask_returns_the_input_untouched(client_and_pipe):
    client, _ = client_and_pipe
    drawing = Image.new("RGB", (256, 256), (12, 200, 34))
    res = client.post(
        "/generate",
        json={
            "image_b64": png_b64(drawing),
            "mask_b64": png_b64(Image.new("L", (256, 256), 0)),
            "prompt": "x",
            "size": 256,
        },
    )
    assert res.status_code == 200
    out = decode_png_b64(res.json()["image_b64"])
    assert out.getpixel((128, 128)) == (12, 200, 34)


def test_white_mask_returns_the_generated_pixels(client_and_pipe):
    client, _ = client_and_pipe
    res = client.post(
        "/generate",
        json={
            "image_b64": png_b64(Image.new("RGB", (256, 256), (12, 200, 34))),
            "mask_b64": png_b64(Image.new("L", (256, 256), 255)),
            "prompt": "x",
            "size": 256,
        },
    )
    out = decode_png_b64(res.json()["image_b64"])
    assert out.getpixel((128, 128)) == (255, 0, 0)


def test_bad_image_is_a_400(client_and_pipe):
    client, _ = client_and_pipe
    res = client.post("/generate", json={"image_b64": "not-base64-png", "prompt": "x"})
    assert res.status_code == 400


def test_oversized_request_is_a_400(client_and_pipe):
    client, _ = client_and_pipe
    res = client.post(
        "/generate",
        json={"image_b64": png_b64(Image.new("RGB", (64, 64), "white")), "prompt": "x", "size": 4096},
    )
    assert res.status_code == 400


def test_data_url_prefix_is_accepted():
    img = Image.new("RGB", (8, 8), "white")
    assert decode_png_b64("data:image/png;base64," + encode_png_b64(img)).size == (8, 8)


def test_round_size_snaps_to_multiple_of_eight():
    assert round_size(1023, 1024) == 1016
    assert round_size(768, 1024) == 768
    with pytest.raises(ValueError):
        round_size(2048, 1024)


def test_steps_for_strength_keeps_the_requested_step_count():
    # Legacy path, no longer used for sampling. 4 steps at 0.55 strength would
    # really run 2, so it asked the scheduler for 8.
    assert steps_for_strength(4, 0.55) == 8
    assert steps_for_strength(4, 1.0) == 4
    assert steps_for_strength(6, 0.5) == 12


def test_steps_for_strength_is_why_08_and_09_collapsed():
    # The defect this replaced: both round to the same scheduler step count and
    # then to the same start index, so 0.8 and 0.9 were byte-identical.
    assert steps_for_strength(4, 0.8) == steps_for_strength(4, 0.9) == 5
    assert 5 - int(5 * 0.8) == 5 - int(5 * 0.9) == 1


def test_lcm_schedule_runs_exactly_the_requested_steps():
    for strength in (0.5, 0.65, 0.8, 0.9, 1.0):
        assert len(lcm_timesteps_for_strength(4, strength)) == 4
    assert len(lcm_timesteps_for_strength(8, 0.9)) == 8
    assert len(lcm_timesteps_for_strength(1, 0.8)) == 1


def test_lcm_schedule_separates_nearby_strengths():
    # The whole point of the rewrite: distinct starting timesteps, and so
    # distinct images, for the four denoise values the grids use.
    starts = [lcm_timesteps_for_strength(4, d)[0] for d in (0.5, 0.65, 0.8, 0.9)]
    assert starts == [499, 639, 799, 899]
    assert len(set(starts)) == 4
    # Resolution is one distillation step (2% at N=50), not 25%.
    assert lcm_timesteps_for_strength(4, 0.80)[0] != lcm_timesteps_for_strength(4, 0.84)[0]


def test_lcm_schedule_is_strictly_descending_and_ends_at_the_bottom():
    for strength in (0.4, 0.65, 0.9, 1.0):
        ts = lcm_timesteps_for_strength(4, strength)
        assert ts == sorted(ts, reverse=True)
        assert len(set(ts)) == len(ts)
        # Always finishes on the last distillation timestep, so the sample is
        # fully denoised however high up the schedule it started.
        assert ts[-1] == 19


def test_lcm_schedule_stays_on_the_distillation_grid():
    # Off-grid timesteps are timesteps the LCM LoRA was never distilled at;
    # diffusers warns about them and quality suffers.
    grid = {20 * i - 1 for i in range(1, 51)}
    for strength in (0.3, 0.5, 0.65, 0.8, 0.9, 1.0):
        assert set(lcm_timesteps_for_strength(4, strength)) <= grid


def test_lcm_schedule_start_rises_monotonically_with_strength():
    previous = -1
    for i in range(2, 101):
        start = lcm_timesteps_for_strength(4, i / 100)[0]
        assert start >= previous
        previous = start
    assert lcm_timesteps_for_strength(4, 1.0)[0] == 999


def test_lcm_schedule_degrades_gracefully_at_very_low_strength():
    # Below ~4/50 there are not 4 distinct timesteps left; run fewer rather
    # than repeat one, which would make the scheduler take zero-length steps.
    ts = lcm_timesteps_for_strength(4, 0.04)
    assert len(ts) == len(set(ts)) <= 4
    assert ts == sorted(ts, reverse=True)


def test_soft_mask_blends_proportionally():
    base = Image.new("RGB", (4, 4), (0, 0, 0))
    gen = Image.new("RGB", (4, 4), (200, 200, 200))
    out = composite_through_mask(base, gen, Image.new("L", (4, 4), 128))
    assert out.getpixel((1, 1)) == (100, 100, 100)


def test_mask_is_resized_to_the_image():
    base = Image.new("RGB", (64, 64), (0, 0, 0))
    gen = Image.new("RGB", (64, 64), (255, 255, 255))
    out = composite_through_mask(base, gen, Image.new("L", (16, 16), 255))
    assert out.size == (64, 64)
    assert out.getpixel((32, 32)) == (255, 255, 255)


def test_dry_run_app_echoes_input():
    app = create_app(Settings(dry_run=True), pipeline=None)
    with TestClient(app) as client:
        assert client.get("/healthz").json()["backend"] == "dry-run"
        res = client.post(
            "/generate",
            json={"image_b64": png_b64(Image.new("RGB", (256, 256), (9, 9, 9))), "prompt": "x", "size": 256},
        )
        assert res.status_code == 200
        assert decode_png_b64(res.json()["image_b64"]).getpixel((4, 4)) == (9, 9, 9)


def test_unload_frees_the_model_and_generate_reloads(client_and_pipe):
    """The 8 GB handover: /unload releases the GPU, the next call reloads."""
    client, pipe = client_and_pipe
    assert client.get("/healthz").json()["loaded"] is True

    assert client.post("/unload").json() == {"ok": True, "loaded": False}
    assert pipe.unloads == 1
    assert client.get("/healthz").json()["loaded"] is False

    loads_before = pipe.loads
    res = client.post(
        "/generate",
        json={"image_b64": png_b64(Image.new("RGB", (256, 256), "white")), "prompt": "x", "size": 256},
    )
    assert res.status_code == 200
    assert pipe.loads == loads_before + 1
    assert client.get("/healthz").json()["loaded"] is True


def test_load_is_idempotent(client_and_pipe):
    client, pipe = client_and_pipe
    loads_before = pipe.loads
    assert client.post("/load").json() == {"ok": True, "loaded": True}
    assert pipe.loads == loads_before  # already loaded: no second load


class FakeTorch:
    """Just enough torch for StreamPipeline.generate's bookkeeping."""

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
    """A StreamPipeline whose diffusion call raises, with torch faked out."""
    from stream_worker.pipeline import StreamPipeline

    p = StreamPipeline(Settings())
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
    from stream_worker.pipeline import CancelledError as PipelineCancelled

    p = pipeline_that_fails(PipelineCancelled("req-1"))
    with pytest.raises(PipelineCancelled):
        run_generate(p)
    assert p._torch.empty_cache_calls == 1


def test_allocator_cache_is_returned_when_a_run_raises():
    p = pipeline_that_fails(RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError):
        run_generate(p)
    assert p._torch.empty_cache_calls == 1


def test_allocator_cleanup_can_be_switched_off():
    import dataclasses as dc

    from stream_worker.pipeline import StreamPipeline

    p = StreamPipeline(dc.replace(Settings(), empty_cache_each_run=False))
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
