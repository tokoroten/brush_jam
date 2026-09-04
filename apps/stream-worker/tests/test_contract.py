"""GPU-free tests: the HTTP contract, mask compositing, and step scaling."""

from __future__ import annotations

import base64
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
    steps_for_strength,
)


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
    # 4 steps at 0.55 strength would really run 2, so ask the scheduler for 8.
    assert steps_for_strength(4, 0.55) == 8
    assert steps_for_strength(4, 1.0) == 4
    assert steps_for_strength(6, 0.5) == 12


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
