"""Config parsing and validation (the retired Node server's test/config.test.ts)."""

from __future__ import annotations

import pytest

from brushjam.ai.backends.base import BackendCapabilities
from brushjam.config import Config, ConfigError, load_config, resolve_backend_config, steps_for_profile


def test_defaults() -> None:
    config = load_config({})
    assert config.host == "127.0.0.1"
    assert config.port == 8787
    assert config.ai_backend == "auto"
    assert config.ai_mode == "full"
    assert config.canvas_size == 1024
    assert config.ai_profile == "fast"
    assert config.ai_window == 768  # the fast profile's size
    assert config.ai_apply == 1024  # full mode applies the whole canvas
    assert config.max_resolution == 1024  # a room may still switch to quality
    assert config.ai_denoise == 0.7


def test_full_mode_clamps_the_window_to_the_canvas_but_not_an_explicit_one() -> None:
    assert load_config({"CANVAS_SIZE": "512"}).ai_window == 512
    assert load_config({"CANVAS_SIZE": "512", "AI_WINDOW": "1024"}).ai_window == 1024
    # An explicit window is also a hard ceiling for the room control.
    assert load_config({"AI_WINDOW": "512"}).max_resolution == 512


@pytest.mark.parametrize(
    "env,fragment",
    [
        ({"PORT": "no"}, "PORT must be a number"),
        ({"PORT": "70000"}, "PORT must be between 1 and 65535"),
        ({"CANVAS_SIZE": "1000"}, "CANVAS_SIZE must be a multiple of 64"),
        ({"CANVAS_SIZE": "4096"}, "AI_MODE=full needs CANVAS_SIZE <= 2048"),
        ({"AI_WINDOW": "300"}, "AI_WINDOW must be a multiple of 64"),
        ({"AI_DENOISE": "0.73"}, "AI_DENOISE must be a multiple of 0.05"),
        ({"AI_DENOISE": "0.99"}, "AI_DENOISE must be between 0.2 and 0.95"),
        ({"AI_VAE_TILE": "32"}, "AI_VAE_TILE must be 0"),
        ({"AI_BACKEND": "banana"}, "AI_BACKEND must be one of"),
        ({"AI_PROFILE": "turbo"}, "AI_PROFILE must be fast or quality"),
        ({"AI_MODE": "sideways"}, "AI_MODE must be full or patch"),
        ({"COMFYUI_URL": "not a url"}, "COMFYUI_URL is not a valid URL"),
        ({"AI_BACKEND": "runpod"}, "requires RUNPOD_ENDPOINT_ID"),
    ],
)
def test_bad_values_refuse_to_boot(env, fragment: str) -> None:
    with pytest.raises(ConfigError) as err:
        load_config(env)
    assert fragment in str(err.value)


def test_patch_mode_is_documented_as_not_ported() -> None:
    with pytest.raises(ConfigError) as err:
        load_config({"AI_MODE": "patch"})
    assert "not implemented by the Python server" in str(err.value)


def test_ai_fast_alias() -> None:
    assert load_config({"AI_FAST": "0"}).ai_profile == "quality"
    assert load_config({"AI_FAST": "1"}).ai_profile == "fast"
    # AI_PROFILE wins over the alias.
    assert load_config({"AI_FAST": "0", "AI_PROFILE": "fast"}).ai_profile == "fast"


def test_an_empty_fast_lora_demotes_comfyui_to_quality() -> None:
    config = load_config({"AI_BACKEND": "comfyui", "COMFYUI_FAST_LORA": ""})
    assert config.ai_profile == "quality"
    assert config.fast_disabled is True
    # ...but it says nothing about a backend that does not build a ComfyUI graph.
    assert load_config({"AI_BACKEND": "mock", "COMFYUI_FAST_LORA": ""}).ai_profile == "fast"


def test_steps_follow_the_profile() -> None:
    config = load_config({"AI_STEPS": "20", "AI_FAST_STEPS": "3"})
    assert steps_for_profile(config, "fast") == 3
    assert steps_for_profile(config, "quality") == 20


def test_the_backend_gets_the_last_word_on_an_unpinned_setting() -> None:
    config = load_config({"AI_BACKEND": "mock"})
    fast_only = BackendCapabilities(["fast"], 768, 0.8, {"fast": True, "quality": True})
    resolved = resolve_backend_config(config, "stream", fast_only)
    assert resolved.ai_profile == "fast"
    assert resolved.ai_window == 768
    # A few-step backend starts a room at 0.8: 0.7 barely moves the drawing at
    # 4 steps (docs/experiments/2026-09-05-stream/REPORT.md).
    assert resolved.ai_denoise == 0.8
    assert resolved.max_resolution == 768
    strict = BackendCapabilities(["fast"], 768, 0.6, {"fast": True, "quality": True})
    assert resolve_backend_config(config, "stream", strict).ai_denoise == 0.6
    # ...and an explicit value is never replaced by a backend default.
    pinned = load_config({"AI_DENOISE": "0.5", "AI_WINDOW": "512"})
    kept = resolve_backend_config(pinned, "inproc", BackendCapabilities(["fast", "quality"], 1024, 0.9))
    assert (kept.ai_denoise, kept.ai_window) == (0.5, 512)


def test_a_pinned_setting_the_backend_cannot_run_is_an_error() -> None:
    config = load_config({"AI_PROFILE": "quality", "AI_WINDOW": "1024"})
    fast_only = BackendCapabilities(["fast"], 768, 0.95, {"fast": True, "quality": True})
    with pytest.raises(ConfigError) as err:
        resolve_backend_config(config, "stream", fast_only)
    message = str(err.value)
    assert "AI_PROFILE=quality is not supported by the stream backend" in message
    assert "AI_WINDOW=1024 is larger than the stream backend accepts (768)" in message
