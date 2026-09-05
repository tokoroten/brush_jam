"""Environment parsing and validation. Port of apps/server/src/config.ts.

Every value is validated once, at startup, and the process refuses to boot on
bad input: a fractional or negative AI window would otherwise produce broken
crops and a room that throws the same render error every two seconds.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional
from urllib.parse import urlparse

from .constants import AI_PROFILES, MAX_AI_RESOLUTION, MIN_AI_RESOLUTION, PROFILE_DEFAULTS

#: LoRA the fast ComfyUI profile loads; kept so the setting keeps its meaning.
DEFAULT_FAST_LORA = "dmd2_sdxl_4step_lora_fp16.safetensors"

#: The 1024 the quality profile is documented to use.
QUALITY_CEILING = 1024

BACKENDS = ("auto", "comfyui", "mock", "runpod", "stream", "inproc")


class ConfigError(Exception):
    pass


@dataclass
class ExplicitEnv:
    """Which AI settings the operator pinned. Backend defaults may only fill in
    the rest."""

    window: bool = False
    apply: bool = False
    denoise: bool = False
    profile: bool = False
    steps: bool = False


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8787
    ai_backend: str = "auto"
    stream_url: str = "http://127.0.0.1:8790"
    stream_timeout_ms: int = 120_000
    stream_auto: bool = False
    comfy_url: str = "http://127.0.0.1:8188"
    comfy_checkpoint: str = "waiNSFWIllustrious_v150.safetensors"
    canvas_size: int = 1024
    ai_mode: str = "full"
    ai_window: int = 768
    ai_apply: int = 768
    ai_denoise: float = 0.7
    ai_cfg: float = 5.5
    ai_vae_tile: int = 512
    ai_profile: str = "fast"
    ai_steps: int = 14
    ai_fast_steps: int = 4
    fast_disabled: bool = False
    comfy_fast_lora: str = DEFAULT_FAST_LORA
    ai_debounce_ms: int = 400
    ai_watchdog_ms: int = 180_000
    room_idle_ms: int = 30 * 60_000
    runpod_endpoint_id: str = ""
    runpod_api_key: str = ""
    runpod_timeout_ms: int = 300_000
    web_dist: Optional[str] = None
    explicit: ExplicitEnv = field(default_factory=ExplicitEnv)
    #: Ceiling for a room's generation size; the starting size is ai_window.
    max_resolution: int = 0


def _flag(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _num(
    env: Mapping[str, str],
    key: str,
    fallback: float,
    *,
    min: float,
    max: float,
    integer: bool = False,
    multiple_of: Optional[int] = None,
    errors: List[str],
) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return fallback
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{key} must be a number (got {_json(raw)})")
        return fallback
    if not math.isfinite(value):
        errors.append(f"{key} must be a number (got {_json(raw)})")
        return fallback
    if integer and value != int(value):
        errors.append(f"{key} must be a whole number (got {_fmt(value)})")
    if value < min or value > max:
        errors.append(f"{key} must be between {_fmt(min)} and {_fmt(max)} (got {_fmt(value)})")
    if multiple_of and value % multiple_of != 0:
        errors.append(f"{key} must be a multiple of {multiple_of} (got {_fmt(value)})")
    return value


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(round(value, 12))


def _json(raw: Optional[str]) -> str:
    if raw is None:
        return "undefined"
    return '"' + raw.replace('"', '\\"') + '"'


def _valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


def load_config(env: Optional[Mapping[str, str]] = None) -> Config:
    env = os.environ if env is None else env
    errors: List[str] = []
    backend_raw = (env.get("AI_BACKEND") or "auto").lower()
    mode_raw = (env.get("AI_MODE") or "full").lower()
    if mode_raw not in ("full", "patch"):
        errors.append(f"AI_MODE must be full or patch (got {_json(env.get('AI_MODE'))})")
    elif mode_raw == "patch":
        # Documented in docs/PYTHON_SERVER_PLAN.md section 0: only full mode is
        # ported. Refusing to boot beats running a mode that silently is not.
        errors.append(
            "AI_MODE=patch is not implemented by the Python server (full-canvas mode only)"
        )
    if backend_raw not in BACKENDS:
        errors.append(
            "AI_BACKEND must be one of auto, comfyui, mock, runpod, stream "
            f"(got {_json(env.get('AI_BACKEND'))})"
        )

    # The fast profile is only fast if there is a LoRA to load.
    fast_lora = (env.get("COMFYUI_FAST_LORA") if "COMFYUI_FAST_LORA" in env else DEFAULT_FAST_LORA)
    fast_lora = (fast_lora or "").strip()
    # AI_PROFILE is the setting; AI_FAST survives as an alias.
    aliased = None if env.get("AI_FAST") is None else ("fast" if _flag(env.get("AI_FAST")) else "quality")
    profile_raw = (env.get("AI_PROFILE") or aliased or "fast").lower()
    if profile_raw not in AI_PROFILES:
        errors.append(f"AI_PROFILE must be fast or quality (got {_json(env.get('AI_PROFILE'))})")
    requested = "quality" if profile_raw == "quality" else "fast"
    fast_requested = requested == "fast"
    lora_decides = backend_raw in ("comfyui", "runpod")
    fast = fast_requested and (not lora_decides or fast_lora != "")
    profile = "fast" if fast else "quality"

    explicit = ExplicitEnv(
        window=bool(env.get("AI_WINDOW")),
        apply=bool(env.get("AI_APPLY")),
        denoise=bool(env.get("AI_DENOISE")),
        profile=env.get("AI_PROFILE") is not None or env.get("AI_FAST") is not None,
        steps=bool(env.get("AI_STEPS")),
    )

    config = Config(
        host=env.get("HOST") or "127.0.0.1",
        port=int(_num(env, "PORT", 8787, min=1, max=65535, integer=True, errors=errors)),
        ai_backend=backend_raw if backend_raw in BACKENDS else "auto",
        comfy_url=(env.get("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/"),
        stream_url=(env.get("STREAM_URL") or "http://127.0.0.1:8790").rstrip("/"),
        stream_timeout_ms=int(
            _num(env, "STREAM_TIMEOUT_MS", 120_000, min=1000, max=3_600_000, integer=True, errors=errors)
        ),
        stream_auto=_flag(env.get("AI_STREAM_AUTO")),
        comfy_checkpoint=env.get("COMFYUI_CHECKPOINT") or "waiNSFWIllustrious_v150.safetensors",
        canvas_size=int(
            _num(env, "CANVAS_SIZE", 1024, min=512, max=4096, integer=True, multiple_of=64, errors=errors)
        ),
        ai_mode="patch" if mode_raw == "patch" else "full",
        # The profile picks the window when the operator has not.
        ai_window=int(
            _num(
                env,
                "AI_WINDOW",
                PROFILE_DEFAULTS[profile]["resolution"],
                min=256,
                max=2048,
                integer=True,
                multiple_of=64,
                errors=errors,
            )
        ),
        ai_apply=int(
            _num(env, "AI_APPLY", 768, min=128, max=2048, integer=True, multiple_of=64, errors=errors)
        ),
        ai_steps=int(
            _num(env, "AI_STEPS", PROFILE_DEFAULTS["quality"]["steps"], min=1, max=150, integer=True, errors=errors)
        ),
        ai_fast_steps=int(
            _num(env, "AI_FAST_STEPS", PROFILE_DEFAULTS["fast"]["steps"], min=1, max=150, integer=True, errors=errors)
        ),
        ai_denoise=_num(env, "AI_DENOISE", PROFILE_DEFAULTS[profile]["denoise"], min=0, max=1, errors=errors),
        ai_cfg=_num(env, "AI_CFG", 5.5, min=0, max=30, errors=errors),
        ai_vae_tile=int(_num(env, "AI_VAE_TILE", 512, min=0, max=4096, integer=True, errors=errors)),
        ai_profile=profile,
        fast_disabled=fast_requested and not fast,
        comfy_fast_lora=fast_lora,
        ai_debounce_ms=int(
            _num(env, "AI_DEBOUNCE_MS", 400, min=0, max=600_000, integer=True, errors=errors)
        ),
        ai_watchdog_ms=int(
            _num(env, "AI_WATCHDOG_MS", 180_000, min=1000, max=3_600_000, integer=True, errors=errors)
        ),
        room_idle_ms=int(
            _num(env, "ROOM_IDLE_MS", 30 * 60_000, min=10_000, max=24 * 3_600_000, integer=True, errors=errors)
        ),
        runpod_endpoint_id=env.get("RUNPOD_ENDPOINT_ID") or "",
        runpod_api_key=env.get("RUNPOD_API_KEY") or "",
        runpod_timeout_ms=int(
            _num(env, "RUNPOD_TIMEOUT_MS", 300_000, min=1000, max=3_600_000, integer=True, errors=errors)
        ),
        web_dist=env.get("WEB_DIST") or None,
        explicit=explicit,
    )

    # ComfyUI's VAEDecodeTiled has a minimum tile of 64; 0 means "do not tile".
    if config.ai_vae_tile != 0 and config.ai_vae_tile < 64:
        errors.append(f"AI_VAE_TILE must be 0 (no tiling) or at least 64 (got {config.ai_vae_tile})")
    # The room slider only offers 0.2..0.95 in 0.05 steps.
    if config.ai_denoise < 0.2 or config.ai_denoise > 0.95:
        errors.append(f"AI_DENOISE must be between 0.2 and 0.95 (got {_fmt(config.ai_denoise)})")
    elif abs(config.ai_denoise * 100 - round(config.ai_denoise * 100 / 5) * 5) > 1e-6:
        errors.append(f"AI_DENOISE must be a multiple of 0.05 (got {_fmt(config.ai_denoise)})")

    _cross_field_errors(config, errors)
    config.max_resolution = _default_max_resolution(config)
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))
    return config


def _default_max_resolution(config: Config) -> int:
    """A room starts at ai_window but may go up to here. An explicit AI_WINDOW
    is a hard cap: an operator who pins the size means it."""
    ceiling = config.ai_window if config.explicit.window else max(config.ai_window, QUALITY_CEILING)
    return int(min(MAX_AI_RESOLUTION, max(MIN_AI_RESOLUTION, ceiling)))


def _cross_field_errors(config: Config, errors: List[str]) -> None:
    if config.ai_mode == "full":
        # The whole canvas is regenerated, but not necessarily at canvas
        # resolution: AI_WINDOW is the *generation* size.
        if config.canvas_size > 2048:
            errors.append(
                f"AI_MODE=full needs CANVAS_SIZE <= 2048 (got {config.canvas_size}); "
                "use the Node server for a large canvas"
            )
        if not config.explicit.window:
            config.ai_window = min(config.ai_window, config.canvas_size)
        if config.ai_window < 512:
            errors.append(f"AI_MODE=full needs AI_WINDOW >= 512 (got {config.ai_window})")
        # the whole canvas is always the applied area in full mode
        config.ai_apply = config.canvas_size

    if config.ai_mode == "patch" and config.ai_apply > config.ai_window:
        errors.append(f"AI_APPLY ({config.ai_apply}) must not exceed AI_WINDOW ({config.ai_window})")
    if not _valid_url(config.comfy_url):
        errors.append(f"COMFYUI_URL is not a valid URL (got {_json(config.comfy_url)})")
    if not _valid_url(config.stream_url):
        errors.append(f"STREAM_URL is not a valid URL (got {_json(config.stream_url)})")
    if config.ai_backend == "runpod" and (not config.runpod_endpoint_id or not config.runpod_api_key):
        errors.append("AI_BACKEND=runpod requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY")


def resolve_backend_config(
    config: Config, backend_name: str, capabilities
) -> Config:
    """Reconcile the configuration with the backend that was actually selected.

    An explicit choice is never silently replaced: a pinned profile the backend
    cannot run, or a pinned window larger than it accepts, is an error.
    """
    errors: List[str] = []
    next_config = replace(config)

    if not capabilities.profiles or next_config.ai_profile not in capabilities.profiles:
        because = (
            " - COMFYUI_FAST_LORA is empty, so there is no fast workflow to run"
            if next_config.ai_profile == "fast"
            and not config.comfy_fast_lora
            and backend_name in ("comfyui", "runpod")
            else ""
        )
        if config.explicit.profile:
            errors.append(
                f"AI_PROFILE={next_config.ai_profile} is not supported by the {backend_name} "
                f"backend (it offers {', '.join(capabilities.profiles)}){because}"
            )
        else:
            if next_config.ai_profile == "fast":
                next_config.fast_disabled = True
            next_config.ai_profile = capabilities.profiles[0] if capabilities.profiles else "quality"
            if not config.explicit.window:
                next_config.ai_window = min(
                    int(PROFILE_DEFAULTS[next_config.ai_profile]["resolution"]),
                    next_config.ai_window,
                )

    # A window the backend will refuse is a permanent 400 on every generation.
    if next_config.ai_window > capabilities.max_resolution:
        if config.explicit.window:
            errors.append(
                f"AI_WINDOW={next_config.ai_window} is larger than the {backend_name} backend "
                f"accepts ({capabilities.max_resolution})"
            )
        else:
            next_config.ai_window = int(
                max(MIN_AI_RESOLUTION, math.floor(capabilities.max_resolution / 64) * 64)
            )
    if next_config.ai_denoise > capabilities.max_denoise:
        if config.explicit.denoise:
            errors.append(
                f"AI_DENOISE={_fmt(next_config.ai_denoise)} is above what the {backend_name} "
                f"backend accepts ({_fmt(capabilities.max_denoise)})"
            )
        else:
            next_config.ai_denoise = capabilities.max_denoise

    next_config.max_resolution = int(
        min(
            next_config.ai_window if config.explicit.window else max(next_config.ai_window, 1024),
            capabilities.max_resolution,
            MAX_AI_RESOLUTION,
        )
    )

    _cross_field_errors(next_config, errors)
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))
    return next_config


def steps_for_profile(config: Config, profile: str) -> int:
    return config.ai_fast_steps if profile == "fast" else config.ai_steps


def env_origin(before: Optional[str], after: Optional[str]) -> str:
    """Where a setting's value came from, for the startup log."""
    if before:
        return "environment"
    if after:
        return ".env"
    return "unset"
