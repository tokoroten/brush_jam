"""Backend selection. Port of the retired Node server's src/ai/backends/index.ts, plus the
in-process pipeline this port exists for.

An explicit AI_BACKEND always wins. `auto` prefers the in-process model (no HTTP
hop and no second process holding VRAM), then a stream worker if it was opted
in, then ComfyUI, then the mock. Every probe is bounded so a dead endpoint
cannot delay startup, and the chosen backend is logged with the reason.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .base import (
    AbortedError,
    AIBackend,
    BackendCapabilities,
    BackendHttpError,
    GenerateRequest,
    delay,
)
from .comfyui import (
    DEFAULT_FAST_LORA,
    FAST_CFG,
    ComfyUIBackend,
    build_workflow,
    comfy_reachable,
    fast_profile,
    negative_prompt_active,
)
from .inproc import InprocBackend, torch_available
from .mock import MockBackend
from .runpod import RunpodBackend
from .stream import StreamBackend, StreamHealth, png_size, stream_health, stream_reachable

if TYPE_CHECKING:  # pragma: no cover
    from ...config import Config

log = logging.getLogger("brushjam.ai")

__all__ = [
    "AIBackend",
    "AbortedError",
    "BackendCapabilities",
    "BackendHttpError",
    "ComfyUIBackend",
    "DEFAULT_FAST_LORA",
    "FAST_CFG",
    "GenerateRequest",
    "InprocBackend",
    "MockBackend",
    "RunpodBackend",
    "StreamBackend",
    "StreamHealth",
    "build_workflow",
    "comfy_reachable",
    "create_backend",
    "delay",
    "fast_profile",
    "negative_prompt_active",
    "png_size",
    "stream_health",
    "stream_reachable",
    "torch_available",
    "watch_for_stream_worker",
]

#: Bounded so a dead endpoint cannot delay startup.
PROBE_TIMEOUT_MS = 2000
#: How often to look again for a stream worker that was not up at startup.
STREAM_RETRY_MS = 10_000


class Watcher:
    """Stoppable, so a startup watcher does not outlive the registry."""

    def __init__(self, task: asyncio.Task) -> None:
        self._task = task

    def stop(self) -> None:
        self._task.cancel()


def watch_for_stream_worker(
    url: str,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    interval_ms: float = STREAM_RETRY_MS,
    on_ready: Optional[Callable[[], Any]] = None,
) -> Watcher:
    """Poll until the worker answers, then say so once and stop. The backend
    needs no repair - it builds its URL per request - so appearing late is
    enough; the registry picks the real limits up on its own capability poll."""
    emit = log_fn or (lambda m: log.info("%s", m))

    async def loop() -> None:
        while True:
            await asyncio.sleep(interval_ms / 1000)
            health = await stream_health(url, PROBE_TIMEOUT_MS)
            if not health.ok:
                continue
            emit(
                f"stream worker is up at {url}; generations will use it from now on"
                if health.warm
                else f"stream worker is up at {url} but still loading its model; the first request will wait"
            )
            # Anyone who drew while it was down is owed a generation.
            if on_ready is not None:
                result = on_ready()
                if asyncio.iscoroutine(result):
                    await result
            return

    return Watcher(asyncio.ensure_future(loop()))


def _inproc_ready(config: "Config") -> Optional[str]:
    """Why the in-process backend cannot be used, or None when it can."""
    if config.inproc_dry_run:
        return None
    if not torch_available():
        return "torch/diffusers are not installed (uv sync --extra inproc)"
    checkpoint = Path(config.inproc_checkpoint)
    if not config.inproc_checkpoint:
        return (
            "no checkpoint is configured: set INPROC_CHECKPOINT in .env to an SDXL "
            ".safetensors file (uv run --project apps/brushjam python "
            "apps/brushjam/scripts/download_models.py downloads one)"
        )
    if not checkpoint.exists():
        return f"INPROC_CHECKPOINT points at a file that does not exist: {checkpoint}"
    return None


def _make_inproc(config: "Config") -> InprocBackend:
    from ..pipeline import PipelineSettings

    settings = PipelineSettings()
    if config.inproc_checkpoint:
        settings.checkpoint = Path(config.inproc_checkpoint)
    if config.inproc_lora_dir:
        settings.lora_dir = Path(config.inproc_lora_dir)
    return InprocBackend(settings, dry_run=config.inproc_dry_run)


async def create_backend(
    config: "Config",
    log_fn: Optional[Callable[[str], None]] = None,
    *,
    on_watcher: Optional[Callable[[Watcher], None]] = None,
    on_stream_ready: Optional[Callable[[], Any]] = None,
) -> AIBackend:
    emit = log_fn or (lambda m: log.info("%s", m))

    def comfy() -> AIBackend:
        return ComfyUIBackend(
            url=config.comfy_url,
            checkpoint=config.comfy_checkpoint,
            cfg=config.ai_cfg,
            vae_tile=config.ai_vae_tile,
            fast_lora=config.comfy_fast_lora or None,
        )

    def stream() -> AIBackend:
        # The worker may take ~90 s to answer its first request while it loads
        # the model, so it gets a generation-sized deadline, not a probe one.
        return StreamBackend(url=config.stream_url, timeout_ms=config.stream_timeout_ms)

    if config.ai_backend == "mock":
        emit("backend: mock (AI_BACKEND=mock)")
        return MockBackend()

    if config.ai_backend == "inproc":
        why = _inproc_ready(config)
        if why is not None:
            # An explicit choice is never silently replaced with a different
            # backend: that is how a room ends up quietly running the mock.
            raise RuntimeError(f"AI_BACKEND=inproc cannot run: {why}")
        emit(
            "backend: inproc (AI_BACKEND=inproc, dry run)"
            if config.inproc_dry_run
            else f"backend: inproc, {Path(config.inproc_checkpoint).name} resident in this process"
        )
        return _make_inproc(config)

    if config.ai_backend == "runpod":
        emit("backend: runpod")
        return RunpodBackend(
            endpoint_id=config.runpod_endpoint_id,
            api_key=config.runpod_api_key,
            checkpoint=config.comfy_checkpoint,
            cfg=config.ai_cfg,
            vae_tile=config.ai_vae_tile,
            fast_lora=config.comfy_fast_lora or None,
            timeout_ms=config.runpod_timeout_ms,
        )

    if config.ai_backend == "stream":
        emit(f"backend: stream at {config.stream_url} (AI_BACKEND=stream)")
        # An explicit choice is honoured either way, but say so now rather than
        # letting every generation fail with a puzzling 400.
        health = await stream_health(config.stream_url, PROBE_TIMEOUT_MS)
        if not health.ok:
            emit(f"warning: stream worker is not answering ({health.reason or 'no answer'})")
            emit("start it with: cd apps/stream-worker && uv run stream-worker   (~40 s to warm up)")
            watcher = watch_for_stream_worker(
                config.stream_url, log_fn=emit, on_ready=on_stream_ready
            )
            if on_watcher is not None:
                on_watcher(watcher)
        elif not health.warm:
            emit("warning: stream worker is reachable but not warm; the first request will load the model")
        elif 0 < health.max_size < config.ai_window:
            emit(
                f"warning: stream worker max_size {health.max_size} is below AI_WINDOW "
                f"{config.ai_window}; requests will be refused"
            )
        return stream()

    if config.ai_backend == "comfyui":
        emit(f"backend: comfyui at {config.comfy_url} (AI_BACKEND=comfyui)")
        return comfy()

    # ---- auto -------------------------------------------------------------
    why = _inproc_ready(config)
    if why is None:
        emit(
            f"backend: inproc (auto-detected: {Path(config.inproc_checkpoint).name} is present "
            "and torch is installed)"
        )
        return _make_inproc(config)
    emit(f"skipping the in-process model: {why}")

    # The stream worker is NOT auto-selected by default (docs/STREAM_WORKER.md
    # section 6): it answering /healthz only means it is holding ~5 GB of VRAM.
    if config.stream_auto:
        health = await stream_health(config.stream_url, PROBE_TIMEOUT_MS)
        if not health.ok:
            emit(f"skipping stream worker: {health.reason or 'no answer'}")
        elif not health.warm:
            # An unloaded or dry-run worker answers ok and then echoes the input.
            emit(f"skipping stream worker at {config.stream_url}: reachable but not warm (no model loaded)")
        elif 0 < health.max_size < config.ai_window:
            # It would 400 every request; full mode would retry that forever.
            emit(
                f"skipping stream worker at {config.stream_url}: max_size {health.max_size} "
                f"< AI_WINDOW {config.ai_window}"
            )
        else:
            emit(
                f"backend: stream at {config.stream_url} (auto-detected: warm, "
                f"max_size {health.max_size or 'unreported'})"
            )
            return stream()

    if await comfy_reachable(config.comfy_url, PROBE_TIMEOUT_MS):
        reason = "no usable stream worker" if config.stream_auto else "stream is explicit-only"
        emit(f"backend: comfyui at {config.comfy_url} (auto-detected: {reason})")
        return comfy()

    emit(
        f"backend: mock - ComfyUI ({config.comfy_url}) did not answer "
        f"(AI_BACKEND={config.ai_backend})"
    )
    return MockBackend()
