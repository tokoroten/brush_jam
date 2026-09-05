"""The in-process pipeline as an AIBackend.

This is what the plan is for: no HTTP hop, no second process holding VRAM, one
resident model that both room profiles switch between per request.

Concurrency: exactly one generation at a time, on a single dedicated thread, so
the event loop never blocks and two rooms can never be inside the pipeline at
once. Cancellation is cooperative - the async side gives up immediately, sets a
flag, and the pipeline thread stops at its next diffusion step; the next request
queues behind that thread rather than racing it onto the GPU, which is the same
"settle" behaviour the HTTP stream backend buys with /cancel.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from PIL import Image

from ...raster import to_png
from ..pipeline import (
    DryRunPipeline,
    GenerationCancelled,
    InprocPipeline,
    PipelineSettings,
    round_size,
)
from .base import BackendCapabilities, BackendHttpError, GenerateRequest

log = logging.getLogger("brushjam.ai.inproc")


class InprocBackend:
    name = "inproc"

    def __init__(
        self,
        settings: Optional[PipelineSettings] = None,
        pipeline: Any = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings or PipelineSettings()
        if pipeline is not None:
            self.pipeline = pipeline
        elif dry_run:
            self.pipeline = DryRunPipeline(self.settings)
        else:
            self.pipeline = InprocPipeline(self.settings)
        self.dry_run = dry_run or isinstance(self.pipeline, DryRunPipeline)
        #: One GPU, one generation at a time. A single-worker executor *is* the
        #: lock: a cancelled run keeps the thread until it notices, and the next
        #: request waits for it instead of overlapping on the device.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="brushjam-gpu")
        self._loaded = False
        self._error: Optional[str] = None
        self._busy = False
        self._current_request_id: Optional[str] = None
        self._cancelled: "set[str]" = set()
        self._load_lock = asyncio.Lock()
        #: The last generation's stage timings, for /healthz and the DEBUG log.
        self.last_timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ load
    async def load(self) -> None:
        """Load the model off the event loop. Failures are recorded rather than
        raised, so the server stays up and /healthz can report why."""
        async with self._load_lock:
            if self._loaded:
                return
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(self._executor, self.pipeline.load)
                self._loaded = True
                self._error = None
            except Exception as err:  # noqa: BLE001
                self._error = f"{type(err).__name__}: {err}"
                log.exception("model load failed")

    async def unload(self) -> None:
        """Drop the model and free the VRAM without exiting: on 8 GB this and
        ComfyUI cannot both be resident, so they hand the card over."""
        async with self._load_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.pipeline.unload)
            self._loaded = False
            self._error = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ---------------------------------------------------------- capabilities
    async def capabilities(self) -> BackendCapabilities:
        s = self.settings
        # Both profiles come from one model: quality is the checkpoint with the
        # LoRA detached, fast is the same weights with it attached.
        return BackendCapabilities(
            profiles=["fast", "quality"],
            max_resolution=s.max_size,
            max_denoise=s.max_denoise,
            negative_prompt_active=s.negative_prompt_active(),
        )

    def status(self) -> Dict[str, Any]:
        """The fields the stream worker publishes on /healthz, so the same
        tooling can read them off this server instead."""
        s = self.settings
        profile = getattr(self.pipeline, "_profile", None)
        return {
            "backend": getattr(self.pipeline, "backend", "inproc"),
            "model": self.pipeline.model_name(),
            "size": s.warmup_size,
            "max_size": s.max_size,
            "max_denoise": s.max_denoise,
            "steps": s.fast_steps if profile != "quality" else s.quality_steps,
            "guidance": s.guidance_for(profile or "fast"),
            "lora": s.lora,
            # Whether the fast adapter is merged into the UNet weights right
            # now, which is the difference between ~2 s and ~4 s at 768.
            "lora_fused": bool(getattr(self.pipeline, "_fused", False)),
            "vae": s.vae,
            "profile": profile,
            "negative_prompt_active": s.negative_prompt_active(),
            "warm": bool(self._loaded and getattr(self.pipeline, "warm", False)),
            "loaded": self._loaded,
            "busy": self._busy,
            "current_request_id": self._current_request_id,
            "error": self._error,
            "memory": self.pipeline.memory(),
            # The last run's stage breakdown, so a slow edit can be explained
            # without turning on DEBUG logging first.
            "last_timings": dict(self.last_timings),
        }

    # ------------------------------------------------------------- generation
    async def generate(self, req: GenerateRequest) -> bytes:
        if self._error is not None and not self._loaded:
            raise RuntimeError(self._error)
        if not self._loaded:
            await self.load()
            if not self._loaded:
                raise RuntimeError(self._error or "the model is not loaded")

        try:
            width = round_size(req.size, self.settings.max_size)
        except ValueError as err:
            # A refusal, not a failure: retrying the same size cannot work.
            raise BackendHttpError(str(err), 400) from err
        height = width

        # PNG codec work is CPU-bound and must not run on the event loop, and
        # it must not run on the single GPU thread either: that thread is the
        # device lock, so anything done there delays the next request's
        # diffusion for no reason.
        t = time.perf_counter()
        image, mask = await asyncio.to_thread(_decode_inputs, req.image_png, req.mask_png)
        decode_in_ms = (time.perf_counter() - t) * 1000.0
        # Clamped to max_denoise, not to 1.0: above it the model stops
        # reinterpreting the drawing and starts replacing it.
        strength = min(self.settings.max_denoise, max(0.05, req.denoise))
        steps = max(1, min(50, req.steps or self.settings.steps_for(req.profile)))
        request_id = uuid.uuid4().hex

        loop = asyncio.get_running_loop()

        def run() -> Any:
            self._busy = True
            self._current_request_id = request_id
            try:
                return self.pipeline.generate(
                    image=image,
                    mask=mask,
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    strength=strength,
                    steps=steps,
                    seed=req.seed,
                    width=width,
                    height=height,
                    profile=req.profile,
                    should_cancel=lambda: request_id in self._cancelled,
                    request_id=request_id,
                )
            finally:
                self._busy = False
                self._current_request_id = None
                self._cancelled.discard(request_id)

        queued_at = time.perf_counter()
        future = loop.run_in_executor(self._executor, run)
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            # The caller has given up. The thread is still inside the diffusion
            # loop and cannot be stopped from here, so ask it to stop at its
            # next step and leave it to unwind; the executor's single slot keeps
            # the next request behind it.
            self._cancelled.add(request_id)
            raise
        except GenerationCancelled as err:
            raise asyncio.CancelledError() from err

        t = time.perf_counter()
        # `compress_level=1` on purpose: this PNG exists only to hand the image
        # to the compositor in the same process. Level 6 costs ~5x the CPU for
        # bytes nobody transmits.
        out = await asyncio.to_thread(to_png, result.image, 1)
        encode_out_ms = (time.perf_counter() - t) * 1000.0

        timings = dict(result.timings)
        timings["decode_in_ms"] = round(decode_in_ms, 2)
        timings["encode_out_ms"] = round(encode_out_ms, 2)
        # Everything between handing the job to the GPU thread and getting the
        # result back that the pipeline did not account for: queueing behind a
        # previous run, plus the executor hop.
        timings["queue_ms"] = round(
            max(0.0, (t - queued_at) * 1000.0 - timings.get("total_ms", 0.0)), 2
        )
        self.last_timings = timings
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "backend %s: decode_in %.0f queue %.0f pipeline %.0f encode_out %.0f ms",
                request_id,
                decode_in_ms,
                timings["queue_ms"],
                timings.get("total_ms", 0.0),
                encode_out_ms,
            )
        return out


def _decode_inputs(
    image_png: bytes, mask_png: Optional[bytes]
) -> "tuple[Image.Image, Optional[Image.Image]]":
    return (
        _decode(image_png, "image"),
        _decode(mask_png, "mask") if mask_png else None,
    )


def _decode(data: bytes, what: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as raw:
            raw.load()
            return raw.copy()
    except Exception as err:  # noqa: BLE001
        raise BackendHttpError(f"{what} is not a decodable image: {err}", 400) from err


def torch_available() -> bool:
    """Whether a GPU pipeline could be built at all. Import-only: this must not
    initialise CUDA, because it runs during backend selection."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("diffusers") is not None
