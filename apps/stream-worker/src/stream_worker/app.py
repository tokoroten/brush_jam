"""HTTP surface of the stream worker.

Contract (docs/MVP_PLAN.md 6, plus the size fields the worker needs):

    POST /generate
      { image_b64, mask_b64?, prompt, negative_prompt?, strength|denoise?,
        steps?, seed?, width?, height? }
      -> { image_b64, timings: { wait_ms, prompt_ms, diffusion_ms,
                                 composite_ms, total_ms, ... }, width, height }

    GET /healthz -> { ok, backend, model, size, warm }
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import CancelledError, StreamPipeline, decode_png_b64, encode_png_b64, round_size

log = logging.getLogger("stream_worker.app")


class GenerateBody(BaseModel):
    image_b64: str
    mask_b64: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    # `strength` is the diffusers name, `denoise` the ComfyUI/Brush Jam name.
    strength: float | None = None
    denoise: float | None = None
    steps: int | None = None
    seed: int = 0
    width: int | None = None
    height: int | None = None
    # Square convenience field: the server sends `size` in its own contract.
    size: int | None = None
    # Caller-chosen id so an abandoned request can be cancelled. Optional: the
    # worker generates one when the caller does not care.
    request_id: str | None = None
    # Wait for the GPU instead of being refused with 409 while it is busy.
    queue: bool = False

    def resolved_strength(self, default: float = 0.55) -> float:
        value = self.strength if self.strength is not None else self.denoise
        return default if value is None else float(value)


class GenerateResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    request_id: str
    timings: dict[str, float] = Field(default_factory=dict)


class CancelBody(BaseModel):
    request_id: str


def create_app(settings: Settings | None = None, pipeline: Any | None = None) -> FastAPI:
    s = settings or Settings()
    pipe = pipeline if pipeline is not None else (None if s.dry_run else StreamPipeline(s))
    app = FastAPI(title="Brush Jam stream worker", version="0.1.0")
    # One GPU, one generation at a time. The lock (rather than a queue) keeps
    # the worker's behaviour identical to the ComfyUI backend the server already
    # drives: latest-wins is the caller's job, not the worker's.
    gpu_lock = asyncio.Lock()
    state: dict[str, Any] = {"loaded": False, "error": None, "current_request_id": None}
    # Ids asked to stop. A request can be cancelled before it reaches the GPU
    # (it is still queued behind another) or while it is running, so the set is
    # consulted at both points rather than only by the running job.
    cancelled: set[str] = set()
    CANCEL_MEMORY = 256

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if pipe is None:
            log.warning("STREAM_DRY_RUN=1: no model is loaded, /generate echoes its input")
        else:
            loop = asyncio.get_running_loop()

            def _load() -> None:
                pipe.load()

            try:
                # Off the event loop: loading is ~30 s of blocking file + CUDA work.
                await loop.run_in_executor(None, _load)
                state["loaded"] = True
            except Exception as err:  # keep the process up so /healthz can report it
                state["error"] = f"{type(err).__name__}: {err}"
                log.exception("model load failed")
        yield

    app.router.lifespan_context = lifespan

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        warm = bool(pipe is not None and state["loaded"] and pipe.warm)
        return {
            "ok": state["error"] is None,
            "backend": "dry-run" if pipe is None else pipe.backend,
            "model": "none" if pipe is None else pipe.model_name(),
            "size": s.warmup_size,
            "max_size": s.max_size,
            "steps": s.default_steps,
            "guidance": s.guidance,
            "warm": warm,
            "loaded": bool(state["loaded"]),
            "busy": gpu_lock.locked(),
            "current_request_id": state["current_request_id"],
            "error": state["error"],
            "memory": {} if pipe is None or not hasattr(pipe, "memory") else pipe.memory(),
        }

    @app.post("/cancel")
    async def cancel(body: CancelBody) -> dict[str, Any]:
        """Ask a request to stop. Cooperative: a running job stops at its next
        diffusion step, a queued one never starts."""
        running = state["current_request_id"] == body.request_id
        cancelled.add(body.request_id)
        # Bounded: ids accumulate otherwise, and a cancel for an id that never
        # arrives must not pin memory forever.
        while len(cancelled) > CANCEL_MEMORY:
            cancelled.pop()
        return {"ok": True, "request_id": body.request_id, "state": "running" if running else "pending"}

    @app.post("/unload")
    async def unload() -> dict[str, Any]:
        """Drop the model and free the VRAM, without exiting the process.

        On an 8 GB card this worker and ComfyUI cannot both be resident, so the
        two need a way to hand the GPU over. The next /generate reloads (~90 s).
        """
        if pipe is None:
            return {"ok": True, "loaded": False}
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, pipe.unload)
            state["loaded"] = False
            state["error"] = None
        return {"ok": True, "loaded": False}

    @app.post("/load")
    async def load() -> dict[str, Any]:
        """Reload after /unload (or after a failed startup load)."""
        if pipe is None:
            return {"ok": True, "loaded": False}
        async with gpu_lock:
            if state["loaded"]:
                return {"ok": True, "loaded": True}
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, pipe.load)
                state["loaded"] = True
                state["error"] = None
            except Exception as err:
                state["error"] = f"{type(err).__name__}: {err}"
                log.exception("model load failed")
                raise HTTPException(status_code=500, detail=state["error"]) from err
        return {"ok": True, "loaded": True}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(body: GenerateBody) -> GenerateResponse:
        if pipe is not None and state["error"] is not None:
            raise HTTPException(status_code=503, detail=state["error"])
        try:
            image = decode_png_b64(body.image_b64)
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"image_b64 is not a decodable image: {err}") from err
        mask: Image.Image | None = None
        if body.mask_b64:
            try:
                mask = decode_png_b64(body.mask_b64)
            except Exception as err:
                raise HTTPException(status_code=400, detail=f"mask_b64 is not a decodable image: {err}") from err

        try:
            width = round_size(body.width or body.size or image.width, s.max_size)
            height = round_size(body.height or body.size or image.height, s.max_size)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        steps = max(1, min(20, body.steps or s.default_steps))
        strength = min(1.0, max(0.05, body.resolved_strength()))

        request_id = body.request_id or uuid.uuid4().hex

        if pipe is None:
            if request_id in cancelled:
                raise HTTPException(status_code=499, detail=f"cancelled ({request_id})")
            return GenerateResponse(
                image_b64=encode_png_b64(image.convert("RGB").resize((width, height))),
                width=width,
                height=height,
                request_id=request_id,
                timings={"wait_ms": 0.0, "total_ms": 0.0, "dry_run": 1.0},
            )

        # Refuse rather than pile up. asyncio never preempts between this check
        # and the uncontended acquire below, so two callers cannot both pass.
        if gpu_lock.locked() and not body.queue:
            raise HTTPException(
                status_code=409,
                detail=f"busy with {state['current_request_id']}; retry when /healthz reports busy:false, or send queue:true",
                headers={"retry-after": "1"},
            )

        queued_at = time.perf_counter()
        async with gpu_lock:
            wait_ms = (time.perf_counter() - queued_at) * 1000.0
            loop = asyncio.get_running_loop()

            # Reload transparently after an /unload handover.
            if not state["loaded"]:
                try:
                    await loop.run_in_executor(None, pipe.load)
                    state["loaded"] = True
                except Exception as err:
                    state["error"] = f"{type(err).__name__}: {err}"
                    log.exception("model load failed")
                    raise HTTPException(status_code=503, detail=state["error"]) from err

            # Cancelled while it was queued behind another request: never start.
            if request_id in cancelled:
                raise HTTPException(status_code=499, detail=f"cancelled while queued ({request_id})")

            def _run() -> Any:
                return pipe.generate(
                    image=image,
                    mask=mask,
                    prompt=body.prompt,
                    negative_prompt=body.negative_prompt,
                    strength=strength,
                    steps=steps,
                    seed=body.seed,
                    width=width,
                    height=height,
                    should_cancel=lambda: request_id in cancelled,
                    request_id=request_id,
                )

            state["current_request_id"] = request_id
            try:
                result = await loop.run_in_executor(None, _run)
            except CancelledError as err:
                log.info("generation cancelled: %s", request_id)
                raise HTTPException(status_code=499, detail=str(err)) from err
            except Exception as err:
                log.exception("generation failed")
                raise HTTPException(status_code=500, detail=f"{type(err).__name__}: {err}") from err
            finally:
                state["current_request_id"] = None
                cancelled.discard(request_id)

        t = time.perf_counter()
        image_b64 = encode_png_b64(result.image)
        timings = dict(result.timings)
        timings["wait_ms"] = wait_ms
        timings["png_encode_ms"] = (time.perf_counter() - t) * 1000.0
        timings["total_ms"] = timings.get("total_ms", 0.0) + wait_ms + timings["png_encode_ms"]
        if result.image.size != (width, height):
            raise HTTPException(
                status_code=500,
                detail=f"internal error: produced {result.image.size}, expected {(width, height)}",
            )
        return GenerateResponse(
            image_b64=image_b64, width=width, height=height, request_id=request_id, timings=timings
        )

    return app
