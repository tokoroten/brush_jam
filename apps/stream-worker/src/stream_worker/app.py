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
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import StreamPipeline, decode_png_b64, encode_png_b64, round_size

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

    def resolved_strength(self, default: float = 0.55) -> float:
        value = self.strength if self.strength is not None else self.denoise
        return default if value is None else float(value)


class GenerateResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    timings: dict[str, float] = Field(default_factory=dict)


def create_app(settings: Settings | None = None, pipeline: Any | None = None) -> FastAPI:
    s = settings or Settings()
    pipe = pipeline if pipeline is not None else (None if s.dry_run else StreamPipeline(s))
    app = FastAPI(title="Brush Jam stream worker", version="0.1.0")
    # One GPU, one generation at a time. The lock (rather than a queue) keeps
    # the worker's behaviour identical to the ComfyUI backend the server already
    # drives: latest-wins is the caller's job, not the worker's.
    gpu_lock = asyncio.Lock()
    state: dict[str, Any] = {"loaded": False, "error": None}

    @app.on_event("startup")
    async def _startup() -> None:
        if pipe is None:
            log.warning("STREAM_DRY_RUN=1: no model is loaded, /generate echoes its input")
            return
        loop = asyncio.get_running_loop()

        def _load() -> None:
            pipe.load()

        try:
            await loop.run_in_executor(None, _load)
            state["loaded"] = True
        except Exception as err:  # keep the process up so /healthz can report it
            state["error"] = f"{type(err).__name__}: {err}"
            log.exception("model load failed")

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
            "busy": gpu_lock.locked(),
            "error": state["error"],
        }

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

        if pipe is None:
            return GenerateResponse(
                image_b64=encode_png_b64(image.convert("RGB").resize((width, height))),
                width=width,
                height=height,
                timings={"wait_ms": 0.0, "total_ms": 0.0, "dry_run": 1.0},
            )

        queued_at = time.perf_counter()
        async with gpu_lock:
            wait_ms = (time.perf_counter() - queued_at) * 1000.0
            loop = asyncio.get_running_loop()

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
                )

            try:
                result = await loop.run_in_executor(None, _run)
            except Exception as err:
                log.exception("generation failed")
                raise HTTPException(status_code=500, detail=f"{type(err).__name__}: {err}") from err

        t = time.perf_counter()
        image_b64 = encode_png_b64(result.image)
        timings = dict(result.timings)
        timings["wait_ms"] = wait_ms
        timings["encode_ms"] = (time.perf_counter() - t) * 1000.0
        timings["total_ms"] = timings.get("total_ms", 0.0) + wait_ms + timings["encode_ms"]
        return GenerateResponse(image_b64=image_b64, width=width, height=height, timings=timings)

    return app
