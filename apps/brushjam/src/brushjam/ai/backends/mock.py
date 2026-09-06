"""GPU-free stand-in. Port of the retired Node server's src/ai/backends/mock.ts.

A deterministic stylisation (posterise + a prompt-derived hue shift + edge
darkening) applied inside the mask. Same contract as the real backends, so the
whole scheduler path is exercised without a GPU.
"""

from __future__ import annotations

import io
from typing import Dict

import numpy as np
from PIL import Image

from ...constants import MAX_AI_RESOLUTION, MAX_DENOISE
from ...raster import to_png
from .base import BackendCapabilities, GenerateRequest, delay

_MASK32 = 0xFFFFFFFF


def _hash(s: str) -> int:
    h = 2166136261
    for ch in s:
        h = (h ^ (ord(ch) & _MASK32)) & _MASK32
        h = (h * 16777619) & _MASK32
    return h & _MASK32


class MockBackend:
    name = "mock"

    def __init__(self, latency_ms: float = 800) -> None:
        self.latency_ms = latency_ms

    async def capabilities(self) -> BackendCapabilities:
        # The mock stylises rather than samples, so nothing here is a real limit.
        return BackendCapabilities(
            profiles=["fast", "quality"],
            max_resolution=MAX_AI_RESOLUTION,
            max_denoise=MAX_DENOISE,
            negative_prompt_active={"fast": True, "quality": True},
        )

    def identity(self, profile: str) -> Dict[str, str]:
        # Not a checkpoint at all, and saying so is the point: a history entry
        # written against the mock must not read as one made by a model.
        return {"model": "mock"}

    async def generate(self, req: GenerateRequest) -> bytes:
        await delay(self.latency_ms)
        size = req.size
        with Image.open(io.BytesIO(req.image_png)) as raw:
            raw.load()
            image = raw.convert("RGB").resize((size, size), Image.BILINEAR)
        with Image.open(io.BytesIO(req.mask_png)) as raw:
            raw.load()
            mask = raw.convert("L").resize((size, size), Image.BILINEAR)

        original = np.asarray(image, dtype=np.float32)
        m = (np.asarray(mask, dtype=np.float32) / 255.0)[:, :, None]

        h = _hash(req.prompt + str(req.seed))
        shift = np.array([h % 96, (h >> 8) % 96, (h >> 16) % 96], dtype=np.float32)
        levels = 3 + (h % 3)

        left = np.concatenate([original[:, :1, :1], original[:, :-1, :1]], axis=1)
        up = np.concatenate([original[:1, :, :1], original[:-1, :, :1]], axis=0)
        edge = np.minimum(60.0, np.abs(original[:, :, :1] - left) + np.abs(original[:, :, :1] - up))

        posterised = np.round(np.round(original / 255.0 * (levels - 1)) / (levels - 1) * 255.0)
        styled = np.clip(posterised + shift - 48.0 - edge, 0, 255)
        out = np.where(m > 0, original + (styled - original) * m, original)
        return to_png(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"))
