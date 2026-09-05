"""Backend contract. Port of apps/server/src/ai/backends/types.ts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Protocol


@dataclass
class GenerateRequest:
    #: Which workflow to run. Chosen per room and therefore per request.
    profile: str
    prompt: str
    negative_prompt: str
    #: size x size PNG, opaque.
    image_png: bytes
    #: size x size PNG, white = regenerate.
    mask_png: bytes
    size: int
    denoise: float
    steps: int
    seed: int
    #: Free-form tag used for upload filenames / logging.
    tag: str


@dataclass
class BackendCapabilities:
    """What a backend can actually do. Rooms are clamped to this and the UI
    hides what is unavailable."""

    profiles: List[str]
    #: Largest square the backend will generate.
    max_resolution: int
    max_denoise: float
    #: Whether the negative prompt reaches the sampler at all, per profile.
    negative_prompt_active: Dict[str, bool] = field(
        default_factory=lambda: {"fast": True, "quality": True}
    )


class AIBackend(Protocol):
    name: str

    async def capabilities(self) -> BackendCapabilities:
        ...

    async def generate(self, req: GenerateRequest) -> bytes:
        ...


class BackendHttpError(Exception):
    """The backend answered with an HTTP status. Carrying it beats re-deriving
    it from the message: "size 512 out of range" contains something that looks
    like a 5xx."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class AbortedError(Exception):
    def __init__(self) -> None:
        super().__init__("generation aborted")


async def delay(ms: float) -> None:
    await asyncio.sleep(ms / 1000)
