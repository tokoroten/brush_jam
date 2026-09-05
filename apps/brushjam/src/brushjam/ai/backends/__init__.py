"""Backend selection. Port of apps/server/src/ai/backends/index.ts.

Only the mock backend exists until the inference pipeline moves in (plan M2);
`create_backend` already has the shape the rest will slot into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .base import (
    AbortedError,
    AIBackend,
    BackendCapabilities,
    BackendHttpError,
    GenerateRequest,
    delay,
)
from .mock import MockBackend

if TYPE_CHECKING:  # pragma: no cover
    from ...config import Config

__all__ = [
    "AIBackend",
    "AbortedError",
    "BackendCapabilities",
    "BackendHttpError",
    "GenerateRequest",
    "MockBackend",
    "create_backend",
    "delay",
]

#: Bounded so a dead endpoint cannot delay startup.
PROBE_TIMEOUT_MS = 2000


async def create_backend(config: "Config", log: Callable[[str], None] = print) -> AIBackend:
    if config.ai_backend == "mock":
        log("[ai] backend: mock (AI_BACKEND=mock)")
        return MockBackend()
    # Every other backend needs the GPU pipeline or a network probe, neither of
    # which exists yet in this port.
    log(
        f"[ai] backend: mock - no other backend is ported yet "
        f"(AI_BACKEND={config.ai_backend})"
    )
    return MockBackend()
