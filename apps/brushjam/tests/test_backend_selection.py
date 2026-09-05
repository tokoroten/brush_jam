"""Which backend a configuration actually gets, and why.

Mirrors the retired Node server's test/backend-selection.test.ts. Selecting a backend must
never load a model or touch the GPU - only an explicit start does that - so
every case here runs with `INPROC_DRY_RUN` or with the checkpoint missing.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

import brushjam.ai.backends as backends
from brushjam.ai.backends import create_backend
from brushjam.config import load_config


def config(**env):
    settings = {"CANVAS_SIZE": "1024", "AI_WINDOW": "768"}
    settings.update(env)
    return load_config(settings)


class Log:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, message: str) -> None:
        self.lines.append(message)

    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def no_probes(monkeypatch):
    """Nothing on the network answers, unless a test says otherwise."""

    async def unreachable(url: str, timeout_ms: float = 1500) -> bool:
        return False

    async def dead(url: str, timeout_ms: float = 1500):
        return backends.StreamHealth(reason="no answer")

    monkeypatch.setattr(backends, "comfy_reachable", unreachable)
    monkeypatch.setattr(backends, "stream_health", dead)
    # A missing checkpoint is the normal state of a test machine anyway, but be
    # explicit: nothing here may decide to load a model.
    monkeypatch.setattr(backends, "torch_available", lambda: False)


async def test_explicit_mock_wins(no_probes) -> None:
    log = Log()
    backend = await create_backend(config(AI_BACKEND="mock"), log)
    assert backend.name == "mock"
    assert "AI_BACKEND=mock" in log.text()


async def test_auto_falls_back_to_the_mock_and_says_why(no_probes) -> None:
    log = Log()
    backend = await create_backend(config(), log)
    assert backend.name == "mock"
    assert "skipping the in-process model" in log.text()
    assert "did not answer" in log.text()


async def test_auto_prefers_comfyui_when_it_answers(no_probes, monkeypatch) -> None:
    async def reachable(url: str, timeout_ms: float = 1500) -> bool:
        return True

    monkeypatch.setattr(backends, "comfy_reachable", reachable)
    log = Log()
    backend = await create_backend(config(), log)
    assert backend.name == "comfyui"
    assert "stream is explicit-only" in log.text()


async def test_the_stream_worker_is_explicit_only(no_probes, monkeypatch) -> None:
    async def warm(url: str, timeout_ms: float = 1500):
        return backends.StreamHealth(ok=True, warm=True, max_size=1024)

    monkeypatch.setattr(backends, "stream_health", warm)
    # A warm worker is ignored by `auto`: it answering only means it is holding
    # ~5 GB of VRAM, which is a deployment decision.
    assert (await create_backend(config(), Log())).name == "mock"
    # ...unless it is opted in.
    opted = await create_backend(config(AI_STREAM_AUTO="1"), Log())
    assert opted.name == "stream"


@pytest.mark.parametrize(
    "health,fragment",
    [
        (dict(ok=True, warm=False, max_size=1024), "not warm"),
        (dict(ok=True, warm=True, max_size=512), "max_size 512 < AI_WINDOW 768"),
    ],
)
async def test_an_unusable_worker_is_skipped_with_its_reason(
    no_probes, monkeypatch, health: Dict[str, Any], fragment: str
) -> None:
    async def probe(url: str, timeout_ms: float = 1500):
        return backends.StreamHealth(**health)

    monkeypatch.setattr(backends, "stream_health", probe)
    log = Log()
    backend = await create_backend(config(AI_STREAM_AUTO="1"), log)
    assert backend.name == "mock"
    assert fragment in log.text()


async def test_an_explicit_stream_worker_is_honoured_with_a_warning(no_probes) -> None:
    log = Log()
    backend = await create_backend(config(AI_BACKEND="stream"), log)
    assert backend.name == "stream"
    assert "is not answering" in log.text()


async def test_inproc_is_auto_selected_when_it_can_run(no_probes, monkeypatch) -> None:
    log = Log()
    # Dry run stands in for "torch is importable and the checkpoint is there".
    backend = await create_backend(config(INPROC_DRY_RUN="1"), log)
    assert backend.name == "inproc"
    assert "auto-detected" in log.text()


async def test_explicit_inproc_refuses_rather_than_silently_becoming_the_mock(
    no_probes, monkeypatch
) -> None:
    with pytest.raises(RuntimeError, match="torch/diffusers are not installed"):
        await create_backend(config(AI_BACKEND="inproc"), Log())
    monkeypatch.setattr(backends, "torch_available", lambda: True)
    with pytest.raises(RuntimeError, match="INPROC_CHECKPOINT points at a file"):
        await create_backend(
            config(AI_BACKEND="inproc", INPROC_CHECKPOINT="Z:/nope/model.safetensors"), Log()
        )
    # And with nothing configured at all, the message has to say what to set.
    with pytest.raises(RuntimeError, match="INPROC_CHECKPOINT"):
        await create_backend(config(AI_BACKEND="inproc"), Log())


async def test_runpod_needs_its_credentials(no_probes) -> None:
    backend = await create_backend(
        config(AI_BACKEND="runpod", RUNPOD_ENDPOINT_ID="ep1", RUNPOD_API_KEY="key"), Log()
    )
    assert backend.name == "runpod"
