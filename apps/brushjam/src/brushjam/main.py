"""uvicorn entry point. Port of the retired Node server's src/index.ts.

`uv run brushjam` serves the built client, REST, the WebSocket and inference on
one port.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .app import HEARTBEAT_MISSES, HEARTBEAT_MS, MAX_WS_PAYLOAD, create_app
from .config import Config, ConfigError, env_origin, load_config, resolve_backend_config
from .room import RoomLimits


#: The allocator setting torch reads once, when it first talks to the driver.
#:
#: Set here rather than in .env because it has to be in the environment before
#: `import torch`, and this module runs before anything imports it. Expandable
#: segments let the allocator grow a segment instead of reserving a new one, so
#: a long-lived process fragments less - which is what an 8 GB card runs out
#: of first. An operator who has set it keeps their value.
CUDA_ALLOC_CONF = "expandable_segments:True"


def _set_allocator_conf(environ: Optional[dict] = None) -> str:
    """Returns the value in force, whoever set it."""
    env = os.environ if environ is None else environ
    existing = env.get("PYTORCH_CUDA_ALLOC_CONF")
    if existing:
        return existing
    env["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    return CUDA_ALLOC_CONF


def _repo_root() -> Path:
    # src/brushjam/main.py -> src -> apps/brushjam -> apps -> repo root
    return Path(__file__).resolve().parents[4]


def _load_dotenv() -> str:
    """Optional repo-root .env. Values are never logged, and it fills gaps only:
    a variable already in the environment keeps its value."""
    before = os.environ.get("AI_BACKEND")
    env_file = _repo_root() / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        except Exception:
            pass  # a malformed .env is not fatal
    return env_origin(before, os.environ.get("AI_BACKEND"))


def _announce(config: Config, backend_name: str, capabilities, backend_origin: str) -> None:
    log = logging.getLogger("brushjam")
    log.info(
        "server on http://%s:%d (set HOST=0.0.0.0 to expose on the LAN)", config.host, config.port
    )
    log.info(
        "AI_BACKEND is not set (auto-detecting; put AI_BACKEND=mock in .env to pin it)"
        if backend_origin == "unset"
        else "AI_BACKEND=%s (from the %s)"
        % (config.ai_backend, "repo-root .env" if backend_origin == ".env" else "environment"),
    )
    log.info("canvas %d / ai mode %s", config.canvas_size, config.ai_mode)
    note = (
        "same as canvas"
        if config.ai_window == config.canvas_size
        else f"resampled from/to {config.canvas_size}"
    )
    log.info("generation resolution %d (%s)", config.ai_window, note)
    log.info(
        "backend %s supports %s up to %s at denoise <= %s",
        backend_name,
        "/".join(capabilities.profiles),
        capabilities.max_resolution,
        capabilities.max_denoise,
    )


async def _build():
    from .ai.backends import create_backend

    backend_origin = _load_dotenv()
    _set_allocator_conf()
    config = load_config()
    log = logging.getLogger("brushjam")
    backend = await create_backend(config, lambda m: log.info("%s", m.replace("[ai] ", "")))
    # The chosen backend gets the last word on the defaults and on what a room
    # may ask for at all.
    capabilities = await backend.capabilities()
    config = resolve_backend_config(config, backend.name, capabilities)
    limits = RoomLimits(
        profiles=capabilities.profiles,
        max_denoise=capabilities.max_denoise,
        max_resolution=config.max_resolution,
        negative_prompt_active=capabilities.negative_prompt_active,
    )
    _announce(config, backend.name, capabilities, backend_origin)
    return config, create_app(config, backend, limits)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
    )
    import uvicorn

    try:
        config, app = asyncio.run(_build())
    except ConfigError as err:
        print(f"[brushjam] {err}", file=sys.stderr)
        return 1

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_config=None,
        access_log=False,
        ws_max_size=MAX_WS_PAYLOAD,
        # A laptop that closes its lid never sends a close frame; without this
        # its avatar sits in the member list for the rest of the session.
        ws_ping_interval=HEARTBEAT_MS / 1000,
        ws_ping_timeout=(HEARTBEAT_MS * HEARTBEAT_MISSES) / 1000,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
