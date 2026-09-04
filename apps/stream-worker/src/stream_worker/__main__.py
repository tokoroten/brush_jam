"""Entry point: ``uv run stream-worker`` / ``uv run python -m stream_worker``."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _load_repo_env() -> None:
    """Pick up HF_TOKEN from the repo-root .env without printing it."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[4] / ".env"
    if root.exists():
        load_dotenv(root, override=False)


def main() -> None:
    logging.basicConfig(level=os.environ.get("STREAM_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _load_repo_env()

    import uvicorn

    from .app import create_app
    from .config import Settings

    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
