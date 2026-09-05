from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> Any:
    path = FIXTURES / name
    if not path.exists():  # pragma: no cover - guidance, not behaviour
        pytest.skip(
            f"{path.name} is missing; run `pnpm --filter @brushjam/server export-fixtures`"
        )
    return json.loads(path.read_text(encoding="utf-8"))
