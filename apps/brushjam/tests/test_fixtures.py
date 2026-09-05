"""Cross-language parity: replay what the Node implementation actually did.

The fixtures are written by `apps/server/scripts/export-fixtures.ts` from the
real Node modules, so a difference here is a difference between the two
servers, not between two descriptions of them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pytest

from brushjam.noise import fnv1a, noise_rgb
from brushjam.room import (
    RoomLimits,
    apply_client_message,
    create_room,
    join_member,
    snapshot,
    sorted_layers,
)
from brushjam.validate import validate_client_message

from conftest import load_fixture


# ------------------------------------------------------------------ protocol


def _protocol_cases():
    data = load_fixture("protocol-samples.json")
    return [(s["type"], s["input"], s["result"]) for s in data["samples"]]


@pytest.mark.parametrize("kind,payload,expected", _protocol_cases())
def test_validator_matches_node(kind: str, payload: Any, expected: Dict[str, Any]) -> None:
    result = validate_client_message(payload)
    if expected["ok"]:
        assert result.ok, f"{kind}: expected accept, got {result.error}"
        assert result.msg == expected["msg"]
    else:
        assert not result.ok, f"{kind}: expected reject, got {result.msg}"
        assert result.error == expected["error"]


# --------------------------------------------------------------------- noise


def test_fnv1a_matches_node() -> None:
    data = load_fixture("noise-samples.json")
    for case in data["fnv1a"]:
        assert fnv1a(case["text"]) == case["seed"], case["text"]


def test_noise_rgb_matches_node() -> None:
    data = load_fixture("noise-samples.json")
    for case in data["rgb"]:
        assert list(noise_rgb(case["seed"], case["x"], case["y"])) == case["rgb"], case


# ------------------------------------------------------------------- reducer


def _normalise(value: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for real, placeholder in mapping.items():
            out = out.replace(real, placeholder)
        return out
    if isinstance(value, list):
        return [_normalise(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v, mapping) for k, v in value.items()}
    return value


def _resolve(msg: Any, created: List[str]) -> Any:
    """`@layerN` in the script means "the Nth layer ever created in this room"."""
    text = json.dumps(msg)
    text = re.sub(
        r'"@layer(\d+)"',
        lambda m: json.dumps(created[int(m.group(1))] if int(m.group(1)) < len(created) else "missing"),
        text,
    )
    return json.loads(text)


def test_reducer_trace_matches_node() -> None:
    data = load_fixture("reducer-trace.json")
    spec = data["room"]
    room = create_room(
        spec["id"],
        spec["denoise"],
        spec["canvasSize"],
        spec["resolution"],
        spec["adjustable"],
        spec["profile"],
        RoomLimits(
            profiles=["fast", "quality"],
            max_denoise=0.95,
            max_resolution=1024,
            negative_prompt_active={"fast": True, "quality": True},
        ),
    )
    users = [
        join_member(room, "Alice", "tok-alice-0001")["userId"],
        join_member(room, "Bob", "tok-bob-0001")["userId"],
    ]
    created: List[str] = [l["id"] for l in sorted_layers(room)]

    for index, step in enumerate(data["steps"]):
        for layer in sorted_layers(room):
            if layer["id"] not in created:
                created.append(layer["id"])
        resolved = _resolve(step["msg"], created)
        # The fixture keeps the placeholders, so resolve them the same way.
        validated = validate_client_message(resolved)
        mapping = {users[i]: f"U{i}" for i in range(len(users))}
        mapping.update({layer_id: f"L{i}" for i, layer_id in enumerate(created)})

        if "rejected" in step:
            assert not validated.ok, f"step {index}: expected a rejection"
            assert validated.error == step["rejected"]
            continue

        assert validated.ok, f"step {index}: {validated.error}"
        result = apply_client_message(room, users[step["by"]], validated.msg)
        after = snapshot(
            room, users[step["by"]], "idle", {"window": 768, "apply": 1024, "canvasSize": 1024}
        )
        for layer in sorted_layers(room):
            if layer["id"] not in created:
                created.append(layer["id"])
        mapping = {users[i]: f"U{i}" for i in range(len(users))}
        mapping.update({layer_id: f"L{i}" for i, layer_id in enumerate(created)})

        actual = _normalise(
            {
                "broadcast": result.broadcast,
                "relay": result.relay,
                "toSender": result.to_sender,
                "dirty": result.dirty,
                "promptChanged": result.prompt_changed,
            },
            mapping,
        )
        assert json.loads(json.dumps(actual)) == step["result"], f"step {index} ({step['msg']})"
        assert json.loads(json.dumps(_normalise(after, mapping))) == step["snapshot"], (
            f"step {index} snapshot ({step['msg']})"
        )
