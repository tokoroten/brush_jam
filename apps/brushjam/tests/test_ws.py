"""End-to-end over the real ASGI app: two clients in one room, HTTP routes.

This is the closest thing to the browser the test suite has - every frame goes
through the same validation, reducer and broadcast path a real socket does.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image
from starlette.testclient import TestClient

from brushjam.ai.backends.mock import MockBackend
from brushjam.app import create_app
from brushjam.config import load_config
from brushjam.raster import to_png


def make_client(**env) -> TestClient:
    settings = {
        "AI_BACKEND": "mock",
        "CANVAS_SIZE": "512",
        "AI_WINDOW": "512",
        "AI_DEBOUNCE_MS": "10",
        # See test_history.py: the suite does not write to the checkout.
        "HISTORY_ENABLED": "0",
    }
    settings.update(env)
    config = load_config(settings)
    # No web/dist in the test environment; the SPA route is exercised separately.
    return TestClient(create_app(config, MockBackend(latency_ms=0)))


def drain(socket, want: str, timeout: float = 5.0, where=None) -> Dict[str, Any]:
    """Read frames until one of type `want` (and matching `where`) arrives."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = json.loads(socket.receive_text())
        if msg["t"] == want and (where is None or where(msg)):
            return msg
    raise AssertionError(f"no {want} within {timeout}s")


def drain_all(socket, want: str, timeout: float = 2.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = json.loads(socket.receive_text())
        except Exception:
            break
        out.append(msg)
        if msg["t"] == want:
            break
    return out


def test_healthz_and_room_creation() -> None:
    with make_client() as client:
        health = client.get("/healthz").json()
        assert health["ok"] is True and health["backend"] == "mock"
        room_id = client.post("/api/rooms").json()["roomId"]
        assert 4 <= len(room_id) <= 16


def test_two_clients_see_each_others_strokes_and_presence() -> None:
    with make_client() as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=Alice") as alice:
            snap = drain(alice, "snapshot")["snapshot"]
            layer = snap["layers"][0]["id"]
            assert snap["canvasSize"] == 512
            assert snap["members"][0]["name"] == "Alice"

            with client.websocket_connect(f"/ws/rooms/{room_id}?name=Bob") as bob:
                bob_snap = drain(bob, "snapshot")["snapshot"]
                assert [m["name"] for m in bob_snap["members"]] == ["Alice", "Bob"]
                # The first presence Alice sees is her own join; wait for Bob's.
                presence = drain(alice, "presence", where=lambda m: len(m["members"]) == 2)
                assert [m["name"] for m in presence["members"]] == ["Alice", "Bob"]

                alice.send_text(
                    json.dumps(
                        {
                            "t": "stroke_start",
                            "stroke": {
                                "id": "a1",
                                "layerId": layer,
                                "tool": "pen",
                                "color": "#ff0000",
                                "width": 10,
                                "points": [{"x": 10, "y": 10}],
                            },
                        }
                    )
                )
                assert drain(bob, "stroke_start")["stroke"]["id"].endswith(":a1")
                alice.send_text(
                    json.dumps({"t": "stroke_end", "strokeId": "a1", "points": [{"x": 80, "y": 90}]})
                )
                committed = drain(bob, "stroke_committed")
                assert committed["humanRevision"] == 1
                assert committed["stroke"]["userId"] == snap["youUserId"]

                # Bob's undo takes Bob's own stroke - he has none.
                bob.send_text(json.dumps({"t": "undo"}))
                # Alice's undo takes hers.
                alice.send_text(json.dumps({"t": "undo"}))
                undone = drain(bob, "undo_applied")
                assert undone["strokeId"] == committed["stroke"]["id"]
                assert undone["humanRevision"] == 2

            # Bob left: Alice sees presence shrink again.
            presence = drain(alice, "presence", where=lambda m: len(m["members"]) == 1)
            assert presence["members"][0]["name"] == "Alice"


def test_a_reconnect_with_a_token_keeps_the_identity_and_the_undo_stack() -> None:
    with make_client() as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        token = "tok-alice-0001"
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=Alice&token={token}") as alice:
            snap = drain(alice, "snapshot")["snapshot"]
            user_id = snap["youUserId"]
            layer = snap["layers"][0]["id"]
            alice.send_text(
                json.dumps(
                    {
                        "t": "stroke_start",
                        "stroke": {
                            "id": "a1",
                            "layerId": layer,
                            "tool": "pen",
                            "color": "#ff0000",
                            "width": 10,
                            "points": [{"x": 10, "y": 10}],
                        },
                    }
                )
            )
            alice.send_text(json.dumps({"t": "stroke_end", "strokeId": "a1", "points": []}))
            drain(alice, "stroke_committed")
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=Alice&token={token}") as again:
            snap = drain(again, "snapshot")["snapshot"]
            assert snap["youUserId"] == user_id
            assert len(snap["strokes"]) == 1
            again.send_text(json.dumps({"t": "undo"}))
            assert drain(again, "undo_applied")["strokeId"] == snap["strokes"][0]["id"]


def test_an_invalid_frame_answers_with_an_error_and_keeps_the_socket() -> None:
    with make_client() as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=A") as socket:
            drain(socket, "snapshot")
            socket.send_text("not json")
            assert drain(socket, "error")["message"] == "invalid json"
            socket.send_text(json.dumps({"t": "nonsense"}))
            assert drain(socket, "error")["message"] == "unknown message type: nonsense"
            socket.send_text(json.dumps({"t": "set_prompt", "prompt": "still alive"}))
            assert drain(socket, "prompt_changed")["prompt"] == "still alive"


def test_a_stroke_produces_an_ai_result_that_can_be_fetched() -> None:
    with make_client() as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=A") as socket:
            snap = drain(socket, "snapshot")["snapshot"]
            layer = snap["layers"][0]["id"]
            socket.send_text(
                json.dumps(
                    {
                        "t": "stroke_start",
                        "stroke": {
                            "id": "a1",
                            "layerId": layer,
                            "tool": "pen",
                            "color": "#000000",
                            "width": 20,
                            "points": [{"x": 50, "y": 50}],
                        },
                    }
                )
            )
            socket.send_text(
                json.dumps({"t": "stroke_end", "strokeId": "a1", "points": [{"x": 400, "y": 400}]})
            )
            result = drain(socket, "ai_result", timeout=20)
            assert result["aiGeneration"] == 1
            assert result["crop"] == {"x": 0, "y": 0, "width": 512, "height": 512}
            assert result["profile"] == "fast"
        patch = client.get(result["url"])
        assert patch.status_code == 200 and patch.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(patch.content)) as img:
            assert img.size == (512, 512)
        full = client.get(f"/rooms/{room_id}/ai.png")
        assert full.status_code == 200


def test_image_upload_guards() -> None:
    with make_client() as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        png = to_png(Image.new("RGB", (64, 48), (1, 2, 3)))
        ok = client.post(f"/rooms/{room_id}/images", content=png, headers={"content-type": "image/png"})
        assert ok.status_code == 200
        body = ok.json()
        assert (body["width"], body["height"]) == (64, 48)
        fetched = client.get(f"/rooms/{room_id}/images/{body['imageId']}")
        assert fetched.status_code == 200 and fetched.content == png

        bad_type = client.post(
            f"/rooms/{room_id}/images", content=png, headers={"content-type": "image/gif"}
        )
        assert bad_type.status_code == 415
        mismatched = client.post(
            f"/rooms/{room_id}/images", content=png, headers={"content-type": "image/jpeg"}
        )
        assert mismatched.status_code == 400
        truncated = client.post(
            f"/rooms/{room_id}/images", content=png[:24], headers={"content-type": "image/png"}
        )
        assert truncated.status_code == 400


def test_a_reference_layer_can_be_created_from_an_upload() -> None:
    with make_client() as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        png = to_png(Image.new("RGB", (32, 32), (200, 30, 30)))
        image_id = client.post(
            f"/rooms/{room_id}/images", content=png, headers={"content-type": "image/png"}
        ).json()["imageId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=A") as socket:
            drain(socket, "snapshot")
            socket.send_text(
                json.dumps(
                    {"t": "layer_create", "layer": {"kind": "reference", "imageId": image_id, "x": 4, "y": 4}}
                )
            )
            layer = drain(socket, "layer_created")["layer"]
            assert layer["kind"] == "reference"
            assert layer["includeInAI"] is False
            assert (layer["imageWidth"], layer["imageHeight"]) == (32, 32)


def test_unknown_room_assets_are_404() -> None:
    with make_client() as client:
        assert client.get("/rooms/abcd1234/ai.png").status_code == 404
        assert client.get("/rooms/abcd1234/patches/none.png").status_code == 404
        assert client.get("/rooms/abcd1234/images/none").status_code == 404


def test_healthz_reports_a_resident_backends_sampling_fields() -> None:
    """The TS tooling reads {ok, backend, rooms} and, for a resident model, the
    same steps/guidance/vae/model/lora it used to fetch from the worker."""
    from brushjam.ai.backends.inproc import InprocBackend
    from brushjam.ai.pipeline import PipelineSettings

    config = load_config({"AI_BACKEND": "inproc", "INPROC_DRY_RUN": "1", "CANVAS_SIZE": "512"})
    with TestClient(create_app(config, InprocBackend(PipelineSettings(), dry_run=True))) as client:
        body = client.get("/healthz").json()
    assert body["ok"] is True and body["backend"] == "inproc" and body["rooms"] == 0
    for key in ("model", "steps", "guidance", "vae", "lora", "max_size", "max_denoise", "warm"):
        assert key in body, key
