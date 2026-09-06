"""The saved-results store and the two routes that read it.

The store is the only part of the server that writes to disk, so most of this
is about what happens when the disk says no: a full one, a read-only one, a
room id that is really a path. None of it may cost a generation.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image
from starlette.testclient import TestClient

from brushjam.ai.backends.mock import MockBackend
from brushjam.app import create_app
from brushjam.config import load_config
import brushjam.history as _history
from brushjam.history import HistoryStore

JPEG = b"\xff\xd8\xff" + b"x" * 4000
INPUT_JPEG = b"\xff\xd8\xff" + b"i" * 1500


def store(tmp_path: Path, **kw) -> HistoryStore:
    return HistoryStore(tmp_path / "history", **kw)


def entry(**kw) -> Dict[str, Any]:
    base = {
        "aiRevision": 3,
        "aiGeneration": 1,
        "time": int(time.time() * 1000),
        "prompt": "a hill",
        "negativePrompt": "",
        "denoise": 0.8,
        "seed": 42,
        "profile": "fast",
        "aiResolution": 768,
        "latencyMs": 1800,
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------- storing


def test_record_writes_the_image_and_the_entry(tmp_path: Path) -> None:
    s = store(tmp_path)
    n = s.record("abcd", JPEG, entry())
    assert n == 0
    assert (tmp_path / "history" / "abcd" / "0.jpg").read_bytes() == JPEG
    saved = json.loads((tmp_path / "history" / "abcd" / "0.json").read_text(encoding="utf-8"))
    assert saved["n"] == 0 and saved["prompt"] == "a hill" and saved["seed"] == 42
    assert s.record("abcd", JPEG, entry()) == 1


def test_listing_is_newest_first_and_carries_a_url(tmp_path: Path) -> None:
    s = store(tmp_path)
    for i in range(3):
        s.record("abcd", JPEG, entry(prompt=f"p{i}"))
    listed = s.list("abcd")
    assert [e["n"] for e in listed] == [2, 1, 0]
    assert listed[0]["prompt"] == "p2"
    assert listed[0]["url"] == "/rooms/abcd/history/2.jpg"
    assert s.read("abcd", 2) == JPEG
    assert s.read("abcd", 99) is None


def test_a_disabled_store_writes_nothing(tmp_path: Path) -> None:
    s = store(tmp_path, enabled=False)
    assert s.record("abcd", JPEG, entry()) is None
    assert s.list("abcd") == []
    assert s.read("abcd", 0) is None
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize("bad", ["../etc", "ab", "AbCd", "a" * 17, "a/b", "", "abc.d"])
def test_path_unsafe_ids_are_refused(tmp_path: Path, bad: str) -> None:
    s = store(tmp_path)
    assert s.record(bad, JPEG, entry()) is None
    assert s.list(bad) == []
    assert s.read(bad, 0) is None
    # and nothing was created outside the room directories
    assert not any(p.name.startswith("..") for p in (tmp_path / "history").glob("*")) if (
        tmp_path / "history"
    ).exists() else True


def test_a_write_failure_is_survivable(tmp_path: Path, monkeypatch) -> None:
    s = store(tmp_path)
    assert s.record("abcd", JPEG, entry()) == 0

    def boom(*_args, **_kw):
        raise OSError("no space left on device")

    monkeypatch.setattr("brushjam.history._write_atomic", boom)
    assert s.record("abcd", JPEG, entry()) is None
    # The store is still usable once the disk comes back.
    monkeypatch.undo()
    assert s.record("abcd", JPEG, entry()) is not None


# ------------------------------------------------------- the input canvas


def test_the_input_canvas_is_stored_beside_the_result(tmp_path: Path) -> None:
    s = store(tmp_path)
    assert s.record("abcd", JPEG, entry(), INPUT_JPEG) == 0
    room = tmp_path / "history" / "abcd"
    assert (room / "0.in.jpg").read_bytes() == INPUT_JPEG
    assert s.read("abcd", 0, "in") == INPUT_JPEG
    assert s.read("abcd", 0) == JPEG
    assert s.read("abcd", 0, "json") is None and s.read("abcd", 0, "nope") is None
    listed = s.list("abcd")[0]
    assert listed["inputUrl"] == "/rooms/abcd/history/0.in.jpg"
    # All three files are charged to the entry.
    assert s.room_bytes_used("abcd") == (
        len(JPEG) + len(INPUT_JPEG) + (room / "0.json").stat().st_size
    )


def test_an_entry_without_an_input_is_still_an_entry(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.record("abcd", JPEG, entry())
    listed = s.list("abcd")[0]
    assert "inputUrl" not in listed
    assert s.read("abcd", 0, "in") is None
    assert s.count("abcd") == 1


def test_eviction_removes_the_input_too(tmp_path: Path) -> None:
    s = store(tmp_path, room_bytes=12_000, total_bytes=10_000_000)
    for _ in range(4):
        s.record("abcd", JPEG, entry(), INPUT_JPEG)
    room = tmp_path / "history" / "abcd"
    assert [e["n"] for e in s.list("abcd")] == [3, 2]
    for gone in (0, 1):
        for name in (f"{gone}.jpg", f"{gone}.json", f"{gone}.in.jpg"):
            assert not (room / name).exists(), name
    assert s.room_bytes_used("abcd") <= 12_000


def test_a_reloaded_store_counts_and_serves_the_input(tmp_path: Path) -> None:
    first = store(tmp_path)
    first.record("abcd", JPEG, entry(), INPUT_JPEG)
    first.record("abcd", JPEG, entry())  # an old-style entry, no input
    used = first.total_bytes_used

    second = store(tmp_path)
    assert [e["n"] for e in second.list("abcd")] == [1, 0]
    assert second.total_bytes_used == used
    assert second.read("abcd", 0, "in") == INPUT_JPEG
    assert "inputUrl" not in second.list("abcd")[0]


def test_an_input_with_no_pair_is_cleaned_up_at_load(tmp_path: Path) -> None:
    room = tmp_path / "history" / "abcd"
    room.mkdir(parents=True)
    (room / "7.in.jpg").write_bytes(INPUT_JPEG)
    s = store(tmp_path)
    assert s.count("abcd") == 0
    assert not (room / "7.in.jpg").exists()
    # 7 was still handed out once, so it is never handed out again.
    assert s.record("abcd", JPEG, entry()) == 8


# ------------------------------------------------------------------- budgets


def test_the_room_budget_evicts_oldest_first(tmp_path: Path) -> None:
    # Four entries of ~4 KB each into a 10 KB room budget.
    s = store(tmp_path, room_bytes=10_000, total_bytes=10_000_000)
    for _ in range(4):
        s.record("abcd", JPEG, entry())
    numbers = [e["n"] for e in s.list("abcd")]
    assert numbers == [3, 2], numbers
    assert not (tmp_path / "history" / "abcd" / "0.jpg").exists()
    assert not (tmp_path / "history" / "abcd" / "0.json").exists()
    assert s.room_bytes_used("abcd") <= 10_000


def test_the_total_budget_evicts_across_rooms(tmp_path: Path) -> None:
    s = store(tmp_path, room_bytes=10_000_000, total_bytes=10_000)
    s.record("aaaa", JPEG, entry())
    s.record("bbbb", JPEG, entry())
    s.record("cccc", JPEG, entry())
    # The oldest room's entry went, not the newest room's.
    assert s.list("aaaa") == []
    assert [e["n"] for e in s.list("cccc")] == [0]
    assert s.total_bytes_used <= 10_000


def test_a_result_larger_than_the_budget_is_still_served(tmp_path: Path) -> None:
    """Never store a result and then immediately delete it: the client has
    already been told its number."""
    s = store(tmp_path, room_bytes=100, total_bytes=100)
    n = s.record("abcd", JPEG, entry())
    assert n == 0 and s.read("abcd", 0) == JPEG


# ------------------------------------------------------------------ restarts


def test_a_new_store_adopts_what_is_on_disk(tmp_path: Path) -> None:
    first = store(tmp_path)
    first.record("abcd", JPEG, entry())
    first.record("abcd", JPEG, entry())

    second = store(tmp_path)
    assert [e["n"] for e in second.list("abcd")] == [1, 0]
    # Numbering continues rather than overwriting entry 0.
    assert second.record("abcd", JPEG, entry()) == 2
    assert second.total_bytes_used > 0
    # ... and the budgets count what was already there.
    third = store(tmp_path, total_bytes=5_000)
    third.record("abcd", JPEG, entry())
    assert third.total_bytes_used <= 5_000


def test_stray_files_are_ignored(tmp_path: Path) -> None:
    room = tmp_path / "history" / "abcd"
    room.mkdir(parents=True)
    (room / "notes.txt").write_text("hello", encoding="utf-8")
    (room / "7.jpg").write_bytes(JPEG)
    (tmp_path / "history" / "NOT-A-ROOM").mkdir()
    s = store(tmp_path)
    # A JPEG with no JSON is half of an interrupted write, not an entry: it is
    # cleaned up rather than left as bytes no budget knows about.
    assert s.count("abcd") == 0
    assert not (room / "7.jpg").exists()
    assert (room / "notes.txt").exists()  # somebody else's file is not ours
    # The number is still spent: 7 was handed out once, so it never is again.
    assert s.record("abcd", JPEG, entry()) == 8
    assert [e["n"] for e in s.list("abcd")] == [8]


def test_concurrent_records_do_not_share_a_number(tmp_path: Path) -> None:
    s = store(tmp_path)

    async def run() -> List[int]:
        return list(
            await asyncio.gather(
                *[asyncio.to_thread(s.record, "abcd", JPEG, entry()) for _ in range(16)]
            )
        )

    numbers = asyncio.run(run())
    assert sorted(numbers) == list(range(16))
    assert s.count("abcd") == 16


# ------------------------------------------------------------------- the app


def app_client(tmp_path: Path, **env) -> TestClient:
    settings = {
        "AI_BACKEND": "mock",
        "CANVAS_SIZE": "512",
        "AI_WINDOW": "512",
        "AI_DEBOUNCE_MS": "10",
        "HISTORY_DIR": str(tmp_path / "history"),
    }
    settings.update(env)
    return TestClient(create_app(load_config(settings), MockBackend(latency_ms=0)))


def wait_for(socket, want: str, timeout: float = 5.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = json.loads(socket.receive_text())
        if msg["t"] == want:
            return msg
    raise AssertionError(f"no {want} within {timeout}s")


def test_a_generation_is_saved_and_served(tmp_path: Path) -> None:
    with app_client(tmp_path) as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=a") as socket:
            wait_for(socket, "snapshot")
            socket.send_text(json.dumps({"t": "set_prompt", "prompt": "a hill"}))
            result = wait_for(socket, "ai_result")
        assert result["historyN"] == 0

        listing = client.get(f"/rooms/{room_id}/history").json()
        assert listing["enabled"] is True and listing["roomId"] == room_id
        assert len(listing["entries"]) == 1
        saved = listing["entries"][0]
        assert saved["prompt"] == "a hill"
        # The backend names itself, and the mock says so rather than passing
        # for a checkpoint.
        assert saved["model"] == "mock"
        assert saved["profile"] in ("fast", "quality")
        assert saved["aiResolution"] == 512 and saved["latencyMs"] >= 0
        assert saved["url"] == f"/rooms/{room_id}/history/0.jpg"

        image = client.get(saved["url"])
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        with Image.open(io.BytesIO(image.content)) as decoded:
            assert decoded.format == "JPEG" and decoded.size == (512, 512)


def test_a_generation_saves_the_canvas_it_was_generated_from(tmp_path: Path) -> None:
    with app_client(tmp_path) as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=a") as socket:
            wait_for(socket, "snapshot")
            socket.send_text(json.dumps({"t": "set_prompt", "prompt": "a hill"}))
            wait_for(socket, "ai_result")

        saved = client.get(f"/rooms/{room_id}/history").json()["entries"][0]
        assert saved["inputUrl"] == f"/rooms/{room_id}/history/0.in.jpg"
        image = client.get(saved["inputUrl"])
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.headers["cache-control"] == "public, max-age=86400, immutable"
        with Image.open(io.BytesIO(image.content)) as decoded:
            # The pixels the pipeline was handed: the generation size, not the
            # canvas size, because that is what it actually saw.
            assert decoded.format == "JPEG" and decoded.size == (512, 512)
        # An empty room draws nothing, so the input is the white background the
        # AI input is composited on rather than a black frame.
        with Image.open(io.BytesIO(image.content)) as decoded:
            assert decoded.convert("RGB").getpixel((5, 5)) > (240, 240, 240)


def test_history_outlives_the_room(tmp_path: Path) -> None:
    with app_client(tmp_path) as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=a") as socket:
            wait_for(socket, "snapshot")
            socket.send_text(json.dumps({"t": "set_prompt", "prompt": "kept"}))
            wait_for(socket, "ai_result")
        registry = client.app.state.registry
        registry.sweep(now=2**62)  # evict everything
        assert registry.size == 0
        assert client.get(f"/rooms/{room_id}/history").json()["entries"][0]["prompt"] == "kept"
        assert client.get(f"/rooms/{room_id}/history/0.jpg").status_code == 200

    # ... and a fresh process serves it too.
    with app_client(tmp_path) as second:
        assert len(second.get(f"/rooms/{room_id}/history").json()["entries"]) == 1


def test_routes_refuse_unsafe_names(tmp_path: Path) -> None:
    with app_client(tmp_path) as client:
        assert client.get("/rooms/AB/history").status_code == 404
        assert client.get("/rooms/abcd/history/0.png").status_code == 404
        assert client.get("/rooms/abcd/history/x.jpg").status_code == 404
        # An escaped traversal never reaches the store. (Whether it 404s or
        # falls through to the SPA depends on whether a client is built here;
        # what matters is that no file is served as an image.)
        traversal = client.get("/rooms/abcd/history/..%2F..%2Fsecret.jpg")
        assert traversal.headers["content-type"] != "image/jpeg"
        # A room with no history is an empty list, not an error.
        empty = client.get("/rooms/abcd/history")
        assert empty.status_code == 200 and empty.json()["entries"] == []
        assert client.get("/rooms/abcd/history/0.jpg").status_code == 404


def test_disabled_history_says_so_and_omits_historyn(tmp_path: Path) -> None:
    with app_client(tmp_path, HISTORY_ENABLED="0") as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=a") as socket:
            wait_for(socket, "snapshot")
            socket.send_text(json.dumps({"t": "set_prompt", "prompt": "nope"}))
            result = wait_for(socket, "ai_result")
        assert "historyN" not in result
        listing = client.get(f"/rooms/{room_id}/history").json()
        assert listing["enabled"] is False and listing["entries"] == []
        assert not (tmp_path / "history").exists()


def test_healthz_reports_activity(tmp_path: Path) -> None:
    with app_client(tmp_path) as client:
        idle = client.get("/healthz").json()
        assert idle["active_sockets"] == 0 and idle["last_generation_at"] is None
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=a") as socket:
            wait_for(socket, "snapshot")
            assert client.get("/healthz").json()["active_sockets"] == 1
            socket.send_text(json.dumps({"t": "set_prompt", "prompt": "busy"}))
            wait_for(socket, "ai_result")
        busy = client.get("/healthz").json()
        assert busy["generations"] == 1
        assert isinstance(busy["last_generation_at"], int)
        assert busy["active_sockets"] == 0


# ------------------------------------------- review 3: numbering and rollback


def test_numbers_are_never_reused_after_everything_is_evicted(tmp_path: Path) -> None:
    """Finding 4. `/rooms/{id}/history/{n}.jpg` is declared immutable, so a
    number that comes round again shows a browser the old picture under the new
    entry's settings. The counter outlives the entries it numbered."""
    first = store(tmp_path, room_bytes=10_000_000, total_bytes=10_000_000)
    for _ in range(3):
        first.record("abcd", JPEG, entry())
    assert [e["n"] for e in first.list("abcd")] == [2, 1, 0]

    # Another room fills up the store and evicts every one of abcd's entries.
    evicting = store(tmp_path, room_bytes=10_000_000, total_bytes=5_000)
    evicting.record("bbbb", JPEG, entry())
    assert evicting.list("abcd") == []
    assert evicting.count("abcd") == 0

    # A restart sees a room directory with no entries left in it.
    restarted = store(tmp_path)
    assert restarted.count("abcd") == 0
    assert restarted.record("abcd", JPEG, entry()) == 3


def test_the_counter_survives_a_restart_with_no_entries_at_all(tmp_path: Path) -> None:
    s = store(tmp_path)
    assert s.record("abcd", JPEG, entry()) == 0
    for suffix in ("jpg", "json"):
        (tmp_path / "history" / "abcd" / f"0.{suffix}").unlink()
    assert store(tmp_path).record("abcd", JPEG, entry()) == 1


def test_a_half_written_pair_is_rolled_back(tmp_path: Path, monkeypatch) -> None:
    """Finding 5. The JPEG used to be installed before the JSON was even
    written, so a failure there left an image nothing listed and no budget
    counted."""
    s = store(tmp_path)
    real = _history._write_atomic

    def fail_on_json(path: Path, data: bytes) -> None:
        if path.name.endswith(".json"):
            raise OSError("no space left on device")
        real(path, data)

    monkeypatch.setattr(_history, "_write_atomic", fail_on_json)
    assert s.record("abcd", JPEG, entry()) is None
    monkeypatch.undo()

    room = tmp_path / "history" / "abcd"
    assert not (room / "0.jpg").exists(), "the image outlived the entry"
    assert list(room.glob("*.part")) == []
    assert s.total_bytes_used == 0
    # The number is still spent, and the store still works.
    assert s.record("abcd", JPEG, entry()) == 1


def test_an_interrupted_write_is_reconciled_on_load(tmp_path: Path) -> None:
    room = tmp_path / "history" / "abcd"
    room.mkdir(parents=True)
    (room / "0.jpg").write_bytes(JPEG)  # installed
    (room / "0.json").write_text('{"prompt":"a hill"}', encoding="utf-8")
    (room / "1.jpg").write_bytes(JPEG)  # the crash landed here
    (room / "2.jpg.part").write_bytes(JPEG)  # ...and here
    s = store(tmp_path)
    assert [e["n"] for e in s.list("abcd")] == [0]
    assert not (room / "1.jpg").exists()
    assert not (room / "2.jpg.part").exists()
    assert s.total_bytes_used == (room / "0.jpg").stat().st_size + (
        room / "0.json"
    ).stat().st_size


def test_bytes_that_would_not_delete_are_still_counted(tmp_path: Path, monkeypatch) -> None:
    """Finding 5, second half: eviction subtracted the bytes and forgot the
    files, so the store reported nothing while still holding them."""
    s = store(tmp_path, room_bytes=10_000, total_bytes=10_000_000)
    s.record("abcd", JPEG, entry())
    s.record("abcd", JPEG, entry())
    used = s.total_bytes_used

    locked = {"on": True}
    real_unlink = Path.unlink

    def refuse(self: Path, *args, **kw):
        if locked["on"] and self.name.startswith("0."):
            raise PermissionError("the file is open in another process")
        return real_unlink(self, *args, **kw)

    monkeypatch.setattr(Path, "unlink", refuse)
    s.record("abcd", JPEG, entry())  # this one pushes entry 0 out
    assert ("abcd", 0) in s.undeleted()
    assert (tmp_path / "history" / "abcd" / "0.jpg").exists()
    assert s.total_bytes_used > used - 1, "bytes still on the disk stopped being counted"

    # The lock goes away and the next write clears the backlog.
    locked["on"] = False
    s.record("abcd", JPEG, entry())
    assert s.undeleted() == set()
    assert not (tmp_path / "history" / "abcd" / "0.jpg").exists()
    assert s.total_bytes_used == sum(
        f.stat().st_size for f in (tmp_path / "history" / "abcd").glob("*.js*")
    ) + sum(f.stat().st_size for f in (tmp_path / "history" / "abcd").glob("*.jpg"))


# ------------------------------------------- review 4: counters and leftovers


def _plant(root: Path, room: str, n: int) -> None:
    """One complete entry, written the way a pre-counter version would have."""
    directory = root / "history" / room
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{n}.jpg").write_bytes(JPEG)
    (directory / f"{n}.json").write_text(json.dumps({"n": n, "prompt": "old"}), encoding="utf-8")


def test_a_room_from_before_counters_gets_one_written_at_load(tmp_path: Path) -> None:
    """Finding 2. The high-water mark was reconstructed into memory only, so
    an eviction in the same process took away the files it was reconstructed
    from and the next restart began again at 0."""
    _plant(tmp_path, "abcd", 7)
    counter = tmp_path / "history" / "abcd" / "counter"
    assert not counter.exists()

    # Loading is enough: the counter is on disk before anything can evict.
    s = store(tmp_path, room_bytes=10_000_000, total_bytes=10_000_000)
    assert s.count("abcd") == 1
    assert counter.read_text(encoding="ascii").strip() == "8"

    # Now the reviewer's sequence: another room's write evicts abcd entirely.
    evicting = store(tmp_path, room_bytes=10_000_000, total_bytes=5_000)
    evicting.record("bbbb", JPEG, entry())
    assert evicting.list("abcd") == []
    # ...and a restart still does not reissue 7.
    assert store(tmp_path).record("abcd", JPEG, entry()) == 8


def test_entries_are_kept_when_their_counter_cannot_be_written(
    tmp_path: Path, monkeypatch
) -> None:
    """If the number cannot be written down, the files that say what it is are
    the only record left, so they are not evicted."""
    _plant(tmp_path, "abcd", 4)
    real = _history._write_atomic

    def refuse_counter(path: Path, data: bytes) -> None:
        if path.name == "counter" and "abcd" in str(path):
            raise OSError("read-only file system")
        real(path, data)

    monkeypatch.setattr(_history, "_write_atomic", refuse_counter)
    s = store(tmp_path, room_bytes=10_000_000, total_bytes=1_000)
    assert s.unprotected() == {"abcd"}
    s.record("bbbb", JPEG, entry())  # would evict abcd under the total budget
    assert s.count("abcd") == 1, "the only record of the room's numbering was evicted"
    assert (tmp_path / "history" / "abcd" / "4.jpg").exists()

    # The disk comes back: the counter is written on the next write and the
    # room stops being protected.
    monkeypatch.undo()
    s.record("bbbb", JPEG, entry())
    assert s.unprotected() == set()
    assert (tmp_path / "history" / "abcd" / "counter").read_text(encoding="ascii").strip() == "5"


def test_a_failed_replace_leaves_no_temporary_behind(tmp_path: Path, monkeypatch) -> None:
    """Finding 3. The JSON's `.part` file survived a failed install, with the
    store reporting no bytes at all."""
    s = store(tmp_path)
    real_replace = os.replace

    def refuse_json(src, dst, *args, **kw):
        if str(dst).endswith(".json"):
            raise OSError("no space left on device")
        return real_replace(src, dst, *args, **kw)

    monkeypatch.setattr(_history.os, "replace", refuse_json)
    assert s.record("abcd", JPEG, entry()) is None
    monkeypatch.undo()

    room = tmp_path / "history" / "abcd"
    assert sorted(f.name for f in room.iterdir()) == ["counter"], sorted(
        f.name for f in room.iterdir()
    )
    assert s.total_bytes_used == 0
    assert s.stray() == set()


def test_leftovers_that_will_not_delete_are_counted_and_retried(
    tmp_path: Path, monkeypatch
) -> None:
    s = store(tmp_path)
    real_replace = os.replace
    real_unlink = Path.unlink
    locked = {"on": True}

    def refuse_json(src, dst, *args, **kw):
        if str(dst).endswith(".json"):
            raise OSError("no space left on device")
        return real_replace(src, dst, *args, **kw)

    def refuse_unlink(self: Path, *args, **kw):
        if locked["on"] and self.name.endswith(".json.part"):
            raise PermissionError("the file is open in another process")
        return real_unlink(self, *args, **kw)

    monkeypatch.setattr(_history.os, "replace", refuse_json)
    monkeypatch.setattr(Path, "unlink", refuse_unlink)
    assert s.record("abcd", JPEG, entry()) is None
    leftover = tmp_path / "history" / "abcd" / "0.json.part"
    assert leftover.exists()
    assert s.stray() == {leftover}
    assert s.total_bytes_used == leftover.stat().st_size, "stray bytes were not counted"

    # The next write retries it, and the accounting follows the disk.
    monkeypatch.undo()
    locked["on"] = False
    assert s.record("abcd", JPEG, entry()) == 1
    assert not leftover.exists()
    assert s.stray() == set()
    assert s.total_bytes_used == sum(
        f.stat().st_size for f in (tmp_path / "history" / "abcd").glob("1.*")
    )


def test_a_part_file_on_disk_is_cleaned_up_at_load(tmp_path: Path) -> None:
    room = tmp_path / "history" / "abcd"
    room.mkdir(parents=True)
    (room / "0.jpg.part").write_bytes(JPEG)
    s = store(tmp_path)
    assert s.count("abcd") == 0
    assert not (room / "0.jpg.part").exists()
    assert s.total_bytes_used == 0


# --------------------------------------- review 5: what eviction may not take


def test_the_entry_just_recorded_is_never_the_one_evicted(tmp_path: Path, monkeypatch) -> None:
    """`record` returns a number the client is about to fetch. Skipping over a
    protected room used to walk the eviction loop all the way to the newest
    entry, so a full store answered with a number whose image it had just
    deleted."""
    _plant(tmp_path, "aaaa", 0)  # an old room, and its counter will not write
    real = _history._write_atomic

    def refuse_that_rooms_counter(path: Path, data: bytes) -> None:
        if path.name == "counter" and "aaaa" in str(path):
            raise OSError("read-only file system")
        real(path, data)

    monkeypatch.setattr(_history, "_write_atomic", refuse_that_rooms_counter)
    s = store(tmp_path, room_bytes=10_000_000, total_bytes=1_000)
    assert s.unprotected() == {"aaaa"}  # over budget and cannot be evicted

    n = s.record("bbbb", JPEG, entry())
    assert n == 0
    assert s.read("bbbb", 0) == JPEG, "the store returned a number it had deleted"
    assert [e["n"] for e in s.list("bbbb")] == [0]


def test_a_room_over_its_own_budget_keeps_the_entry_just_written(tmp_path: Path) -> None:
    s = store(tmp_path, room_bytes=100, total_bytes=10_000_000)
    for _ in range(3):
        n = s.record("abcd", JPEG, entry())
        assert s.read("abcd", n) == JPEG
    assert [e["n"] for e in s.list("abcd")] == [2]


def test_incomplete_files_are_kept_until_their_number_is_written_down(
    tmp_path: Path, monkeypatch
) -> None:
    """Finding 4. A pre-counter room left holding only `7.json` after an
    interrupted eviction: deleting it and then failing to write the counter
    threw away the only evidence that 7 was ever used, and a restart would
    reissue that immutable URL."""
    room = tmp_path / "history" / "abcd"
    room.mkdir(parents=True)
    (room / "7.json").write_text('{"n": 7}', encoding="utf-8")  # half an entry
    real = _history._write_atomic

    def refuse_counter(path: Path, data: bytes) -> None:
        if path.name == "counter":
            raise OSError("read-only file system")
        real(path, data)

    monkeypatch.setattr(_history, "_write_atomic", refuse_counter)
    s = store(tmp_path)
    assert s.count("abcd") == 0  # it is not an entry
    assert (room / "7.json").exists(), "the only record of number 7 was deleted"
    assert s.unprotected() == {"abcd"}
    assert s.total_bytes_used == (room / "7.json").stat().st_size  # ...but counted

    # The disk comes back. The counter is written, and only then is the
    # leftover cleaned up.
    monkeypatch.undo()
    assert s.record("abcd", JPEG, entry()) == 8  # 7 is spent, and stays spent
    assert (room / "counter").read_text(encoding="ascii").strip() == "9"
    assert not (room / "7.json").exists()
    assert s.stray() == set()


def test_a_counter_temporary_that_survives_is_counted_and_retried(
    tmp_path: Path, monkeypatch
) -> None:
    """Finding 6."""
    s = store(tmp_path)
    real_replace = os.replace
    real_unlink = Path.unlink
    locked = {"on": True}

    def refuse_counter(src, dst, *args, **kw):
        if str(dst).endswith("counter"):
            raise OSError("no space left on device")
        return real_replace(src, dst, *args, **kw)

    def refuse_unlink(self: Path, *args, **kw):
        if locked["on"] and self.name == "counter.part":
            raise PermissionError("the file is open in another process")
        return real_unlink(self, *args, **kw)

    monkeypatch.setattr(_history.os, "replace", refuse_counter)
    monkeypatch.setattr(Path, "unlink", refuse_unlink)
    assert s.record("abcd", JPEG, entry()) is None
    leftover = tmp_path / "history" / "abcd" / "counter.part"
    assert leftover.exists()
    assert leftover in s.stray()
    assert s.total_bytes_used == leftover.stat().st_size

    monkeypatch.undo()
    locked["on"] = False
    assert s.record("abcd", JPEG, entry()) == 0
    assert not leftover.exists()
    assert s.stray() == set()
