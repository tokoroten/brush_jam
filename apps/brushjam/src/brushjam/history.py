"""Saved AI results: one JPEG plus one JSON entry per accepted generation.

A room is memory - it is evicted when nobody is in it, and it dies with the
process. What the room *made* is worth more than that, so every accepted
`ai_result` is also written to disk, under `HISTORY_DIR/<roomId>/`, and served
back by `/rooms/{id}/history` regardless of whether that room still exists.

Everything here is blocking and belongs on a worker thread; `record` is the
only entry point the runtime uses, and it is called through `asyncio.to_thread`.
Nothing in here may raise into a generation: a full disk must cost the history,
never the picture.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("brushjam.history")

#: Ids that may become a directory name. Identical to the route's, because a
#: path element is exactly what this turns them into.
ROOM_ID = re.compile(r"^[a-z0-9]{4,16}$")

#: Newest entries a listing returns without being asked for more. A room that
#: has been running all afternoon holds thousands; a gallery shows a strip.
DEFAULT_LIST_LIMIT = 120
MAX_LIST_LIMIT = 500

#: JPEG quality for the stored composite. 90 is where an SDXL result stops
#: gaining anything visible and the file is a fifth of the PNG.
JPEG_QUALITY = 90


def is_room_id(room_id: str) -> bool:
    return bool(ROOM_ID.match(room_id or ""))


class HistoryStore:
    """Per-room saved results, with a per-room and a whole-store byte budget.

    Sizes are held in memory and maintained incrementally: the alternative is
    walking the directory on every generation, which is a file system scan
    every two seconds for as long as anybody is drawing.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        room_bytes: int = 200 * 1024 * 1024,
        total_bytes: int = 2000 * 1024 * 1024,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.room_bytes = int(room_bytes)
        self.total_bytes = int(total_bytes)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._loaded = False
        #: room -> {n: bytes on disk}
        self._sizes: Dict[str, Dict[int, int]] = {}
        #: room -> the next entry number to hand out
        self._next: Dict[str, int] = {}
        #: (room, n) oldest first, which is the order eviction removes them in.
        self._order: List[Tuple[str, int]] = []
        self._total = 0

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        """Adopt whatever is already on disk. Once, lazily, under the lock.

        Restarting must not restart the numbering, or a new entry would
        overwrite an old one, and the budgets have to count what is already
        there rather than only what this process wrote.
        """
        if self._loaded:
            return
        self._loaded = True
        entries: List[Tuple[float, str, int, int]] = []
        try:
            rooms = list(os.scandir(self.root)) if self.root.exists() else []
        except OSError as err:  # pragma: no cover - unreadable root
            log.warning("history: cannot read %s (%s); starting empty", self.root, err)
            return
        for room in rooms:
            if not room.is_dir() or not is_room_id(room.name):
                continue
            try:
                files = list(os.scandir(room.path))
            except OSError:  # pragma: no cover - vanished between calls
                continue
            sizes: Dict[int, int] = {}
            stamps: Dict[int, float] = {}
            for item in files:
                n = _entry_number(item.name)
                if n is None:
                    continue
                try:
                    stat = item.stat()
                except OSError:  # pragma: no cover
                    continue
                sizes[n] = sizes.get(n, 0) + stat.st_size
                stamps[n] = max(stamps.get(n, 0.0), stat.st_mtime)
            if not sizes:
                continue
            self._sizes[room.name] = sizes
            self._next[room.name] = max(sizes) + 1
            self._total += sum(sizes.values())
            for n, size in sizes.items():
                entries.append((stamps.get(n, 0.0), room.name, n, size))
        # Oldest first by mtime, then by number, so eviction order survives a
        # restart in the order the entries were actually written.
        entries.sort(key=lambda e: (e[0], e[2]))
        self._order = [(room, n) for _stamp, room, n, _size in entries]

    # -- writing ----------------------------------------------------------

    def record(self, room_id: str, jpeg: bytes, entry: Dict[str, Any]) -> Optional[int]:
        """Store one result. Returns its number, or None if nothing was stored.

        Never raises: a history that cannot be written is a log line, not a
        failed generation.
        """
        if not self.enabled or not jpeg or not is_room_id(room_id):
            return None
        try:
            with self._lock:
                self._load()
                n = self._next.get(room_id, 0)
                self._next[room_id] = n + 1
                directory = self.root / room_id
                directory.mkdir(parents=True, exist_ok=True)
                record = dict(entry)
                record["n"] = n
                jpg_path = directory / f"{n}.jpg"
                json_path = directory / f"{n}.json"
                _write_atomic(jpg_path, jpeg)
                _write_atomic(json_path, json.dumps(record, separators=(",", ":")).encode("utf-8"))
                size = len(jpeg) + json_path.stat().st_size
                self._sizes.setdefault(room_id, {})[n] = size
                self._order.append((room_id, n))
                self._total += size
                self._evict(room_id)
                return n
        except Exception as err:
            log.warning("history: could not store a result for room %s (%s)", room_id, err)
            return None

    def _evict(self, room_id: str) -> None:
        """Oldest first: the room's own budget, then the whole store's.

        The entry just written is the newest, so it is last in `_order` and in
        the room's numbering: both loops stop before reaching it. Storing a
        result and immediately deleting it would hand the client a number
        nothing serves, which is worse than being a little over budget.
        """
        while len(self._sizes.get(room_id) or {}) > 1:
            room_sizes = self._sizes[room_id]
            if sum(room_sizes.values()) <= self.room_bytes:
                break
            self._remove(room_id, min(room_sizes))
        while self._total > self.total_bytes and len(self._order) > 1:
            room, n = self._order[0]
            self._remove(room, n)

    def _remove(self, room_id: str, n: int) -> None:
        size = (self._sizes.get(room_id) or {}).pop(n, 0)
        self._total -= size
        try:
            self._order.remove((room_id, n))
        except ValueError:  # pragma: no cover - already gone
            pass
        for suffix in (".jpg", ".json"):
            try:
                (self.root / room_id / f"{n}{suffix}").unlink()
            except FileNotFoundError:
                pass
            except OSError as err:  # pragma: no cover - locked by a reader
                log.debug("history: could not remove %s/%s%s (%s)", room_id, n, suffix, err)
        if not self._sizes.get(room_id):
            self._sizes.pop(room_id, None)

    # -- reading ----------------------------------------------------------

    def list(self, room_id: str, limit: int = DEFAULT_LIST_LIMIT) -> List[Dict[str, Any]]:
        """Newest first, each entry carrying the URL of its image."""
        if not self.enabled or not is_room_id(room_id):
            return []
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        try:
            with self._lock:
                self._load()
                numbers = sorted((self._sizes.get(room_id) or {}).keys(), reverse=True)[:limit]
        except Exception as err:  # pragma: no cover - defensive
            log.warning("history: cannot list room %s (%s)", room_id, err)
            return []
        out: List[Dict[str, Any]] = []
        for n in numbers:
            try:
                raw = (self.root / room_id / f"{n}.json").read_text(encoding="utf-8")
                entry = json.loads(raw)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            entry["n"] = n
            entry["url"] = f"/rooms/{room_id}/history/{n}.jpg"
            out.append(entry)
        return out

    def read(self, room_id: str, n: int) -> Optional[bytes]:
        if not self.enabled or not is_room_id(room_id):
            return None
        try:
            return (self.root / room_id / f"{int(n)}.jpg").read_bytes()
        except (OSError, ValueError):
            return None

    # -- introspection, for the tests and /healthz ------------------------

    @property
    def total_bytes_used(self) -> int:
        with self._lock:
            self._load()
            return self._total

    def room_bytes_used(self, room_id: str) -> int:
        with self._lock:
            self._load()
            return sum((self._sizes.get(room_id) or {}).values())

    def count(self, room_id: str) -> int:
        with self._lock:
            self._load()
            return len(self._sizes.get(room_id) or {})


def _entry_number(name: str) -> Optional[int]:
    stem, dot, suffix = name.rpartition(".")
    if not dot or suffix not in ("jpg", "json") or not stem.isdigit():
        return None
    return int(stem)


def _write_atomic(path: Path, data: bytes) -> None:
    """A reader must never see half a JPEG, and a crash must not leave one."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)
