"""Saved AI results: one JPEG plus one JSON entry per accepted generation.

A room is memory - it is evicted when nobody is in it, and it dies with the
process. What the room *made* is worth more than that, so every accepted
`ai_result` is also written to disk, under `HISTORY_DIR/<roomId>/`, and served
back by `/rooms/{id}/history` regardless of whether that room still exists.

Everything here is blocking and belongs on a worker thread; `record` is the
only entry point the runtime uses, and it is called through `asyncio.to_thread`.
Nothing in here may raise into a generation: a full disk must cost the history,
never the picture.

Three things on disk make one room: `<n>.jpg`, `<n>.json`, and a `counter` file
holding the next number to hand out. An entry exists when *both* of its files
do - the JSON is installed last and is the commit point - and a number is never
handed out twice, because the counter outlives the entries it numbered.

A fourth file, `<n>.in.jpg`, is the human canvas that was the *input* of that
generation - the raster the pipeline was handed. It is optional: entries
written before it existed have none, and an encode that failed leaves none, so
its absence is never an incomplete entry. It is written first, before the pair,
because the pair is what defines the entry and the extra must never be the
thing that is half installed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

#: The next entry number this room will use, kept beside its entries.
#:
#: Rebuilding the number from the files that survive is not good enough: when
#: the global budget evicts every entry a room has, a restart would begin again
#: at 0 and reissue numbers that browsers already hold - and `/history/<n>.jpg`
#: is declared immutable, so the old picture would be shown under the new
#: entry's settings.
COUNTER_NAME = "counter"


#: The three files one entry can own, and the suffix each is stored under.
#: `jpg` and `json` make the entry; `in` is the optional input canvas.
ENTRY_SUFFIX = {"jpg": "jpg", "json": "json", "in": "in.jpg"}


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
        #: Entries whose files would not delete (a reader holding them open on
        #: Windows, a read-only mount). Their bytes are still on the disk, so
        #: they are still counted, and every later write tries again.
        self._undeleted: Dict[Tuple[str, int], int] = {}
        #: Files that belong to no entry - a temporary file from a write that
        #: failed, half of a pair - and would not delete. Their bytes are on
        #: the disk, so they are counted, and every later write tries again.
        self._stray: Dict[Path, int] = {}
        #: Rooms whose counter could not be written. Their entries are not
        #: evicted: those files are the only record of the numbers this room
        #: has already handed out, and losing them would reissue them.
        self._unprotected: Set[str] = set()
        self._total = 0

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        """Adopt whatever is already on disk. Once, lazily, under the lock.

        Restarting must not restart the numbering, or a new entry would
        overwrite an old one, and the budgets have to count what is already
        there rather than only what this process wrote. This is also where a
        half-written pair is cleaned up: an interrupted write leaves a `.part`
        file, and a crash between the two installs leaves a JPEG with no JSON.
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
            found: Dict[str, Dict[int, int]] = {kind: {} for kind in ENTRY_SUFFIX}
            stamps: Dict[int, float] = {}
            high_water = 0
            counter = 0
            for item in files:
                if item.name == COUNTER_NAME:
                    counter = _read_counter(Path(item.path))
                    continue
                if item.name.endswith(".part"):
                    # An interrupted write. It was never installed, so nothing
                    # refers to it - but it is bytes on the disk until it goes.
                    self._discard_file(Path(item.path))
                    continue
                parsed = _entry_file(item.name)
                if parsed is None:
                    continue
                n, kind = parsed
                high_water = max(high_water, n + 1)
                try:
                    stat = item.stat()
                except OSError:  # pragma: no cover
                    continue
                found[kind][n] = stat.st_size
                stamps[n] = max(stamps.get(n, 0.0), stat.st_mtime)
            self._next[room.name] = max(counter, high_water)
            written = counter >= high_water or self._persist_counter(
                # A room written before there were counters, or one whose
                # counter was lost. Its numbering exists only in the names of
                # the files that are here, so write it down BEFORE touching any
                # of them - including the incomplete ones below, whose names
                # may be the highest number this room ever used.
                room.name,
                self._next[room.name],
            )
            sizes: Dict[int, int] = {}
            for n in sorted(set().union(*(set(v) for v in found.values()))):
                if n in found["jpg"] and n in found["json"]:
                    # The input canvas is optional, but its bytes are on the
                    # disk, so the entry is charged for all three files.
                    sizes[n] = found["jpg"][n] + found["json"][n] + found["in"].get(n, 0)
                    continue
                if not written:
                    # The counter did not go down. These filenames are the only
                    # record of the numbers this room has spent, so they stay -
                    # unlisted, but on the disk - until a later write manages to
                    # write the counter (`_retry_deletions`).
                    log.warning(
                        "history: room %s entry %d is incomplete but its counter is not "
                        "written; keeping the file as the record of that number",
                        room.name,
                        n,
                    )
                    for kind, found_sizes in found.items():
                        if n in found_sizes:
                            path = self.root / room.name / f"{n}.{ENTRY_SUFFIX[kind]}"
                            self._stray[path] = found_sizes[n]
                            self._total += found_sizes[n]
                    continue
                # Half a pair: the write was interrupted between installing the
                # image and installing the entry, or between deleting them.
                # Either way it is not an entry, and leaving it would be bytes
                # on the disk that no budget knows about.
                log.info("history: room %s entry %d is incomplete; removing it", room.name, n)
                for suffix in ENTRY_SUFFIX.values():
                    self._discard_file(self.root / room.name / f"{n}.{suffix}")
            if not sizes:
                continue
            self._sizes[room.name] = sizes
            self._total += sum(sizes.values())
            for n, size in sizes.items():
                entries.append((stamps.get(n, 0.0), room.name, n, size))
        # Oldest first by mtime, then by number, so eviction order survives a
        # restart in the order the entries were actually written.
        entries.sort(key=lambda e: (e[0], e[2]))
        self._order = [(room, n) for _stamp, room, n, _size in entries]

    def _persist_counter(self, room_id: str, nxt: int) -> bool:
        """Write down the next number this room will use. Caller holds the lock.

        A room whose counter cannot be written keeps its files instead: they
        are the only thing that says which numbers are spent.
        """
        try:
            (self.root / room_id).mkdir(parents=True, exist_ok=True)
            _write_atomic(self.root / room_id / COUNTER_NAME, f"{nxt}\n".encode("ascii"))
        except Exception as err:
            # `_write_atomic` takes its own temporary with it where it can;
            # what it could not remove is counted and retried like any other
            # stray byte.
            self._discard_file(self.root / room_id / (COUNTER_NAME + ".part"))
            if room_id not in self._unprotected:
                log.warning(
                    "history: cannot write the counter for room %s (%s); "
                    "its entries will not be evicted",
                    room_id,
                    err,
                )
            self._unprotected.add(room_id)
            return False
        self._unprotected.discard(room_id)
        return True

    # -- writing ----------------------------------------------------------

    def record(
        self,
        room_id: str,
        jpeg: bytes,
        entry: Dict[str, Any],
        input_jpeg: Optional[bytes] = None,
    ) -> Optional[int]:
        """Store one result. Returns its number, or None if nothing was stored.

        `input_jpeg` is the human canvas that produced it, and is optional: it
        is stored beside the pair when it is there and skipped when it is not.
        A generation whose input could not be encoded is still a generation.

        Never raises: a history that cannot be written is a log line, not a
        failed generation.
        """
        if not self.enabled or not jpeg or not is_room_id(room_id):
            return None
        try:
            with self._lock:
                self._load()
                self._retry_deletions()
                n = self._next.get(room_id, 0)
                directory = self.root / room_id
                directory.mkdir(parents=True, exist_ok=True)
                record = dict(entry)
                record["n"] = n
                blob = json.dumps(record, separators=(",", ":")).encode("utf-8")
                jpg_path = directory / f"{n}.jpg"
                json_path = directory / f"{n}.json"
                in_path = directory / f"{n}.in.jpg"
                # The number is spent the moment it is written down, whether or
                # not the files that follow land: reusing it after a failure
                # would put two different pictures behind one immutable URL.
                if not self._persist_counter(room_id, n + 1):
                    return None
                self._next[room_id] = n + 1
                in_size = 0
                try:
                    # First, because it is the one file that is allowed to be
                    # missing: writing it after the commit point would put an
                    # entry on the disk whose input arrives a moment later.
                    if input_jpeg:
                        _write_atomic(in_path, input_jpeg)
                        in_size = len(input_jpeg)
                    _write_atomic(jpg_path, jpeg)
                    # The JSON is the commit point: an entry exists when both
                    # files do, and `_load` cleans up anything that is half a
                    # pair. So install it last, and take the image back out if
                    # it does not land.
                    _write_atomic(json_path, blob)
                except Exception:
                    # Every name and every temporary: a failed `os.replace`
                    # leaves the `.part` file behind, and it is as much a stray
                    # byte as a half-installed entry.
                    for path in (jpg_path, json_path, in_path):
                        self._discard_file(path)
                        self._discard_file(path.with_name(path.name + ".part"))
                    raise
                size = len(jpeg) + len(blob) + in_size
                self._sizes.setdefault(room_id, {})[n] = size
                self._order.append((room_id, n))
                self._total += size
                self._evict(room_id, keep=(room_id, n))
                return n
        except Exception as err:
            log.warning("history: could not store a result for room %s (%s)", room_id, err)
            return None

    def _evict(self, room_id: str, keep: Optional[Tuple[str, int]] = None) -> None:
        """Oldest first: the room's own budget, then the whole store's.

        `keep` is the entry just written, and it is never evicted: storing a
        result and immediately deleting it would hand the client a number
        nothing serves, which is worse than being a little over budget. It used
        to be enough that it was last in `_order`, until protected rooms could
        be skipped over - and then the "last one standing" was the entry whose
        number `record` was about to return.
        """
        while room_id not in self._unprotected:
            room_sizes = self._sizes.get(room_id) or {}
            if sum(room_sizes.values()) <= self.room_bytes:
                break
            oldest = min((n for n in room_sizes if (room_id, n) != keep), default=None)
            if oldest is None:
                break
            self._remove(room_id, oldest)
        index = 0
        while self._total > self.total_bytes and index < len(self._order):
            room, n = self._order[index]
            if room in self._unprotected or (room, n) == keep:
                index += 1
                continue
            self._remove(room, n)

    def _remove(self, room_id: str, n: int) -> None:
        """Take one entry out of the store, and off the disk if it will go.

        The accounting follows the disk, not the intention: bytes that are
        still there are still counted, and the deletion is tried again on the
        next write. Subtracting first meant a store that reported nothing while
        holding files it had failed to delete.
        """
        size = (self._sizes.get(room_id) or {}).pop(n, 0)
        try:
            self._order.remove((room_id, n))
        except ValueError:  # pragma: no cover - already gone
            pass
        if self._delete_files(room_id, n):
            self._total -= size
        else:
            self._undeleted[(room_id, n)] = size
        if not self._sizes.get(room_id):
            self._sizes.pop(room_id, None)

    def _delete_files(self, room_id: str, n: int) -> bool:
        """True when neither file is on the disk any more."""
        gone = True
        for suffix in ENTRY_SUFFIX.values():
            path = self.root / room_id / f"{n}.{suffix}"
            if not _unlink(path):
                gone = False
        return gone

    def _discard_file(self, path: Path) -> None:
        """Remove a file that belongs to no entry, and keep the books straight.

        Caller holds the lock. A file that will not go is not a file that has
        gone: its bytes stay counted and it is tried again on the next write.
        """
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if _unlink(path):
            if path in self._stray:
                self._total -= self._stray.pop(path)
            return
        if path not in self._stray:
            self._stray[path] = size
            self._total += size

    def _retry_deletions(self) -> None:
        """Caller holds the lock. Everything that would not delete last time."""
        for key in list(self._undeleted):
            room_id, n = key
            if self._delete_files(room_id, n):
                self._total -= self._undeleted.pop(key)
        for room_id in list(self._unprotected):
            self._persist_counter(room_id, self._next.get(room_id, 0))
        for path in list(self._stray):
            # A file kept because its name is the only record of a number is
            # not litter until that number is written down.
            if path.parent.name in self._unprotected:
                continue
            self._discard_file(path)

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
            # Absent rather than null when there is no input: entries written
            # before inputs were stored are still perfectly good entries.
            if (self.root / room_id / f"{n}.in.jpg").exists():
                entry["inputUrl"] = f"/rooms/{room_id}/history/{n}.in.jpg"
            out.append(entry)
        return out

    def read(self, room_id: str, n: int, kind: str = "jpg") -> Optional[bytes]:
        """One stored file: the AI result (`jpg`) or its input (`in`)."""
        suffix = ENTRY_SUFFIX.get(kind)
        if suffix is None or suffix == "json":
            return None
        if not self.enabled or not is_room_id(room_id):
            return None
        try:
            return (self.root / room_id / f"{int(n)}.{suffix}").read_bytes()
        except (OSError, ValueError):
            return None

    def entries(self, room_id: str) -> List[Dict[str, Any]]:
        """Every entry of a room, oldest first, for an export.

        Unlike `list` this is not capped and each entry carries the paths of
        its files rather than their URLs, because the caller is about to read
        them off the disk.
        """
        if not self.enabled or not is_room_id(room_id):
            return []
        try:
            with self._lock:
                self._load()
                numbers = sorted((self._sizes.get(room_id) or {}).keys())
        except Exception as err:  # pragma: no cover - defensive
            log.warning("history: cannot enumerate room %s (%s)", room_id, err)
            return []
        out: List[Dict[str, Any]] = []
        for n in numbers:
            directory = self.root / room_id
            try:
                entry = json.loads((directory / f"{n}.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            entry["n"] = n
            in_path = directory / f"{n}.in.jpg"
            out.append(
                {
                    "n": n,
                    "entry": entry,
                    "gen": directory / f"{n}.jpg",
                    "input": in_path if in_path.exists() else None,
                }
            )
        return out

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

    def undeleted(self) -> Set[Tuple[str, int]]:
        """Entries whose files are still on the disk after a failed delete."""
        with self._lock:
            return set(self._undeleted)

    def stray(self) -> Set[Path]:
        """Files belonging to no entry that would not delete."""
        with self._lock:
            return set(self._stray)

    def unprotected(self) -> Set[str]:
        """Rooms with no counter on disk, whose entries are not evicted."""
        with self._lock:
            self._load()
            return set(self._unprotected)


def _entry_file(name: str) -> Optional[Tuple[int, str]]:
    """`12.jpg` -> (12, "jpg"), `12.json` -> (12, "json"), `12.in.jpg` -> (12, "in")."""
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        return None
    if suffix == "jpg" and stem.endswith(".in"):
        stem, suffix = stem[:-3], "in"
    if suffix not in ENTRY_SUFFIX or not stem.isdigit():
        return None
    return int(stem), suffix


def _entry_number(name: str) -> Optional[int]:
    parsed = _entry_file(name)
    return None if parsed is None else parsed[0]


def _read_counter(path: Path) -> int:
    try:
        return max(0, int(path.read_text(encoding="ascii").strip()))
    except (OSError, ValueError):  # pragma: no cover - hand-edited or unreadable
        log.warning("history: unreadable counter at %s; using the files instead", path)
        return 0


def _unlink(path: Path) -> bool:
    """True when the path is not there afterwards."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError as err:
        log.debug("history: could not remove %s (%s)", path, err)
        return False


def _write_atomic(path: Path, data: bytes) -> None:
    """A reader must never see half a JPEG, and a crash must not leave one.

    A failure here - a full disk mid-write, a replace that will not go - takes
    the temporary file with it where it can, so the ordinary failure leaves
    nothing behind. What it cannot remove, the caller records as stray.
    """
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except Exception:
        _unlink(tmp)
        raise
