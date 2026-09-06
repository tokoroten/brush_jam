"""Taking a room's history away with you: a zip of the frames, or a video.

The gallery shows what a room made; these two endpoints let somebody keep it.
A zip is the archival form - the original JPEGs, byte for byte, plus a manifest
that says what each one was generated with. The AVI is the form you can watch:
one frame per accepted generation, the drawing on the left and what the model
made of it on the right, which is the thing this app is actually about.

Both are built on a worker thread into a temp file and streamed from there.
Building in memory would mean holding a room's whole history - hundreds of
megabytes, by the budget's own definition - in the process, and building on the
event loop would stall every other room's sockets for as long as it took.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PIL import Image

from .avi import AviTooLarge, MjpegAviWriter
from .history import HistoryStore

log = logging.getLogger("brushjam.export")

#: Frames past this and the answer is the zip. 3000 frames is a 12-minute video
#: at 4 fps and a room that has been running for a day and a half.
DEFAULT_MAX_FRAMES = 3000

#: Frame rate bounds for the video. Below 1 there is no frame period an AVI can
#: express; past 30 the file is longer than the flipbook is worth.
MIN_FPS = 1
MAX_FPS = 30
DEFAULT_FPS = 4

#: Whole-frame width bounds. The default is whatever was stored, so this only
#: applies when the caller asks for a size.
MIN_WIDTH = 256
MAX_WIDTH = 4096

#: The frames are recompressed for the video (two stored JPEGs become one
#: side-by-side frame), so this is a second generation of loss on top of the
#: stored 90. 85 is where it stops being visible in motion.
VIDEO_QUALITY = 85

#: Where the temp files go: inside the history root, because that is the volume
#: sized for this data. Not a room id, so the store's loader ignores it.
EXPORT_DIR = ".exports"


class ExportBusy(RuntimeError):
    """An export for this room, or too many overall, is already running."""

    retry_after = 5


class ExportEmpty(RuntimeError):
    """Nothing has been saved for this room."""


class ExportTooLarge(RuntimeError):
    """More frames than the video form can carry. The zip has no such limit."""


class ExportGuard:
    """One export per room, and only so many at once.

    An export reads every JPEG a room has and re-encodes most of them. That is
    a lot of disk and CPU for a request anybody can make, and the obvious way
    to hurt this server is to ask for the same one ten times. So: a room builds
    one export at a time, and the process builds a handful.
    """

    def __init__(self, limit: int = 2) -> None:
        self.limit = max(1, int(limit))
        self._lock = threading.Lock()
        self._rooms: set = set()

    @contextmanager
    def claim(self, room_id: str) -> Iterator[None]:
        with self._lock:
            if room_id in self._rooms or len(self._rooms) >= self.limit:
                raise ExportBusy("an export is already being built")
            self._rooms.add(room_id)
        try:
            yield
        finally:
            with self._lock:
                self._rooms.discard(room_id)

    @property
    def running(self) -> int:
        with self._lock:
            return len(self._rooms)


def temp_path(store: HistoryStore, room_id: str, suffix: str) -> Path:
    """A private file to build into, beside the data it is built from."""
    directory = store.root / EXPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{room_id}-{os.getpid()}-{time.time_ns()}{suffix}"


def frame_name(prefix: str, n: int) -> str:
    """`draw_00007.jpg` - numbered by the entry, not by its position, so a
    name in the zip and a number in the gallery are the same thing."""
    return f"{prefix}_{n:05d}.jpg"


# --------------------------------------------------------------------- zip


def build_zip(
    store: HistoryStore,
    room_id: str,
    dest: Path,
    *,
    canvas_size: int,
) -> int:
    """Every frame of a room, oldest first, plus a manifest. Returns the count.

    ZIP_STORED, not deflate: these are JPEGs. Deflating them costs the whole
    export's CPU again and wins a fraction of a percent.
    """
    entries = store.entries(room_id)
    if not entries:
        raise ExportEmpty(room_id)
    frames: List[Dict[str, Any]] = []
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as archive:
        for item in entries:
            n = item["n"]
            entry = item["entry"]
            draw_name: Optional[str] = None
            if item["input"] is not None:
                try:
                    archive.write(item["input"], frame_name("draw", n))
                    draw_name = frame_name("draw", n)
                except OSError:  # pragma: no cover - vanished mid-export
                    draw_name = None
            try:
                archive.write(item["gen"], frame_name("gen", n))
            except OSError:  # pragma: no cover - vanished mid-export
                continue
            frames.append(
                {
                    "n": n,
                    "draw": draw_name,
                    "gen": frame_name("gen", n),
                    **{
                        key: entry.get(key)
                        for key in (
                            "time",
                            "prompt",
                            "negativePrompt",
                            "denoise",
                            "seed",
                            "profile",
                            "aiResolution",
                            "latencyMs",
                            "aiRevision",
                            "aiGeneration",
                        )
                    },
                }
            )
        if not frames:  # pragma: no cover - every file vanished mid-export
            raise ExportEmpty(room_id)
        manifest = {
            "roomId": room_id,
            "canvasSize": int(canvas_size),
            "exportedAt": int(time.time() * 1000),
            "frames": frames,
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    return len(frames)


# --------------------------------------------------------------------- avi


def _half_size(gen: Path, width: Optional[int]) -> Tuple[int, int]:
    """Half a frame: the stored size, or the asked-for width split in two.

    The aspect ratio comes from the stored result rather than being assumed
    square, so a non-square canvas is scaled rather than stretched.
    """
    with Image.open(gen) as image:
        stored_w, stored_h = image.size
    if width is None:
        half_w, half_h = stored_w, stored_h
    else:
        half_w = max(1, int(width) // 2)
        half_h = max(1, round(half_w * stored_h / stored_w))
    # Even on both axes: MJPEG's chroma subsampling halves each dimension, and
    # an odd one is where a decoder that rounds the other way puts a green line.
    return half_w + (half_w & 1), half_h + (half_h & 1)


def build_avi(
    store: HistoryStore,
    room_id: str,
    dest: Path,
    *,
    fps: int = DEFAULT_FPS,
    width: Optional[int] = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Tuple[int, int, int]:
    """One frame per entry: input left, result right. Returns (frames, w, h)."""
    entries = store.entries(room_id)
    if not entries:
        raise ExportEmpty(room_id)
    if len(entries) > max_frames:
        raise ExportTooLarge(
            f"{len(entries)} entries is more than the {max_frames} this can make a "
            "video of; the zip has every frame"
        )
    fps = max(MIN_FPS, min(int(fps), MAX_FPS))
    half_w, half_h = _half_size(entries[0]["gen"], width)
    frame_size = (half_w * 2, half_h)
    written = 0
    with open(dest, "wb") as handle:
        writer = MjpegAviWriter(handle, frame_size[0], frame_size[1], fps)
        for item in entries:
            frame = _compose(item, half_w, half_h)
            if frame is None:
                continue
            writer.add_frame(frame)
            written += 1
        if written == 0:  # pragma: no cover - every file vanished mid-export
            raise ExportEmpty(room_id)
        writer.close()
    return written, frame_size[0], frame_size[1]


def _compose(item: Dict[str, Any], half_w: int, half_h: int) -> Optional[bytes]:
    """One side-by-side frame, or None when the result cannot be read.

    White, not black, for the missing half: it is the same background the AI
    input is composited on, so an entry with no stored input reads as an empty
    canvas rather than as a hole in the video.
    """
    canvas = Image.new("RGB", (half_w * 2, half_h), (255, 255, 255))
    try:
        _paste(canvas, item["gen"], half_w, half_w, half_h)
    except Exception:
        log.warning("export: skipping entry %s", item.get("n"), exc_info=True)
        return None
    if item["input"] is not None:
        try:
            _paste(canvas, item["input"], 0, half_w, half_h)
        except Exception:
            # A frame with a white left half is still a frame; a failed export
            # over one unreadable input would not be.
            log.warning("export: entry %s has no usable input", item.get("n"), exc_info=True)
    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=VIDEO_QUALITY)
    return out.getvalue()


def _paste(canvas: Image.Image, path: Path, left: int, half_w: int, half_h: int) -> None:
    with Image.open(path) as image:
        image.load()
        piece = image.convert("RGB")
        if piece.size != (half_w, half_h):
            piece = piece.resize((half_w, half_h), Image.LANCZOS)
        canvas.paste(piece, (left, 0))
