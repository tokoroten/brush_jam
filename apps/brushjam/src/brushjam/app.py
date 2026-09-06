"""HTTP routes, the WebSocket endpoint and the static client.
Port of the retired Node server's src/server.ts.

One process serves all of it, so a room's frames, its result PNGs and the page
that draws them all come from the same origin and the same port.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

from .ai.backends import AIBackend, MockBackend
from .config import Config
from .constants import AI_RESOLUTIONS
from .export import (
    DEFAULT_FPS,
    MAX_FPS,
    MAX_WIDTH,
    MIN_FPS,
    MIN_WIDTH,
    ExportBusy,
    ExportEmpty,
    ExportGuard,
    ExportTooLarge,
    build_avi,
    build_zip,
    temp_path,
)
from .history import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT
from .raster import build_full_mask
from .room import RoomLimits
from .runtime import MAX_BUFFERED_BYTES, RoomRegistry

log = logging.getLogger("brushjam.http")

ROOM_ID = re.compile(r"^[a-z0-9]{4,16}$")
SESSION_TOKEN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
#: A saved result is `<n>.jpg`, and the human canvas it was generated from is
#: `<n>.in.jpg`; `n` is what the store numbered it. Matched rather than
#: sanitised: the name becomes a path.
HISTORY_IMAGE = re.compile(r"^(\d{1,9})(\.in)?\.jpg$")
MAX_IMAGE_BYTES = 12 * 1024 * 1024
#: One frame is never legitimately larger than this.
MAX_WS_PAYLOAD = 1024 * 1024
IMAGE_MIME = re.compile(r"^image/(png|jpeg|webp)$")

#: How often each socket is pinged, and how many silences end it. uvicorn's
#: websocket layer runs the ping loop, so these are passed to it rather than
#: being timed here (the Node server had to do it by hand).
HEARTBEAT_MS = 30_000
HEARTBEAT_MISSES = 2

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
}


def _remove(path: Path) -> None:
    """Delete a finished (or abandoned) export. Never raises: a temp file that
    will not go is a log line, not a failed download."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as err:  # pragma: no cover - a reader still holding it
        log.warning("could not remove the export %s (%s)", path, err)


class SocketConnection:
    """A WebSocket plus an outbound queue, so a slow reader buffers in one place
    and can be cut off once it buffers too much."""

    def __init__(self, socket: WebSocket) -> None:
        self.socket = socket
        self._queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        self._buffered = 0
        self._open = True
        self._revoked = False
        #: The close code the transport is to go away with. One owner closes
        #: the socket - the writer's cleanup - so this is how the reason
        #: travels to it. None means an ordinary close (1000).
        self._close_code: Optional[int] = None
        #: While held, frames are kept aside rather than queued. A join needs
        #: this: its snapshot is serialised off the loop, and anything
        #: broadcast during that await has to arrive *after* the snapshot, not
        #: instead of it.
        self._held = True
        self._buffer: "list[str]" = []
        self._writer = asyncio.ensure_future(self._pump())

    @property
    def open(self) -> bool:
        return self._open

    @property
    def buffered_bytes(self) -> int:
        return self._buffered

    def send_text(self, data: str) -> None:
        if not self._open:
            return
        if self._held:
            self._buffered += len(data)
            self._buffer.append(data)
            return
        self._buffered += len(data)
        self._queue.put_nowait(data)

    def release(self, first: Optional[str] = None) -> None:
        """Stop holding: `first` goes out ahead of everything buffered.

        That ordering is the whole point. `first` is the join snapshot, which
        the protocol says arrives before any event; the buffer is what was
        broadcast while it was being built, which must arrive after it rather
        than be dropped.
        """
        if not self._held:
            if first is not None:
                self.send_text(first)
            return
        self._held = False
        pending, self._buffer = self._buffer, []
        if not self._open:
            self._buffered -= sum(len(item) for item in pending)
            return
        if first is not None:
            self._buffered += len(first)
            self._queue.put_nowait(first)
        for item in pending:
            self._queue.put_nowait(item)

    @property
    def held(self) -> bool:
        return self._held

    def close_now(self) -> None:
        if not self._open:
            return
        self._open = False
        self._queue.put_nowait(None)

    def discard_held(self) -> None:
        """Drop what was buffered for a join that is not going to happen."""
        self._buffered -= sum(len(item) for item in self._buffer)
        self._buffer = []

    def revoke(self, code: int = 1012) -> None:
        """Cut this socket off *now*, without draining what is queued.

        `close_now` puts a sentinel behind every pending frame, so a slow reader
        keeps its socket - and its identity - for as long as its backlog takes
        to flush. A superseded connection has to lose both immediately, or two
        sockets act as the same user (Node called `terminate()` here).
        """
        self._open = False
        self._revoked = True
        self._held = False
        self._buffer = []
        self._buffered = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - drained concurrently
                break
        # The code is published BEFORE the writer is cancelled, because the
        # writer's cleanup is what closes the transport. Racing it with a
        # second close from here is how a superseded socket used to go away
        # with 1000: the cancelled writer's `finally` got there first, the
        # client saw an ordinary close, reconnected, and evicted the tab that
        # had just evicted it.
        self._close_code = code
        if self._writer is not None and not self._writer.done():
            self._writer.cancel()
            return
        # The writer has already finished (or there never was one), so nobody
        # else is going to close the transport.
        asyncio.ensure_future(self._close_transport(code))

    async def _close_transport(self, code: Optional[int] = None) -> None:
        try:
            await self.socket.close(code=1000 if code is None else code)
        except Exception:
            pass

    @property
    def revoked(self) -> bool:
        return self._revoked

    async def _pump(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                try:
                    await self.socket.send_text(item)
                except Exception:
                    break
                finally:
                    self._buffered -= len(item)
        except asyncio.CancelledError:
            pass
        finally:
            self._open = False
            # The one place the transport is closed, with whatever reason
            # `revoke` left behind (1000 for an ordinary close).
            await self._close_transport(self._close_code)

    async def aclose(self) -> None:
        self.close_now()
        if self._writer is not None:
            try:
                await asyncio.wait_for(self._writer, 2)
            except Exception:
                self._writer.cancel()


def prebuild_full_masks(config: Config) -> None:
    """Every generation size a room can ask for, built once at startup."""
    sizes = {config.ai_window, config.canvas_size}
    sizes.update(s for s in AI_RESOLUTIONS if s <= config.canvas_size)
    if config.max_resolution:
        sizes.add(config.max_resolution)
    for size in sorted(sizes):
        if size > 0:
            build_full_mask(int(size))


def _client_of(scope: Any) -> str:
    """The address a request or a WebSocket came from, for rate limiting."""
    client = getattr(scope, "client", None)
    return getattr(client, "host", None) or "unknown"


def _contained(candidate: Path, root: Path) -> bool:
    try:
        return candidate == root or candidate.relative_to(root) is not None
    except ValueError:
        return False


def default_web_dist() -> Path:
    """The built client: `static/` inside the package (what scripts/build_web.py
    fills), else the repo's apps/web/dist for a dev checkout."""
    packaged = Path(__file__).resolve().parent / "static"
    if (packaged / "index.html").exists():
        return packaged
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def create_app(
    config: Config, backend: Optional[AIBackend] = None, limits: Optional[RoomLimits] = None
) -> FastAPI:
    backend = backend or MockBackend()
    registry = RoomRegistry(backend, config, limits)
    #: One export per room and a couple in the process: an export reads and
    #: re-encodes everything a room ever made, and it is an unauthenticated GET.
    exports = ExportGuard()
    web_dist = Path(config.web_dist).resolve() if config.web_dist else default_web_dist()
    has_web = (web_dist / "index.html").exists()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry.start_sweeper()
        registry.start_capability_watch()
        # A resident model takes ~40 s to load. Do it now, in the background, so
        # the server answers /healthz while it happens and the first person to
        # draw does not pay for it.
        loader = None
        if config.inproc_preload and hasattr(backend, "load"):
            loader = asyncio.ensure_future(backend.load())
        # Build the full-canvas masks now, off the loop: the first generation at
        # a size would otherwise PNG-encode one while every room's sockets wait.
        await asyncio.to_thread(prebuild_full_masks, config)
        try:
            yield
        finally:
            if loader is not None:
                loader.cancel()
            registry.dispose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.registry = registry
    app.state.exports = exports
    app.state.config = config
    app.state.backend = backend
    app.state.web_dist = web_dist

    def png(body: bytes, cache_seconds: int = 0) -> Response:
        return Response(
            content=body,
            media_type="image/png",
            headers={
                "cache-control": f"public, max-age={cache_seconds}"
                if cache_seconds > 0
                else "no-store"
            },
        )

    @app.post("/api/rooms")
    async def create_room_route(request: Request) -> Response:
        # Unauthenticated, and each success reserves one of a small number of
        # room slots. `create` rate limits per client address, as every path
        # that can bring a room into existence does.
        room = registry.create(_client_of(request))
        if room is None:
            return JSONResponse(
                {"error": "no room right now; too many rooms, or too many from here"},
                status_code=429,
                headers={"retry-after": "60"},
            )
        return JSONResponse({"roomId": room.state.id})

    @app.get("/healthz")
    async def healthz() -> Response:
        # `ok`, `backend` and `rooms` are what the TS tooling reads
        # (the retired Node server's scripts/latency.ts). A resident backend adds the fields
        # that tooling used to fetch from the stream worker's own /healthz -
        # steps, guidance, vae, model, lora - so there is one place to look.
        # `active_sockets` and `last_generation_at` are what an outside watcher
        # needs to tell a busy server from an idle one - deploy/runpod's
        # `watch` stops the pod on them - and neither can be inferred from
        # `rooms`, which counts rooms nobody is in yet as well.
        body: Dict[str, Any] = {
            "ok": True,
            "backend": backend.name,
            "rooms": registry.size,
            "active_sockets": registry.active_sockets,
            "last_generation_at": registry.last_generation_at,
            "generations": registry.generations,
        }
        status = getattr(backend, "status", None)
        if callable(status):
            extra = status()
            body.update(extra)
            # The room server's own name, not the pipeline's internal one.
            body["backend"] = backend.name
            body["ok"] = extra.get("error") is None
        return JSONResponse(body)

    @app.get("/rooms/{room_id}/ai.png")
    async def ai_png(room_id: str) -> Response:
        if not ROOM_ID.match(room_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        room = registry.get(room_id)
        if room is None or not room.has_ai():
            return JSONResponse({"error": "no AI output yet"}, status_code=404)
        return png(await room.ai_png())

    @app.get("/rooms/{room_id}/history")
    async def room_history(room_id: str, limit: int = DEFAULT_LIST_LIMIT) -> Response:
        # No registry lookup on purpose: what a room made outlives the room, so
        # a gallery still opens after the room has been evicted or the server
        # restarted. The id is validated here because it becomes a directory.
        if not ROOM_ID.match(room_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            wanted = max(1, min(int(limit), MAX_LIST_LIMIT))
        except (TypeError, ValueError):
            wanted = DEFAULT_LIST_LIMIT
        entries = await asyncio.to_thread(registry.history.list, room_id, wanted)
        return JSONResponse(
            {"roomId": room_id, "enabled": registry.history.enabled, "entries": entries},
            headers={"cache-control": "no-store"},
        )

    @app.get("/rooms/{room_id}/history/{name}")
    async def room_history_image(room_id: str, name: str) -> Response:
        matched = HISTORY_IMAGE.match(name)
        if not ROOM_ID.match(room_id) or matched is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        body = await asyncio.to_thread(
            registry.history.read,
            room_id,
            int(matched.group(1)),
            "in" if matched.group(2) else "jpg",
        )
        if body is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return Response(
            content=body,
            media_type="image/jpeg",
            # Immutable: entry n of a room is written once and never rewritten.
            headers={"cache-control": "public, max-age=86400, immutable"},
        )

    async def _export(room_id: str, suffix: str, build, media_type: str) -> Response:
        """The shared half of both exports: validate, guard, build, hand over.

        The build runs on a worker thread and writes a temp file, which the
        response streams and a background task deletes. Nothing is held in
        memory, and nothing survives the download.
        """
        if not ROOM_ID.match(room_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        if not registry.history.enabled:
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            with exports.claim(room_id):
                dest = temp_path(registry.history, room_id, suffix)
                try:
                    await asyncio.to_thread(build, dest)
                except BaseException:
                    # Including a cancelled request: the caller went away, and
                    # the half-built file must not stay behind.
                    _remove(dest)
                    raise
        except ExportBusy as busy:
            return JSONResponse(
                {"error": "an export of this room is already being built"},
                status_code=429,
                headers={"retry-after": str(busy.retry_after), "cache-control": "no-store"},
            )
        except ExportEmpty:
            return JSONResponse({"error": "nothing saved for this room"}, status_code=404)
        except ExportTooLarge as big:
            return JSONResponse({"error": str(big)}, status_code=413)
        except Exception:
            log.warning("[room %s] export failed", room_id, exc_info=True)
            return JSONResponse({"error": "could not build the export"}, status_code=500)
        return FileResponse(
            dest,
            media_type=media_type,
            headers={
                "content-disposition": (
                    f'attachment; filename="brushjam-{room_id}-history{suffix}"'
                ),
                "cache-control": "no-store",
            },
            background=BackgroundTask(_remove, dest),
        )

    @app.get("/rooms/{room_id}/history.zip")
    async def room_history_zip(room_id: str) -> Response:
        return await _export(
            room_id,
            ".zip",
            lambda dest: build_zip(
                registry.history, room_id, dest, canvas_size=config.canvas_size
            ),
            "application/zip",
        )

    @app.get("/rooms/{room_id}/history.avi")
    async def room_history_avi(
        room_id: str, fps: int = DEFAULT_FPS, width: Optional[int] = None
    ) -> Response:
        # Clamped rather than refused: these are the two knobs on a download
        # link, and a link with a silly number in it should still give you the
        # video rather than an error page.
        try:
            rate = max(MIN_FPS, min(int(fps), MAX_FPS))
        except (TypeError, ValueError):
            rate = DEFAULT_FPS
        target: Optional[int] = None
        if width is not None:
            try:
                target = max(MIN_WIDTH, min(int(width), MAX_WIDTH))
            except (TypeError, ValueError):
                target = None
        return await _export(
            room_id,
            ".avi",
            lambda dest: build_avi(
                registry.history,
                room_id,
                dest,
                fps=rate,
                width=target,
                max_frames=config.history_export_max_frames,
            ),
            "video/x-msvideo",
        )

    @app.get("/rooms/{room_id}/patches/{patch}")
    async def patch_png(room_id: str, patch: str) -> Response:
        if not ROOM_ID.match(room_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        room = registry.get(room_id)
        body = room.patch(re.sub(r"\.png$", "", patch)) if room else None
        if body is None:
            return JSONResponse({"error": "patch not found"}, status_code=404)
        return png(body, 300)

    @app.post("/rooms/{room_id}/images")
    async def upload_image(room_id: str, request: Request) -> Response:
        if not ROOM_ID.match(room_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        mime = request.headers.get("content-type") or "image/png"
        # Validate before touching the registry: `ensure` would allocate a room.
        if not IMAGE_MIME.match(mime):
            return JSONResponse({"error": "unsupported image type"}, status_code=415)
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_IMAGE_BYTES:
            return JSONResponse({"error": "image too large"}, status_code=413)
        # Admission BEFORE the body is read: a 12 MiB bytearray is accumulated
        # here before the room limiter or the image store has any say, so this
        # is what bounds a spike from modest unauthenticated concurrency. The
        # reservation is for the largest the body may be, because
        # content-length is optional and not to be trusted.
        client = _client_of(request)
        slot = registry.uploads.acquire(client, MAX_IMAGE_BYTES)  # == MAX_IMAGE_BYTES_PER_SLOT
        if slot is None:
            return JSONResponse(
                {"error": "too many uploads in flight; try again shortly"},
                status_code=429,
                headers={"retry-after": "5"},
            )
        try:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_IMAGE_BYTES:
                    return JSONResponse({"error": "image too large"}, status_code=413)
            # An upload to an id nobody has opened yet creates the room, so it
            # is rate limited exactly like a POST to /api/rooms.
            target = registry.get_or_create(room_id, client)
            if target is None:
                return JSONResponse(
                    {"error": "no room right now; too many rooms, or too many from here"},
                    status_code=429,
                    headers={"retry-after": "60"},
                )
            stored = await target.add_image(bytes(body), mime)
            if "error" in stored:
                return JSONResponse(stored, status_code=400)
            return JSONResponse(stored)
        finally:
            slot.release()

    @app.get("/rooms/{room_id}/images/{image_id}")
    async def get_image(room_id: str, image_id: str) -> Response:
        if not ROOM_ID.match(room_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        room = registry.get(room_id)
        stored = room.state.images.get(image_id) if room else None
        if stored is None:
            return JSONResponse({"error": "image not found"}, status_code=404)
        return Response(
            content=stored.data,
            media_type=stored.mime,
            headers={"cache-control": "public, max-age=3600"},
        )

    @app.websocket("/ws/rooms/{room_id}")
    async def room_socket(socket: WebSocket, room_id: str) -> None:
        if not ROOM_ID.match(room_id):
            await socket.close(code=1008)
            return
        raw_token = socket.query_params.get("token") or ""
        token = raw_token if SESSION_TOKEN.match(raw_token) else None

        room = registry.get(room_id)
        if room is None:
            # A link to a room that does not exist yet creates it - at the same
            # rate as any other creation, and *before* a socket is accepted, so
            # a refused connection cannot have spent a room slot.
            room = registry.create_named(room_id, _client_of(socket))
            if room is None:
                await socket.accept()
                await socket.close(code=1013, reason="no room right now")
                return

        # One synchronous check-and-take, before the first await: two
        # handshakes that both passed a check before either incremented would
        # both be admitted. A reconnect takes over the lease of the socket it
        # replaces rather than opening a second one.
        reservation = registry.reserve_socket(room, token)
        if reservation is None:
            await socket.accept()
            await socket.close(code=1013, reason=registry.can_accept_socket(room) or "full")
            return

        connection: Optional[SocketConnection] = None
        user_id: Optional[str] = None
        try:
            await socket.accept()
            connection = SocketConnection(socket)
            user_id = await room.join_async(
                connection, socket.query_params.get("name") or "", token, reservation
            )
            while True:
                data = await socket.receive_text()
                if len(data) > MAX_WS_PAYLOAD:
                    await socket.close(code=1009)
                    break
                room.handle(user_id, data, connection)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.debug("[room %s] socket error", room_id, exc_info=True)
            # A join that never finished must not leave a member nobody can
            # reach, or a slot nobody will give back.
            if connection is not None:
                room.rollback_join(user_id, connection)
                user_id = None
        finally:
            reservation.release()
            if user_id is not None and connection is not None:
                room.leave(user_id, connection)
            if connection is not None:
                await connection.aclose()

    if has_web:

        @app.get("/{path:path}")
        async def static_files(path: str) -> Response:
            rel = "index.html" if path in ("", "/") else path
            candidate = (web_dist / rel).resolve()
            # Containment by resolved path, not by string prefix: a sibling
            # directory whose name merely starts with web_dist's would pass a
            # `startswith` check.
            if _contained(candidate, web_dist) and candidate.is_file():
                media = MIME.get(candidate.suffix) or mimetypes.guess_type(candidate.name)[0] or (
                    "application/octet-stream"
                )
                return FileResponse(candidate, media_type=media)
            # SPA fallback: /r/<id> is a client route, not a file.
            return FileResponse(web_dist / "index.html", media_type=MIME[".html"])

    return app
