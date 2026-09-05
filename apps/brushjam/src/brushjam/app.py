"""HTTP routes, the WebSocket endpoint and the static client.
Port of apps/server/src/server.ts.

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
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

from .ai.backends import AIBackend, MockBackend
from .config import Config
from .constants import AI_RESOLUTIONS
from .raster import build_full_mask
from .room import RoomLimits
from .runtime import MAX_BUFFERED_BYTES, RoomRegistry

log = logging.getLogger("brushjam.http")

ROOM_ID = re.compile(r"^[a-z0-9]{4,16}$")
SESSION_TOKEN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
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


class SocketConnection:
    """A WebSocket plus an outbound queue, so a slow reader buffers in one place
    and can be cut off once it buffers too much."""

    def __init__(self, socket: WebSocket) -> None:
        self.socket = socket
        self._queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        self._buffered = 0
        self._open = True
        self._revoked = False
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
        self._buffered += len(data)
        self._queue.put_nowait(data)

    def close_now(self) -> None:
        if not self._open:
            return
        self._open = False
        self._queue.put_nowait(None)

    def revoke(self) -> None:
        """Cut this socket off *now*, without draining what is queued.

        `close_now` puts a sentinel behind every pending frame, so a slow reader
        keeps its socket - and its identity - for as long as its backlog takes
        to flush. A superseded connection has to lose both immediately, or two
        sockets act as the same user (Node called `terminate()` here).
        """
        self._open = False
        self._revoked = True
        self._buffered = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - drained concurrently
                break
        if self._writer is not None:
            self._writer.cancel()
        # The transport close cannot be awaited from here; it is the last thing
        # the cancelled writer does, and this covers the case where it is
        # already finished.
        asyncio.ensure_future(self._close_transport())

    async def _close_transport(self) -> None:
        try:
            await self.socket.close(code=1012)
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
            try:
                await self.socket.close()
            except Exception:
                pass

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
        # room slots, so it is rate limited per client address.
        client = request.client.host if request.client else "unknown"
        if not registry.allow_create(client):
            return JSONResponse(
                {"error": "too many rooms created from here; try again shortly"},
                status_code=429,
                headers={"retry-after": "60"},
            )
        room = registry.create()
        if room is None:
            return JSONResponse(
                {"error": "the server is holding too many rooms right now"}, status_code=429
            )
        return JSONResponse({"roomId": room.state.id})

    @app.get("/healthz")
    async def healthz() -> Response:
        # `ok`, `backend` and `rooms` are what the TS tooling reads
        # (apps/server/scripts/latency.ts). A resident backend adds the fields
        # that tooling used to fetch from the stream worker's own /healthz -
        # steps, guidance, vae, model, lora - so there is one place to look.
        body: Dict[str, Any] = {"ok": True, "backend": backend.name, "rooms": registry.size}
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
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_IMAGE_BYTES:
                return JSONResponse({"error": "image too large"}, status_code=413)
        target = registry.ensure(room_id)
        if target is None:
            return JSONResponse(
                {"error": "the server is holding too many rooms right now"}, status_code=429
            )
        stored = await target.add_image(bytes(body), mime)
        if "error" in stored:
            return JSONResponse(stored, status_code=400)
        return JSONResponse(stored)

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
        room = registry.ensure(room_id)
        if room is None:
            await socket.accept()
            await socket.close(code=1013, reason="server is holding too many rooms")
            return
        # Capacity is decided before anything is constructed or joined, so a
        # refusal allocates nothing and leaves membership untouched.
        refusal = registry.can_accept_socket(room)
        if refusal is not None:
            await socket.accept()
            await socket.close(code=1013, reason=refusal)
            return
        await socket.accept()
        raw_token = socket.query_params.get("token") or ""
        token = raw_token if SESSION_TOKEN.match(raw_token) else None
        connection = SocketConnection(socket)
        registry.note_socket_open()
        user_id = await room.join_async(
            connection, socket.query_params.get("name") or "", token
        )
        try:
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
        finally:
            registry.note_socket_closed()
            room.leave(user_id, connection)
            await connection.aclose()

    if has_web:

        @app.get("/{path:path}")
        async def static_files(path: str) -> Response:
            rel = "index.html" if path in ("", "/") else path
            candidate = (web_dist / rel).resolve()
            if str(candidate).startswith(str(web_dist)) and candidate.is_file():
                media = MIME.get(candidate.suffix) or mimetypes.guess_type(candidate.name)[0] or (
                    "application/octet-stream"
                )
                return FileResponse(candidate, media_type=media)
            # SPA fallback: /r/<id> is a client route, not a file.
            return FileResponse(web_dist / "index.html", media_type=MIME[".html"])

    return app
