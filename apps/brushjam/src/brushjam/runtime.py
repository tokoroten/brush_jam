"""Room runtime: sockets, presence, broadcast, image store, eviction.
Port of apps/server/src/runtime.ts.

A `Connection` is anything that can be handed a JSON string and told to go away,
so the tests can drive a room without a real socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Protocol

from .ai.backends import AIBackend
from .config import Config
from .geometry import Rect, intersect_rect
from .ids import short_id
from .imageinfo import check_image
from .protocol import Message
from .raster import AICanvas, build_full_mask, decode_upload, forget_images, render_crop_input
from .room import (
    ApplyResult,
    RoomImage,
    RoomLimits,
    RoomState,
    apply_client_message,
    capture_render_snapshot,
    create_room,
    expire_pending_strokes,
    join_member,
    now_ms,
    remove_member,
    snapshot,
)
from .scheduler import AIScheduler, RenderJob, SchedulerOptions
from .validate import validate_client_message

log = logging.getLogger("brushjam.runtime")

MAX_PATCHES = 24
#: Full-canvas results are whole-canvas PNGs, so only the current one and one
#: predecessor (for a client still fetching the previous URL) are worth holding.
MAX_PATCHES_FULL = 2
#: Every live room holds rasters and uploads; this is a hard ceiling.
MAX_ROOMS = 64
#: A slow client is dropped rather than allowed to buffer without limit.
MAX_BUFFERED_BYTES = 8 * 1024 * 1024
MAX_IMAGES_PER_ROOM = 32
#: An uploaded image gets this long to become a layer before it is swept.
IMAGE_GRACE_MS = 2 * 60_000
MAX_IMAGE_BYTES_PER_ROOM = 64 * 1024 * 1024
#: How often idle capabilities are re-checked; a restarted worker is silent.
CAPABILITY_POLL_MS = 60_000


class Connection(Protocol):
    """The socket-shaped surface the runtime needs."""

    def send_text(self, data: str) -> None:
        ...

    def close_now(self) -> None:
        ...

    @property
    def buffered_bytes(self) -> int:
        ...

    @property
    def open(self) -> bool:
        ...


class RoomRuntime:
    def __init__(
        self,
        room_id: str,
        backend: AIBackend,
        config: Config,
        limits: Optional[RoomLimits] = None,
    ) -> None:
        self.config = config
        self.state: RoomState = create_room(
            room_id,
            config.ai_denoise,
            config.canvas_size,
            config.ai_window,
            config.ai_mode == "full",
            config.ai_profile,
            limits,
        )
        #: Allocated on the first accepted AI result: a full-size raster is big.
        self._ai: Optional[AICanvas] = None
        self._sockets: "Dict[str, Connection]" = {}
        self._patches: "Dict[str, bytes]" = {}
        self._image_bytes = 0
        #: Uploads past their limit check but still decoding, so they count too.
        self._pending_images = 0
        self._pending_image_bytes = 0
        #: Set by the registry so a failure can trigger a capability re-probe.
        self.on_error: Optional[Callable[[str, int], None]] = None

        runtime = self

        class _Host:
            def get_revision(self) -> int:
                return runtime.state.human_revision

            def begin_job(self) -> RenderJob:
                # Captured synchronously: the render must not observe later strokes.
                snap = capture_render_snapshot(runtime.state)
                return RenderJob(
                    revision=snap.revision,
                    prompt=snap.prompt,
                    denoise=snap.denoise,
                    negative_prompt=snap.negative_prompt,
                    resolution=snap.ai_resolution,
                    profile=snap.ai_profile,
                    render=lambda crop, size: render_crop_input(snap, crop, size),
                )

            def build_full_mask(self, size: int):
                return build_full_mask(size)

            async def apply_result(self, patch, crop, apply, mask, for_revision):
                png = await asyncio.to_thread(runtime._ai_canvas().composite, patch, crop, mask)
                patch_id = short_id(10)
                runtime._patches[patch_id] = png
                keep = MAX_PATCHES_FULL if config.ai_mode == "full" else MAX_PATCHES
                while len(runtime._patches) > keep:
                    oldest = next(iter(runtime._patches))
                    del runtime._patches[oldest]
                runtime.state.ai_revision = for_revision
                # Counted separately from the revision: a settings-triggered run
                # in an untouched room lands at revision 0, and a late joiner has
                # to tell that from "nothing has ever been generated".
                runtime.state.ai_generation += 1
                return {
                    "rect": crop,
                    "url": f"/rooms/{room_id}/patches/{patch_id}.png",
                    "aiGeneration": runtime.state.ai_generation,
                }

            def emit(self, msg: Message) -> None:
                runtime.broadcast(msg)

            def on_error(self, message: str, repeated: int) -> None:
                if runtime.on_error:
                    runtime.on_error(message, repeated)

        self.scheduler = AIScheduler(
            _Host(),
            backend,
            SchedulerOptions(
                window=config.ai_window,
                apply=config.ai_apply,
                steps=config.ai_steps,
                fast_steps=config.ai_fast_steps,
                denoise=config.ai_denoise,
                debounce_ms=config.ai_debounce_ms,
                canvas_size=config.canvas_size,
                watchdog_ms=config.ai_watchdog_ms,
                tag=room_id,
            ),
        )

    # -- ai raster --------------------------------------------------------

    def _ai_canvas(self) -> AICanvas:
        if self._ai is None:
            self._ai = AICanvas(self.config.canvas_size)
        return self._ai

    @property
    def patch_count(self) -> int:
        return len(self._patches)

    @property
    def member_count(self) -> int:
        return len(self._sockets)

    def has_ai(self) -> bool:
        """False until the first accepted result, so a GET cannot force the
        allocation."""
        return self._ai is not None

    def ai_png(self) -> bytes:
        return self._ai_canvas().to_png()

    def patch(self, patch_id: str) -> Optional[bytes]:
        return self._patches.get(patch_id)

    # -- membership -------------------------------------------------------

    def join(self, socket: Connection, name: str, token: Optional[str] = None) -> str:
        member = join_member(self.state, name, token)
        previous = self._sockets.get(member["userId"])
        if previous is not None and previous is not socket:
            # Same identity resumed while the old socket still looked alive: the
            # newest wins.
            del self._sockets[member["userId"]]
            try:
                previous.close_now()
            except Exception:
                pass  # already gone
        self._sockets[member["userId"]] = socket
        self.send(
            member["userId"],
            {
                "t": "snapshot",
                "snapshot": snapshot(
                    self.state,
                    member["userId"],
                    self.scheduler.state,
                    {
                        "window": self.config.ai_window,
                        "apply": self.config.ai_apply,
                        "canvasSize": self.config.canvas_size,
                    },
                ),
            },
        )
        self.broadcast({"t": "presence", "members": list(self.state.members.values())})
        return member["userId"]

    def leave(self, user_id: str, socket: Optional[Connection] = None) -> None:
        """`socket` identifies *which* connection closed: a superseded socket
        fires its close after the replacement has joined."""
        current = self._sockets.get(user_id)
        if socket is not None and current is not None and current is not socket:
            return
        for cancel in remove_member(self.state, user_id):
            self.broadcast(cancel)
        self._sockets.pop(user_id, None)
        self.broadcast({"t": "presence", "members": list(self.state.members.values())})

    # -- messages ---------------------------------------------------------

    def handle(self, user_id: str, raw: str) -> None:
        try:
            parsed = json.loads(raw)
        except Exception:
            self.send(user_id, {"t": "error", "message": "invalid json"})
            return
        validated = validate_client_message(parsed)
        if not validated.ok:
            self.send(user_id, {"t": "error", "message": validated.error})
            return
        # Belt and braces: a reducer bug must never take the process (and every
        # other room) down from inside a socket handler.
        try:
            result: ApplyResult = apply_client_message(self.state, user_id, validated.msg)
        except Exception:
            log.exception("[room %s] reducer error on %s", self.state.id, validated.msg["t"])
            self.send(user_id, {"t": "error", "message": "the server could not apply that action"})
            return
        for msg in result.broadcast:
            self.broadcast(msg)
        for msg in result.relay:
            self.relay(user_id, msg)
        for msg in result.to_sender:
            self.send(user_id, msg)
        if validated.msg["t"] in ("layer_delete", "layer_create"):
            self.prune_images()
        if result.dirty:
            self.scheduler.mark_dirty(self._within_canvas(result.dirty))
        if result.prompt_changed:
            self.scheduler.nudge()

    def _within_canvas(self, rects: List[Rect]) -> List[Rect]:
        """Dirty regions outside the canvas can never be generated; clip them."""
        canvas: Rect = {
            "x": 0,
            "y": 0,
            "width": self.config.canvas_size,
            "height": self.config.canvas_size,
        }
        clipped = [intersect_rect(r, canvas) for r in rects]
        return [r for r in clipped if r is not None]

    # -- images -----------------------------------------------------------

    async def add_image(self, data: bytes, mime: str) -> Dict[str, Any]:
        # The header check is only a cheap preflight against decompression bombs
        check = check_image(data, mime)
        if not check.ok:
            return {"error": check.error}
        # The quota is *reserved* before the await: concurrent uploads would
        # otherwise all measure the same pre-upload totals and all fit.
        if len(self.state.images) + self._pending_images >= MAX_IMAGES_PER_ROOM:
            return {"error": "this room already holds the maximum number of images"}
        if self._image_bytes + self._pending_image_bytes + len(data) > MAX_IMAGE_BYTES_PER_ROOM:
            return {"error": "this room has reached its image storage limit"}
        self._pending_images += 1
        self._pending_image_bytes += len(data)
        try:
            # ...so decode once here and confirm the file is what it claims.
            ok = await asyncio.to_thread(
                decode_upload, data, check.info.width, check.info.height
            )
            if not ok:
                return {"error": "image could not be decoded"}
            image_id = short_id(10)
            self.state.images[image_id] = RoomImage(
                id=image_id,
                mime=mime,
                data=data,
                width=check.info.width,
                height=check.info.height,
                created_at=now_ms(),
            )
            self._image_bytes += len(data)
            return {"imageId": image_id, "width": check.info.width, "height": check.info.height}
        finally:
            self._pending_images -= 1
            self._pending_image_bytes -= len(data)

    def prune_images(self, now: Optional[int] = None, grace_ms: int = IMAGE_GRACE_MS) -> int:
        """Forget images no layer references any more, after a grace period."""
        now = now_ms() if now is None else now
        referenced = {l.get("imageId") for l in self.state.layers if l.get("imageId")}
        drop = [
            image_id
            for image_id, image in self.state.images.items()
            if image_id not in referenced and now - image.created_at >= grace_ms
        ]
        for image_id in drop:
            self._image_bytes -= len(self.state.images[image_id].data)
            del self.state.images[image_id]
        forget_images(drop)
        return len(drop)

    # -- lifecycle --------------------------------------------------------

    def expire_strokes(self, now: Optional[int] = None) -> int:
        cancels = expire_pending_strokes(self.state, now)
        for cancel in cancels:
            self.broadcast(cancel)
        return len(cancels)

    def is_idle(self, now: int, idle_ms: int) -> bool:
        return len(self._sockets) == 0 and now - self.state.last_active_at > idle_ms

    def dispose(self) -> None:
        self.scheduler.stop()
        forget_images(list(self.state.images.keys()))
        self.state.images.clear()
        self._patches.clear()
        self._ai = None

    def retry_now(self) -> None:
        self.scheduler.retry_now()

    def apply_limits(self, limits: RoomLimits) -> None:
        """The backend's limits changed under us. Clamp the room into the new
        limits, tell everyone, and let the AI re-run at the corrected size."""
        profiles = list(limits.profiles) if limits.profiles else list(self.state.ai_profiles)
        max_denoise = (
            limits.max_denoise if limits.max_denoise is not None else self.state.max_denoise
        )
        max_resolution = (
            limits.max_resolution
            if limits.max_resolution is not None
            else self.state.ai_resolution_max
        )
        if limits.negative_prompt_active:
            self.state.negative_active = dict(limits.negative_prompt_active)
        before = (self.state.denoise, self.state.ai_resolution, self.state.ai_profile)

        self.state.ai_profiles = profiles
        self.state.max_denoise = max_denoise
        self.state.ai_resolution_max = int(max_resolution)
        self.state.denoise = min(self.state.denoise, max_denoise)
        self.state.ai_resolution = int(min(self.state.ai_resolution, max_resolution))
        if self.state.ai_profile not in profiles:
            self.state.ai_profile = profiles[0]

        changed = before != (self.state.denoise, self.state.ai_resolution, self.state.ai_profile)
        active = self.state.negative_active.get(self.state.ai_profile, True) is not False
        # Capabilities first: a client that clamps its own controls before the
        # new settings arrive shows a consistent panel either way round.
        self.broadcast(
            {
                "t": "ai_capabilities",
                "aiProfiles": list(self.state.ai_profiles),
                "maxDenoise": self.state.max_denoise,
                "aiResolutionMax": self.state.ai_resolution_max,
                "negativePromptActive": active,
            }
        )
        self.broadcast(
            {
                "t": "ai_settings_changed",
                "denoise": self.state.denoise,
                "negativePrompt": self.state.negative_prompt,
                "aiResolution": self.state.ai_resolution,
                "aiProfile": self.state.ai_profile,
                "negativePromptActive": active,
            }
        )
        # Retry once, at the size the backend will actually accept.
        if changed:
            self.scheduler.nudge()

    # -- transport --------------------------------------------------------

    def broadcast(self, msg: Message) -> None:
        data = json.dumps(msg, separators=(",", ":"))
        for user_id, socket in list(self._sockets.items()):
            self._try_send(user_id, socket, data)

    def relay(self, except_user_id: str, msg: Message) -> None:
        data = json.dumps(msg, separators=(",", ":"))
        for user_id, socket in list(self._sockets.items()):
            if user_id != except_user_id:
                self._try_send(user_id, socket, data)

    def send(self, user_id: str, msg: Message) -> None:
        socket = self._sockets.get(user_id)
        if socket is not None:
            self._try_send(user_id, socket, json.dumps(msg, separators=(",", ":")))

    def _try_send(self, user_id: str, socket: Connection, data: str) -> None:
        if not socket.open:
            return
        if socket.buffered_bytes > MAX_BUFFERED_BYTES:
            log.warning(
                "[room %s] dropping slow client %s (%d bytes buffered)",
                self.state.id,
                user_id,
                socket.buffered_bytes,
            )
            socket.close_now()
            return
        try:
            socket.send_text(data)
        except Exception:
            pass  # dropped client


def same_limits(a: RoomLimits, b: RoomLimits) -> bool:
    return (
        a.max_denoise == b.max_denoise
        and a.max_resolution == b.max_resolution
        and ",".join(a.profiles or []) == ",".join(b.profiles or [])
        and (a.negative_prompt_active or None) == (b.negative_prompt_active or None)
    )


import re as _re

_LIMIT_ERROR = _re.compile(r"out of range|too large|max_size|not supported|unsupported|400", _re.I)


def looks_like_limit_error(message: str) -> bool:
    """Worth a re-probe: the backend refused the request itself rather than
    failing to do it."""
    return bool(_LIMIT_ERROR.search(message))


class RoomRegistry:
    def __init__(
        self, backend: AIBackend, config: Config, limits: Optional[RoomLimits] = None
    ) -> None:
        self.backend = backend
        self.config = config
        self.limits = (limits or RoomLimits()).copy()
        self._rooms: "Dict[str, RoomRuntime]" = {}
        self._sweeper: Optional[asyncio.Task] = None
        self._capability_task: Optional[asyncio.Task] = None
        #: True once a probe has failed, until one answers again.
        self._probe_failed = False
        self._refreshing: Optional[asyncio.Future] = None

    @property
    def backend_limits(self) -> RoomLimits:
        return self.limits.copy()

    @property
    def size(self) -> int:
        return len(self._rooms)

    @property
    def at_capacity(self) -> bool:
        return len(self._rooms) >= MAX_ROOMS

    def get(self, room_id: str) -> Optional[RoomRuntime]:
        return self._rooms.get(room_id)

    def create(self) -> Optional[RoomRuntime]:
        if self.at_capacity and self.sweep() == 0 and self.at_capacity:
            return None
        room_id = short_id(8)
        while room_id in self._rooms:
            room_id = short_id(8)
        return self._make(room_id)

    def ensure(self, room_id: str) -> Optional[RoomRuntime]:
        """Rooms are created on demand so a shared URL always works."""
        existing = self._rooms.get(room_id)
        if existing is not None:
            return existing
        if self.at_capacity and self.sweep() == 0 and self.at_capacity:
            return None
        return self._make(room_id)

    def _make(self, room_id: str) -> RoomRuntime:
        room = RoomRuntime(room_id, self.backend, self.config, self.limits)
        # A failing generation is the first sign a worker changed under us.
        room.on_error = lambda message, repeated: (
            asyncio.ensure_future(self.refresh_capabilities())
            if looks_like_limit_error(message)
            else None
        )
        self._rooms[room_id] = room
        return room

    def sweep(self, now: Optional[int] = None) -> int:
        """Reclaim rooms nobody has been in for a while (each holds a raster)."""
        now = now_ms() if now is None else now
        removed = 0
        for room_id, room in list(self._rooms.items()):
            if not room.is_idle(now, self.config.room_idle_ms):
                room.expire_strokes(now)
                room.prune_images(now)
                continue
            room.dispose()
            del self._rooms[room_id]
            removed += 1
        return removed

    def start_sweeper(self, interval_ms: float = 15_000) -> None:
        if self._sweeper is not None:
            return
        self._sweeper = asyncio.ensure_future(self._loop(interval_ms, lambda: self.sweep()))

    def start_capability_watch(self, interval_ms: float = CAPABILITY_POLL_MS) -> None:
        """Poll while idle so a worker that came back is noticed without an edit."""
        if self._capability_task is not None:
            return
        self._capability_task = asyncio.ensure_future(
            self._loop(interval_ms, self.refresh_capabilities)
        )

    async def _loop(self, interval_ms: float, fn: Callable[[], Any]) -> None:
        while True:
            try:
                await asyncio.sleep(interval_ms / 1000)
            except asyncio.CancelledError:
                return
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                log.exception("[registry] periodic task failed")

    async def refresh_capabilities(self) -> None:
        """Re-probe the backend and push any change into every room."""
        if self._refreshing is not None:
            await self._refreshing
            return
        self._refreshing = asyncio.get_event_loop().create_future()
        try:
            try:
                caps = await self.backend.capabilities()
            except Exception:
                # Not a reason to change what rooms believe - but the next probe
                # that answers is now a recovery, and owed work should run then.
                self._probe_failed = True
                return
            max_resolution = min(
                caps.max_resolution,
                self.config.ai_window
                if self.config.explicit.window
                else max(self.config.ai_window, 1024),
            )
            nxt = RoomLimits(
                profiles=list(caps.profiles),
                max_denoise=caps.max_denoise,
                max_resolution=max_resolution,
                negative_prompt_active=dict(caps.negative_prompt_active),
            )
            # Only a CHANGE justifies re-running rejected work.
            recovered = self._probe_failed
            self._probe_failed = False
            changed = not same_limits(self.limits, nxt)
            if recovered or changed:
                self.retry_all()
            if not changed:
                return
            log.info(
                "[ai] backend limits changed: %s up to %s at denoise <= %s",
                "/".join(nxt.profiles or []),
                nxt.max_resolution,
                nxt.max_denoise,
            )
            self.limits = nxt
            for room in self._rooms.values():
                room.apply_limits(nxt)
        finally:
            fut, self._refreshing = self._refreshing, None
            if not fut.done():
                fut.set_result(None)

    def retry_all(self) -> None:
        for room in self._rooms.values():
            room.retry_now()

    def dispose(self) -> None:
        for task in (self._sweeper, self._capability_task):
            if task is not None:
                task.cancel()
        self._sweeper = None
        self._capability_task = None
        for room in self._rooms.values():
            room.dispose()
        self._rooms.clear()
