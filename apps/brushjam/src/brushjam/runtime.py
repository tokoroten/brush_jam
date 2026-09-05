"""Room runtime: sockets, presence, broadcast, image store, eviction.
Port of apps/server/src/runtime.ts.

A `Connection` is anything that can be handed a JSON string and told to go away,
so the tests can drive a room without a real socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

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
from .scheduler import AIScheduler, GenerationAdmission, RenderJob, SchedulerOptions
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

    def revoke(self) -> None:
        """Cut the socket off immediately, discarding anything queued."""
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
        admission: Optional[GenerationAdmission] = None,
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
            config.max_room_points,
        )
        #: Allocated on the first accepted AI result: a full-size raster is big.
        self._ai: Optional[AICanvas] = None
        #: One writer or one reader at a time. Compositing mutates the Pillow
        #: image while /ai.png may be encoding it, and neither is thread-safe.
        self._ai_lock = asyncio.Lock()
        self._sockets: "Dict[str, Connection]" = {}
        #: False until somebody actually connects. A room created by a POST and
        #: never joined is a reservation holding a slot in a small table.
        self.ever_joined = False
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
                async with runtime._ai_lock:
                    # The canvas is built inside the lock and inside the thread:
                    # at a 2048 canvas its first allocation is 16 MiB, which is
                    # not something to do on the event loop.
                    png = await asyncio.to_thread(runtime._composite, patch, crop, mask)
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
                admission=admission,
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

    def _composite(self, patch: bytes, crop: Rect, mask: Any) -> bytes:
        """Runs on a worker thread, inside `_ai_lock`."""
        return self._ai_canvas().composite(patch, crop, mask)

    async def ai_png(self) -> bytes:
        async with self._ai_lock:
            return await asyncio.to_thread(lambda: self._ai_canvas().to_png())

    def patch(self, patch_id: str) -> Optional[bytes]:
        return self._patches.get(patch_id)

    # -- membership -------------------------------------------------------

    def _begin_join(self, socket: Connection, name: str, token: Optional[str]) -> Tuple[str, Message]:
        """Membership and the snapshot message, without registering the socket.

        The socket is registered only once its snapshot is queued: a broadcast
        that reached it earlier would arrive before the snapshot, and the
        protocol says the snapshot is first.
        """
        member = join_member(self.state, name, token)
        previous = self._sockets.get(member["userId"])
        if previous is not None and previous is not socket:
            # Same identity resumed while the old socket still looked alive: the
            # newest wins, and the old one loses its identity synchronously.
            # Draining its queue first would let it keep sending as this user.
            del self._sockets[member["userId"]]
            revoke = getattr(previous, "revoke", None)
            try:
                revoke() if callable(revoke) else previous.close_now()
            except Exception:
                pass  # already gone
        msg: Message = {
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
        }
        return member["userId"], msg

    def _finish_join(self, user_id: str, socket: Connection, data: str) -> str:
        if len(data) > MAX_BUFFERED_BYTES:
            # One frame already over the slow-client cap. Queueing it would put
            # the room past a limit that exists to bound exactly this.
            log.warning(
                "[room %s] snapshot for %s is %d bytes; refusing the join",
                self.state.id,
                user_id,
                len(data),
            )
            revoke = getattr(socket, "revoke", None)
            try:
                revoke() if callable(revoke) else socket.close_now()
            except Exception:
                pass
            return user_id
        self._sockets[user_id] = socket
        self.ever_joined = True
        self._try_send(user_id, socket, data)
        self.broadcast({"t": "presence", "members": list(self.state.members.values())})
        return user_id

    def join(self, socket: Connection, name: str, token: Optional[str] = None) -> str:
        user_id, msg = self._begin_join(socket, name, token)
        return self._finish_join(user_id, socket, json.dumps(msg, separators=(",", ":")))

    async def join_async(self, socket: Connection, name: str, token: Optional[str] = None) -> str:
        """`join`, with the snapshot serialised off the event loop.

        A long session's log is megabytes of JSON; `json.dumps` on it stalls
        every other room's sockets, presence and stroke relay while it runs.
        """
        user_id, msg = self._begin_join(socket, name, token)
        data = await asyncio.to_thread(json.dumps, msg, separators=(",", ":"))
        return self._finish_join(user_id, socket, data)

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

    def handle(self, user_id: str, raw: str, connection: Optional[Connection] = None) -> None:
        if connection is not None and self._sockets.get(user_id) is not connection:
            # A superseded socket still draining its receive buffer. It is no
            # longer this user; anything it says is from a connection that has
            # already been replaced.
            return
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

    def is_abandoned_reservation(self, now: int, ttl_ms: int) -> bool:
        """Created, never joined, and old enough that nobody is coming."""
        return not self.ever_joined and now - self.state.created_at > ttl_ms

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
        #: Live sockets across every room, so one room cannot exhaust the process.
        self._sockets_open = 0
        #: One generation at a time across the whole process, first come first
        #: served - acquired before a room rasterises anything.
        self.admission = GenerationAdmission()
        #: Room-creation token buckets, keyed by client address.
        self._create_buckets: "Dict[str, Tuple[float, float]]" = {}
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

    # -- socket accounting -------------------------------------------------

    @property
    def sockets_open(self) -> int:
        return self._sockets_open

    def can_accept_socket(self, room: "RoomRuntime") -> Optional[str]:
        """Why this socket cannot be accepted, or None if it can.

        Checked before the connection object exists and before membership is
        touched, so a refusal costs nothing and changes nothing.
        """
        if self._sockets_open >= self.config.max_total_sockets:
            return "the server is holding too many connections"
        if room.member_count >= self.config.max_room_sockets:
            return "this room is full"
        return None

    def note_socket_open(self) -> None:
        self._sockets_open += 1

    def note_socket_closed(self) -> None:
        self._sockets_open = max(0, self._sockets_open - 1)

    # -- room creation rate limit -----------------------------------------

    def allow_create(self, client: str, now: Optional[float] = None) -> bool:
        """A token bucket per client address: `room_create_per_min` creations a
        minute, refilled continuously so a burst is allowed but a flood is not.

        Rooms are created by unauthenticated POSTs and each one reserves a slot
        in a table of 64, so this is what stands between a script and a server
        that cannot host anybody.
        """
        rate = self.config.room_create_per_min
        now = time.monotonic() if now is None else now
        tokens, last = self._create_buckets.get(client, (float(rate), now))
        tokens = min(float(rate), tokens + (now - last) * rate / 60.0)
        if tokens < 1.0:
            self._create_buckets[client] = (tokens, now)
            return False
        self._create_buckets[client] = (tokens - 1.0, now)
        if len(self._create_buckets) > 4096:
            # Never unbounded: drop buckets that have refilled anyway.
            self._create_buckets = {
                key: value
                for key, value in self._create_buckets.items()
                if value[0] < rate - 0.001
            }
        return True

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
        room = RoomRuntime(room_id, self.backend, self.config, self.limits, self.admission)
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
            reservation = room.is_abandoned_reservation(now, self.config.unjoined_room_ttl_ms)
            if not reservation and not room.is_idle(now, self.config.room_idle_ms):
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
