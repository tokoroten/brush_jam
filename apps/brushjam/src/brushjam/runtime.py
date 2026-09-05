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
from .constants import CLOSE_SUPERSEDED
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
#: Rate-limit buckets held at once. Address cardinality is not ours to choose.
MAX_RATE_BUCKETS = 4096
#: Uploads whose bodies may be in memory at once, process-wide, and the bytes
#: they may hold between them. An upload body is read into a bytearray before
#: any room, image or socket limit has a say, so this is the only thing
#: standing between modest concurrency and a large transient spike.
MAX_CONCURRENT_UPLOADS = 4
MAX_UPLOAD_BYTES_IN_FLIGHT = 48 * 1024 * 1024
#: What one upload is charged. content-length is optional and not to be
#: trusted, so a slot reserves the largest body the route will accept.
MAX_IMAGE_BYTES_PER_SLOT = 12 * 1024 * 1024
MAX_IMAGES_PER_ROOM = 32
#: An uploaded image gets this long to become a layer before it is swept.
IMAGE_GRACE_MS = 2 * 60_000
MAX_IMAGE_BYTES_PER_ROOM = 64 * 1024 * 1024
#: How often idle capabilities are re-checked; a restarted worker is silent.
CAPABILITY_POLL_MS = 60_000



def _revoke(socket: Any, code: int = 1012) -> None:
    """Cut a socket off, with the close code that says why.

    Written defensively because the tests drive rooms with sockets that
    implement only part of `Connection`, and an older one takes no code.
    """
    revoke = getattr(socket, "revoke", None)
    if not callable(revoke):
        try:
            socket.close_now()
        except Exception:
            pass  # already gone
        return
    try:
        revoke(code)
    except TypeError:
        try:
            revoke()
        except Exception:
            pass
    except Exception:
        pass  # already gone


class Connection(Protocol):
    """The socket-shaped surface the runtime needs."""

    def send_text(self, data: str) -> None:
        ...

    def close_now(self) -> None:
        ...

    def revoke(self, code: int = 1012) -> None:
        """Cut the socket off immediately, discarding anything queued."""
        ...

    def release(self, first: Optional[str] = None) -> None:
        """Deliver `first`, then anything buffered since the connection was
        opened, and stop buffering."""
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
            config.max_room_snapshot_bytes,
        )
        #: Allocated on the first accepted AI result: a full-size raster is big.
        self._ai: Optional[AICanvas] = None
        #: One writer or one reader at a time. Compositing mutates the Pillow
        #: image while /ai.png may be encoding it, and neither is thread-safe.
        self._ai_lock = asyncio.Lock()
        self._sockets: "Dict[str, Connection]" = {}
        #: False until a join actually completes. A room created by a POST, or
        #: touched by a connection that never got its snapshot, is a
        #: reservation holding a slot in a small table.
        self.ever_joined = False
        #: Sockets holding a reservation on this room. Every live socket holds
        #: exactly one for its whole life - taken before the handshake's first
        #: await, released in the route's finally - so this, not the member
        #: count, is what the room cap is about.
        self.reserved_sockets = 0
        #: The lease each joined member's socket holds, so a reconnect can take
        #: over the one it replaces rather than opening a second.
        self._leases: "Dict[str, Any]" = {}
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
                    seed=snap.seed,
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

    def _begin_join(
        self,
        socket: Connection,
        name: str,
        token: Optional[str],
        lease: Optional[Any] = None,
    ) -> Tuple[str, Message]:
        """Membership, the connection mapping and the snapshot, all in one
        synchronous step.

        The replacement is installed *before* the old socket is revoked, so
        there is never an instant where the user id maps to nothing: the old
        route reaching `leave` in that instant used to remove the member the
        new socket had just resumed.

        The connection is registered held (see `SocketConnection.hold`), so
        anything broadcast while the snapshot is being serialised is kept for
        it and delivered after the snapshot rather than dropped.
        """
        member = join_member(self.state, name, token)
        user_id = member["userId"]
        previous = self._sockets.get(user_id)
        # Install first, revoke second - and take the lease over at the same
        # moment, so a handshake that failed before this point left the socket
        # it was replacing holding its own slot.
        self._sockets[user_id] = socket
        if lease is not None:
            commit = getattr(lease, "commit", None)
            if callable(commit):
                commit()
            self._leases[user_id] = lease
        else:
            self._leases.pop(user_id, None)
        if previous is not None and previous is not socket:
            # Same identity resumed while the old socket still looked alive: the
            # newest wins, and the old one loses its identity synchronously.
            # Draining its queue first would let it keep sending as this user.
            # A distinct code, not a transport error: the replaced tab must
            # know it was superseded and stop reconnecting, or two tabs sharing
            # a session token evict each other for as long as they are open.
            _revoke(previous, CLOSE_SUPERSEDED)
        msg: Message = {
            "t": "snapshot",
            # Detached containers: this is serialised on another thread while
            # the loop keeps mutating the room.
            "snapshot": snapshot(
                self.state,
                user_id,
                self.scheduler.state,
                {
                    "window": self.config.ai_window,
                    "apply": self.config.ai_apply,
                    "canvasSize": self.config.canvas_size,
                },
            ),
        }
        return user_id, msg

    def _release(self, socket: Connection, first: Optional[str] = None) -> None:
        """Hand the connection its snapshot and let its held frames follow."""
        release = getattr(socket, "release", None)
        if callable(release):
            release(first)
        elif first is not None:
            socket.send_text(first)

    def _finish_join(self, user_id: str, socket: Connection, data: str) -> str:
        if self._sockets.get(user_id) is not socket:
            # Superseded again while its snapshot was being serialised. The
            # newer connection owns the identity; this one is already revoked.
            return user_id
        self.ever_joined = True
        if not socket.open:
            return user_id
        if socket.buffered_bytes + len(data) > MAX_BUFFERED_BYTES:
            # Snapshot plus everything held while it was being built: they are
            # released together, so they have to fit together. Revoking rather
            # than closing discards the held buffer too - there is no point
            # keeping frames for a connection that never got its snapshot.
            log.warning(
                "[room %s] snapshot for %s does not fit (%d + %d bytes)",
                self.state.id,
                user_id,
                len(data),
                socket.buffered_bytes,
            )
            self._sockets.pop(user_id, None)
            self._leases.pop(user_id, None)
            remove_member(self.state, user_id)
            _revoke(socket)
            return user_id
        # The snapshot first, then everything that arrived while it was being
        # built - not the other way round, and not instead of them.
        self._release(socket, data)
        self.broadcast({"t": "presence", "members": list(self.state.members.values())})
        return user_id

    def rollback_join(self, user_id: Optional[str], socket: Connection) -> None:
        """Undo a join that never completed, so a failure cannot leave a ghost
        member or a mapping to a socket nobody is reading."""
        if user_id is None:
            return
        if self._sockets.get(user_id) is socket:
            self._sockets.pop(user_id, None)
            self._leases.pop(user_id, None)
            remove_member(self.state, user_id)
            self.broadcast({"t": "presence", "members": list(self.state.members.values())})

    def join(
        self,
        socket: Connection,
        name: str,
        token: Optional[str] = None,
        lease: Optional[Any] = None,
    ) -> str:
        user_id, msg = self._begin_join(socket, name, token, lease)
        try:
            return self._finish_join(user_id, socket, json.dumps(msg, separators=(",", ":")))
        except BaseException:
            # A half-finished join must not leave a member nobody can reach.
            self.rollback_join(user_id, socket)
            raise

    async def join_async(
        self,
        socket: Connection,
        name: str,
        token: Optional[str] = None,
        lease: Optional[Any] = None,
    ) -> str:
        """`join`, with the snapshot serialised off the event loop.

        A long session's log is megabytes of JSON; `json.dumps` on it stalls
        every other room's sockets, presence and stroke relay while it runs.
        """
        user_id, msg = self._begin_join(socket, name, token, lease)
        try:
            data = await asyncio.to_thread(json.dumps, msg, separators=(",", ":"))
            return self._finish_join(user_id, socket, data)
        except BaseException:
            # Includes cancellation at shutdown: a member inserted by
            # _begin_join is rolled back on every path that is not a join.
            self.rollback_join(user_id, socket)
            raise

    def lease_for_token(self, token: Optional[str]) -> Optional[Any]:
        """The reservation held by the socket this token would replace.

        A reconnect that takes over an existing connection needs no new slot,
        and refusing it at exactly the cap is the one case where the cap does
        the opposite of what it is for - but it must take over that socket's
        lease, not open a second one beside it.
        """
        if not token:
            return None
        prior = self.state.sessions.get(token)
        if prior is None:
            return None
        user_id = prior["userId"]
        if user_id not in self._sockets:
            return None
        return self._leases.get(user_id)

    def leave(self, user_id: str, socket: Optional[Connection] = None) -> None:
        """`socket` identifies *which* connection closed: a superseded socket
        fires its close after the replacement has joined.

        Exact ownership, including "the mapping is gone": a socket that no
        longer holds the identity must not remove the member who does.
        """
        current = self._sockets.get(user_id)
        if socket is not None and current is not socket:
            return
        for cancel in remove_member(self.state, user_id):
            self.broadcast(cancel)
        self._sockets.pop(user_id, None)
        self._leases.pop(user_id, None)
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

    def is_pinned(self) -> bool:
        """A handshake holds a slot here right now.

        The reservation is taken before the socket is accepted, so without this
        the sweeper could delete the room between those two instants: the route
        would then join a RoomRuntime the registry no longer knows, while every
        HTTP route 404s and the same id could be recreated as a different room.
        """
        return self.reserved_sockets > 0

    def is_idle(self, now: int, idle_ms: int) -> bool:
        if self.is_pinned():
            return False
        return len(self._sockets) == 0 and now - self.state.last_active_at > idle_ms

    def is_abandoned_reservation(self, now: int, ttl_ms: int) -> bool:
        """Created, never joined, and old enough that nobody is coming."""
        if self.is_pinned():
            return False
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
                "seed": self.state.seed,
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
        # What it would hold *after* this frame, not what it holds now: a
        # client one byte under the cap could otherwise be handed a
        # multi-megabyte stroke_committed and end up far above it.
        if socket.buffered_bytes + len(data) > MAX_BUFFERED_BYTES:
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


class UploadSlot:
    """One upload body's claim on the process's in-flight budget."""

    __slots__ = ("_gate", "_client", "_bytes", "_released")

    def __init__(self, gate: "UploadGate", client: str, size: int) -> None:
        self._gate = gate
        self._client = client
        self._bytes = size
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._gate._release(self._client, self._bytes)


class UploadGate:
    """Admission for upload bodies, taken before a single byte is read.

    One at a time per address, a few at a time overall, and a hard ceiling on
    the bytes held between them - because the room rate limit and the image
    store only get a say once the body is already in memory.
    """

    def __init__(
        self,
        max_concurrent: int = MAX_CONCURRENT_UPLOADS,
        max_bytes: int = MAX_UPLOAD_BYTES_IN_FLIGHT,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_bytes = max_bytes
        self._in_flight = 0
        self._bytes = 0
        self._by_client: "Dict[str, int]" = {}

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def bytes_in_flight(self) -> int:
        return self._bytes

    def acquire(self, client: str, size: int) -> Optional[UploadSlot]:
        """A slot, or None when the process is already holding enough."""
        if self._by_client.get(client, 0) >= 1:
            return None
        if self._in_flight >= self.max_concurrent:
            return None
        if self._bytes + size > self.max_bytes:
            return None
        self._in_flight += 1
        self._bytes += size
        self._by_client[client] = self._by_client.get(client, 0) + 1
        return UploadSlot(self, client, size)

    def _release(self, client: str, size: int) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        self._bytes = max(0, self._bytes - size)
        remaining = self._by_client.get(client, 0) - 1
        if remaining > 0:
            self._by_client[client] = remaining
        else:
            self._by_client.pop(client, None)


class SocketReservation:
    """One socket's claim on the global and per-room budgets.

    Released exactly once, however many times `release` is called: the route's
    `finally`, a revocation and a failed join all want to give it back, and
    only one of them may.

    A reconnect does not take a second claim, and it does not take the first
    one early either. It gets a *ticket*: a provisional claim that owns
    nothing and moves no counters until `commit`, because everything between
    the reservation and the join can fail, and until it succeeds the socket
    being replaced is still connected and still needs its slot.

    There is exactly one counted slot per identity for the whole crossover,
    and it is never in flight. Whichever of the two lets go first hands it to
    the other: the original releasing while a ticket is out transfers the slot
    (and with it the room's pin) to the ticket rather than decrementing, and a
    ticket that commits inherits it rather than incrementing. There is no path
    that takes a slot without one having been checked for.
    """

    __slots__ = (
        "_registry",
        "_room",
        "_released",
        "_owns",
        "_supersedes",
        "_ticket",
    )

    def __init__(
        self,
        registry: "RoomRegistry",
        room: "RoomRuntime",
        owns: bool = True,
        supersedes: Optional["SocketReservation"] = None,
    ) -> None:
        self._registry = registry
        self._room = room
        self._released = False
        #: Whether this reservation currently holds the counted slot.
        self._owns = owns
        #: The lease this one is taking over, until it commits.
        self._supersedes = supersedes
        #: The outstanding ticket against this lease, if any. One at a time:
        #: two tickets would both believe they were getting the same slot.
        self._ticket: Optional["SocketReservation"] = None

    @property
    def released(self) -> bool:
        return self._released

    @property
    def owns_slot(self) -> bool:
        return self._owns and not self._released

    @property
    def room(self) -> "RoomRuntime":
        return self._room

    def ticket(self) -> Optional["SocketReservation"]:
        """A provisional claim on this lease's slot.

        Nothing moves yet. The original socket keeps its lease, the counters
        keep their values, and if the replacement never arrives the ticket is
        discarded and nothing has to be put back.
        """
        if self._released or self._ticket is not None:
            return None
        ticket = SocketReservation(self._registry, self._room, owns=False, supersedes=self)
        self._ticket = ticket
        return ticket

    def commit(self) -> None:
        """The replacement is installed; the slot is this connection's now."""
        old = self._supersedes
        self._supersedes = None
        if old is None:
            return
        old._ticket = None
        if not self._owns:
            # The original is still holding the slot: inherit it, and make its
            # eventual release a no-op.
            old._owns = False
            old._released = True
        # Otherwise the original released first and handed the slot over then;
        # this reservation has owned it since.
        self._owns = True

    def release(self) -> None:
        if self._released:
            return
        self._released = True

        if self._supersedes is not None:
            # An uncommitted ticket. Either the original still holds the slot,
            # in which case there is nothing to give back, or it released while
            # this ticket was out and handed the slot over - and now the ticket
            # is the one returning it.
            old = self._supersedes
            self._supersedes = None
            old._ticket = None
            if self._owns:
                self._registry._release_reservation(self._room)
            return

        if self._ticket is not None and not self._ticket._released:
            # A reconnect is mid-handshake against this identity. Handing it
            # the slot keeps the count honest - an unrelated connection must
            # not be admitted into a vacancy that is already spoken for - and
            # keeps the room pinned, so the sweeper cannot delete it out from
            # under a ticket that is about to commit.
            self._ticket._owns = self._owns
            self._owns = False
            return

        if self._owns:
            self._registry._release_reservation(self._room)


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
        #: Upload bodies in memory at once, across every room.
        self.uploads = UploadGate()
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

    def reserve_socket(
        self, room: "RoomRuntime", token: Optional[str] = None
    ) -> Optional["SocketReservation"]:
        """Atomically take a slot, or take over the one a reconnect replaces.

        Check-then-increment has to happen with no await in between: two
        handshakes that both passed a check before either incremented would
        both be admitted, which is how a cap that reads correctly is exceeded
        anyway.

        A valid resume token whose member still holds a lease transfers that
        exact lease. That is what lets a reconnect through a full room without
        letting twenty reconnects through it.
        """
        existing = room.lease_for_token(token)
        if existing is not None:
            ticket = existing.ticket()
            if ticket is not None:
                return ticket
        if self._sockets_open >= self.config.max_total_sockets:
            return None
        if room.reserved_sockets >= self.config.max_room_sockets:
            return None
        self._sockets_open += 1
        room.reserved_sockets += 1
        return SocketReservation(self, room)

    def _release_reservation(self, room: "RoomRuntime") -> None:
        self._sockets_open = max(0, self._sockets_open - 1)
        room.reserved_sockets = max(0, room.reserved_sockets - 1)

    def can_accept_socket(self, room: "RoomRuntime") -> Optional[str]:
        """Why a socket could not be accepted right now, or None. Advisory:
        `reserve_socket` is what actually decides."""
        if self._sockets_open >= self.config.max_total_sockets:
            return "the server is holding too many connections"
        if room.reserved_sockets >= self.config.max_room_sockets:
            return "this room is full"
        return None

    # -- room creation rate limit -----------------------------------------

    def allow_create(self, client: str, now: Optional[float] = None) -> bool:
        """A token bucket per client address: `room_create_per_min` creations a
        minute, refilled continuously so a burst is allowed but a flood is not.

        Rooms are created by unauthenticated requests - a POST, an upload, or a
        WebSocket to an id that does not exist yet - and each one reserves a
        slot in a table of 64, so every one of those paths comes through here.
        """
        rate = self.config.room_create_per_min
        now = time.monotonic() if now is None else now
        tokens, last = self._create_buckets.pop(client, (float(rate), now))
        tokens = min(float(rate), tokens + (now - last) * rate / 60.0)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        # Re-inserted last: the dict is the LRU order.
        self._create_buckets[client] = (tokens, now)
        self._prune_buckets(now)
        return allowed

    def _prune_buckets(self, now: float) -> None:
        """Bounded on both paths, accepted and rejected.

        A bucket that has refilled says nothing that a fresh one would not, so
        it is dropped on age; beyond that the table is capped and the oldest
        entries go, because address cardinality is not something a server gets
        to choose.
        """
        rate = float(self.config.room_create_per_min)
        full_after = 60.0  # a bucket is fully refilled a minute after its last use
        if len(self._create_buckets) > 64:
            for key, (tokens, last) in list(self._create_buckets.items()):
                if now - last >= full_after and tokens < rate:
                    del self._create_buckets[key]
                elif now - last >= full_after:
                    del self._create_buckets[key]
        while len(self._create_buckets) > MAX_RATE_BUCKETS:
            self._create_buckets.pop(next(iter(self._create_buckets)))

    def get(self, room_id: str) -> Optional[RoomRuntime]:
        return self._rooms.get(room_id)

    def create(self, client: str) -> Optional[RoomRuntime]:
        """A room with a fresh id, for `POST /api/rooms`."""
        room_id = short_id(8)
        while room_id in self._rooms:
            room_id = short_id(8)
        return self.create_named(room_id, client)

    def create_named(self, room_id: str, client: str) -> Optional[RoomRuntime]:
        """Create one specific room, rate limited by client address.

        Every path that can bring a room into existence goes through here -
        the POST, an upload to an unknown id, and a WebSocket to a link that
        has not been opened yet. `get` is the lookup that creates nothing, so
        a caller cannot reach creation by accident.
        """
        existing = self._rooms.get(room_id)
        if existing is not None:
            return existing
        # Capacity first: a refusal must not spend the caller's budget, and a
        # sweep may free the slot that was missing.
        if self.at_capacity and self.sweep() == 0 and self.at_capacity:
            return None
        if not self.allow_create(client):
            return None
        return self._make(room_id)

    def get_or_create(self, room_id: str, client: str) -> Optional[RoomRuntime]:
        """Rooms are created on demand so a shared URL always works - at the
        same rate as any other creation."""
        existing = self._rooms.get(room_id)
        if existing is not None:
            return existing
        return self.create_named(room_id, client)

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
