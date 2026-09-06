"""Authoritative room reducer. Port of the retired Node server's src/room.ts.

Room state is plain data; `apply_client_message` is the only thing that changes
it and returns the frames to send, exactly like the TypeScript reducer. Nothing
here awaits, so a render can never observe a half-applied message.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .constants import (
    MAX_SEED,
    AI_PROFILES,
    CANVAS_SIZE,
    DEFAULT_STROKE_ALPHA,
    DENOISE_STEP,
    MAX_AI_RESOLUTION,
    MAX_DENOISE,
    MAX_LAYERS,
    MAX_NEGATIVE_PROMPT,
    MAX_STROKE_ALPHA,
    MIN_AI_RESOLUTION,
    MIN_DENOISE,
    MIN_STROKE_ALPHA,
    PROFILE_DEFAULTS,
)
from .geometry import Point, Rect, stroke_bbox, translate_rect, union_rects
from .ids import member_color, short_id
from .protocol import Layer, Member, Message, RoomSnapshot, Stroke, StrokeInit

log = logging.getLogger("brushjam.room")

#: Matches the AI_DENOISE default; a room can be created with the configured one.
DEFAULT_DENOISE = 0.55

#: Hard caps on a single in-progress stroke.
MAX_STROKE_POINTS = 50_000
MAX_STROKE_MS = 60_000
#: A stroke nobody has added points to in this long is abandoned.
PENDING_STROKE_IDLE_MS = 60_000
#: One person cannot have more than this many strokes in progress at once.
MAX_PENDING_PER_USER = 4
#: Upper bound on a room's committed stroke log (snapshots are sent in full).
MAX_STROKES_PER_ROOM = 20_000
#: Aggregate committed points in one room. The stroke count alone is not a
#: bound: 20,000 strokes of 50,000 points is a billion point dicts.
MAX_ROOM_POINTS = 2_000_000
#: What the log is estimated to serialise to, which is the limit that actually
#: matters: past the outbound frame cap the room is one nobody can join or
#: resume into, so it must never be reachable. Sized below MAX_BUFFERED_BYTES
#: (8 MiB) with room for the rest of the snapshot.
MAX_ROOM_SNAPSHOT_BYTES = 6 * 1024 * 1024
def drop_pending(room: RoomState, stroke_id: str) -> Optional[PendingStroke]:
    """Forget a stroke in progress and give its points back to the budget.

    Every path that ends a pending stroke goes through here - commit, cancel,
    the author leaving, a layer cleared or deleted, and the idle sweep - so a
    reservation cannot outlive the stroke that took it.
    """
    p = room.pending.pop(stroke_id, None)
    if p is not None:
        room.pending_points = max(0, room.pending_points - len(p.points))
    return p


def room_is_full(room: RoomState, extra_points: int) -> bool:
    """Whether `extra_points` more would put the room past what it may hold.

    Committed and pending together: a client that never finishes a stroke
    occupies exactly as much memory as one that does, and four 50,000-point
    strokes each from sixteen members is 3.2 million point dicts that no
    commit-time check would ever see.
    """
    return room.committed_points + room.pending_points + extra_points > room.max_points


def stroke_snapshot_bytes(stroke: Stroke) -> int:
    """Exactly what this stroke adds to a snapshot, plus its separating comma.

    Measured rather than estimated. A per-point constant cannot be both safe
    and useful here: points are clamped but not rounded, so one client sending
    `1.2345678901234567` serialises to nearly three times what a well-behaved
    one does, and an estimate large enough to bound that would shrink an
    ordinary session's log to a fraction of the budget. The strokes this runs
    on are small - it is one `dumps` of one stroke at commit time.
    """
    return len(json.dumps(stroke, separators=(",", ":"))) + 1

MAX_SESSIONS = 64

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class OrderedSet:
    """Insertion-ordered set, because `undone` is sent in the snapshot and a JS
    Set iterates in insertion order. A plain Python set would reorder it, and
    two servers would disagree about a message they both consider correct."""

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[str] = ()) -> None:
        self._items: Dict[str, None] = dict.fromkeys(items)

    def add(self, item: str) -> None:
        self._items[item] = None

    def discard(self, item: str) -> None:
        self._items.pop(item, None)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"OrderedSet({list(self._items)!r})"


def now_ms() -> int:
    return int(time.time() * 1000)


def _finite(n: Any) -> bool:
    return isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n)


def _clamp(n: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, n))


def _is_hex_color(s: Any) -> bool:
    return isinstance(s, str) and bool(_HEX_COLOR.match(s))


def clamp_resolution(value: float, maximum: float) -> int:
    """Snap a requested generation size to the 64 grid and the allowed range."""
    if not _finite(value):
        return int(maximum)
    stepped = _js_round(value / 64) * 64
    return int(min(maximum, max(MIN_AI_RESOLUTION, min(MAX_AI_RESOLUTION, stepped))))


def _js_round(v: float) -> float:
    """JS Math.round: half away from zero *upwards* (Math.round(-0.5) === -0)."""
    return math.floor(v + 0.5)


def clamp_seed(value: Any) -> int:
    """A sampling seed: whole, in range, and never negative."""
    number = int(value)
    if number < 0 or number > MAX_SEED:
        number %= MAX_SEED + 1
    return number


def is_seed(value: Any) -> bool:
    """Whether a client sent something that can be a seed at all."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and float(value).is_integer()
        and 0 <= value <= MAX_SEED
    )


def random_seed() -> int:
    return random.randrange(MAX_SEED + 1)


def clamp_denoise(value: float) -> float:
    if not _finite(value):
        return DEFAULT_DENOISE
    stepped = _js_round(value / DENOISE_STEP) * DENOISE_STEP
    return min(MAX_DENOISE, max(MIN_DENOISE, _js_round(stepped * 100) / 100))


@dataclass
class RoomLimits:
    """Backend-derived ceilings a room is created with."""

    profiles: Optional[List[str]] = None
    max_denoise: Optional[float] = None
    negative_prompt_active: Optional[Dict[str, bool]] = None
    #: Ceiling for the generation size, which is NOT the starting size.
    max_resolution: Optional[float] = None

    def copy(self) -> "RoomLimits":
        return RoomLimits(
            profiles=None if self.profiles is None else list(self.profiles),
            max_denoise=self.max_denoise,
            negative_prompt_active=(
                None if self.negative_prompt_active is None else dict(self.negative_prompt_active)
            ),
            max_resolution=self.max_resolution,
        )


@dataclass
class RoomImage:
    id: str
    mime: str
    data: bytes
    width: int
    height: int
    #: Upload time, so an image can be swept once nothing references it.
    created_at: int


@dataclass
class PendingStroke:
    user_id: str
    init: StrokeInit
    points: List[Point]
    started_at: int
    last_activity_at: int


@dataclass
class RoomState:
    id: str
    canvas_size: int
    prompt: str
    denoise: float
    negative_prompt: str
    ai_resolution: int
    ai_resolution_max: int
    ai_resolution_adjustable: bool
    ai_profile: str
    ai_profiles: List[str]
    max_denoise: float
    negative_active: Dict[str, bool]
    #: The room's sampling seed. Fixed, not drawn per generation: with a fresh
    #: seed every time, adding one stroke reshuffled the entire picture, so
    #: nobody could tell their own change from the noise. Re-rolling it is a
    #: deliberate act (the dice in the UI).
    seed: int = 0
    human_revision: int = 0
    ai_revision: int = 0
    ai_generation: int = 0
    layers: List[Layer] = field(default_factory=list)
    strokes: List[Stroke] = field(default_factory=list)
    undone: OrderedSet = field(default_factory=OrderedSet)
    members: "Dict[str, Member]" = field(default_factory=dict)
    member_seq: int = 0
    pending: "Dict[str, PendingStroke]" = field(default_factory=dict)
    images: "Dict[str, RoomImage]" = field(default_factory=dict)
    sessions: "Dict[str, Member]" = field(default_factory=dict)
    created_at: int = 0
    last_active_at: int = 0
    #: Sum of the points in `strokes`, maintained rather than recomputed: it is
    #: consulted on every commit.
    committed_points: int = 0
    #: This room's aggregate point ceiling (the config value at creation).
    max_points: int = MAX_ROOM_POINTS
    #: Points held by strokes still being drawn. They are not in the log yet,
    #: but they are in memory, and the log's budget is what says how much of
    #: this room a client may occupy.
    pending_points: int = 0
    #: Estimated serialised size of the committed stroke log.
    snapshot_bytes: int = 0
    #: Ceiling for the above (the config value at creation).
    max_snapshot_bytes: int = MAX_ROOM_SNAPSHOT_BYTES


@dataclass
class ApplyResult:
    #: Messages to send to every member (including the sender).
    broadcast: List[Message] = field(default_factory=list)
    #: Messages to relay to everyone except the sender.
    relay: List[Message] = field(default_factory=list)
    #: World-space rects that changed and should be reconsidered by the AI.
    dirty: List[Rect] = field(default_factory=list)
    #: Messages for the sender only (validation feedback).
    to_sender: List[Message] = field(default_factory=list)
    #: True when a room setting changed and the AI should re-run.
    prompt_changed: bool = False


def _empty() -> ApplyResult:
    return ApplyResult()


def _refuse(message: str) -> ApplyResult:
    return ApplyResult(to_sender=[{"t": "error", "message": message}])


def _refuse_stroke(user_id: str, stroke_id: str, message: str) -> ApplyResult:
    """Sender-only refusal that also clears the sender's optimistic preview."""
    return ApplyResult(
        to_sender=[
            {"t": "stroke_cancel", "userId": user_id, "strokeId": stroke_id, "reason": message},
            {"t": "error", "message": message},
        ]
    )


def qualify_stroke_id(user_id: str, raw_id: str) -> str:
    """Stroke ids are namespaced by author so two clients cannot collide."""
    return f"{user_id}:{raw_id}"


def create_room(
    room_id: str,
    denoise: float = DEFAULT_DENOISE,
    canvas_size: int = CANVAS_SIZE,
    resolution: Optional[int] = None,
    adjustable_resolution: bool = True,
    profile: str = "fast",
    limits: Optional[RoomLimits] = None,
    max_points: int = MAX_ROOM_POINTS,
    max_snapshot_bytes: int = MAX_ROOM_SNAPSHOT_BYTES,
    seed: Optional[int] = None,
) -> RoomState:
    limits = limits or RoomLimits()
    resolution = canvas_size if resolution is None else resolution
    profiles = list(limits.profiles) if limits.profiles else list(AI_PROFILES)
    resolution_max = clamp_resolution(
        resolution if limits.max_resolution is None else limits.max_resolution, MAX_AI_RESOLUTION
    )
    max_denoise = min(MAX_DENOISE if limits.max_denoise is None else limits.max_denoise, MAX_DENOISE)
    # A room can only start on a profile the backend has.
    start_profile = profile if profile in profiles else profiles[0]
    now = now_ms()
    return RoomState(
        id=room_id,
        canvas_size=canvas_size,
        prompt="anime style, fantasy town, vibrant colors",
        denoise=min(clamp_denoise(denoise), max_denoise),
        negative_prompt="",
        ai_resolution=clamp_resolution(resolution, resolution_max),
        ai_resolution_max=resolution_max,
        ai_resolution_adjustable=adjustable_resolution,
        ai_profile=start_profile,
        ai_profiles=profiles,
        max_denoise=max_denoise,
        negative_active=dict(limits.negative_prompt_active or {"fast": True, "quality": True}),
        # Random per room, so two rooms drawing the same thing do not come out
        # identical, and stable within one.
        seed=random_seed() if seed is None else clamp_seed(seed),
        layers=[
            {
                "id": short_id(6),
                "name": "Layer 1",
                "kind": "draw",
                "visible": True,
                "locked": False,
                "opacity": 1,
                "order": 0,
                "includeInAI": True,
            }
        ],
        created_at=now,
        last_active_at=now,
        max_points=max(1, int(max_points)),
        max_snapshot_bytes=max(1024, int(max_snapshot_bytes)),
    )


def join_member(room: RoomState, name: str, token: Optional[str] = None) -> Member:
    """Join, resuming a previous identity when the client presents a token whose
    member is not currently connected."""
    if token:
        prior = room.sessions.get(token)
        if prior:
            # Resume unconditionally, even if the old socket still looks
            # connected: that is usually a dead TCP connection the server has
            # not noticed. The caller replaces the stale socket.
            room.members[prior["userId"]] = prior
            room.last_active_at = now_ms()
            return prior
    member = add_member(room, name)
    if token:
        # A new token is only recorded if a slot can be freed without evicting
        # somebody who is still connected.
        if len(room.sessions) < MAX_SESSIONS or _evict_oldest_disconnected_session(room):
            room.sessions[token] = member
    return member


def _evict_oldest_disconnected_session(room: RoomState) -> bool:
    for token, member in room.sessions.items():
        if member["userId"] in room.members:
            continue
        del room.sessions[token]
        return True
    # Everyone in the table is still connected: keep them all and refuse to
    # record the newcomer's token.
    return False


def add_member(room: RoomState, name: str) -> Member:
    member: Member = {
        "userId": short_id(10),
        "name": name[:24] or f"artist{room.member_seq + 1}",
        "color": member_color(room.member_seq),
    }
    room.member_seq += 1
    room.members[member["userId"]] = member
    room.last_active_at = now_ms()
    return member


def remove_member(room: RoomState, user_id: str) -> List[Message]:
    """Drops a member and cancels any stroke they were still drawing."""
    room.members.pop(user_id, None)
    room.last_active_at = now_ms()
    cancels: List[Message] = []
    for stroke_id, p in list(room.pending.items()):
        if p.user_id != user_id:
            continue
        drop_pending(room, stroke_id)
        cancels.append(
            {"t": "stroke_cancel", "userId": user_id, "strokeId": stroke_id, "reason": "author left"}
        )
    return cancels


def expire_pending_strokes(
    room: RoomState, now: Optional[int] = None, idle_ms: int = PENDING_STROKE_IDLE_MS
) -> List[Message]:
    """Abandons strokes nobody has added points to for a while."""
    now = now_ms() if now is None else now
    cancels: List[Message] = []
    for stroke_id, p in list(room.pending.items()):
        if now - p.last_activity_at <= idle_ms and now - p.started_at <= MAX_STROKE_MS:
            continue
        drop_pending(room, stroke_id)
        cancels.append(
            {
                "t": "stroke_cancel",
                "userId": p.user_id,
                "strokeId": stroke_id,
                "reason": "stroke abandoned",
            }
        )
    return cancels


def _cancel_pending_on_layer(
    room: RoomState, layer_id: str, reason: str = "layer removed"
) -> List[Message]:
    cancels: List[Message] = []
    for stroke_id, p in list(room.pending.items()):
        if p.init["layerId"] != layer_id:
            continue
        drop_pending(room, stroke_id)
        cancels.append(
            {"t": "stroke_cancel", "userId": p.user_id, "strokeId": stroke_id, "reason": reason}
        )
    return cancels


def sorted_layers(room: RoomState) -> List[Layer]:
    return sorted(room.layers, key=lambda l: l["order"])


def find_layer(room: RoomState, layer_id: str) -> Optional[Layer]:
    for layer in room.layers:
        if layer["id"] == layer_id:
            return layer
    return None


def snapshot(room: RoomState, you_user_id: str, ai_state: str, ai: Dict[str, Any]) -> RoomSnapshot:
    """A detached copy of the room.

    Every container is copied, because this is serialised on a worker thread
    while the event loop keeps mutating the room: a live `strokes` list or a
    live layer dict would either raise mid-`dumps` or produce a snapshot whose
    contents disagree with the `humanRevision` beside them.
    """
    return {
        "roomId": room.id,
        "youUserId": you_user_id,
        "prompt": room.prompt,
        "humanRevision": room.human_revision,
        "aiRevision": room.ai_revision,
        "aiGeneration": room.ai_generation,
        "canvasSize": ai.get("canvasSize", CANVAS_SIZE),
        "aiWindow": ai["window"],
        "aiApply": ai["apply"],
        "denoise": room.denoise,
        "seed": room.seed,
        "negativePrompt": room.negative_prompt,
        "aiResolution": room.ai_resolution,
        "aiResolutionMax": room.ai_resolution_max,
        "aiResolutionAdjustable": room.ai_resolution_adjustable,
        "aiProfile": room.ai_profile,
        "aiProfiles": list(room.ai_profiles),
        "maxDenoise": room.max_denoise,
        "negativePromptActive": room.negative_active.get(room.ai_profile, True) is not False,
        # A server setting, not a room one: it is the same for every room in
        # this process and nothing in the room can change it.
        "r18Presets": bool(ai.get("r18Presets", False)),
        "members": [dict(m) for m in room.members.values()],
        "layers": [dict(l) for l in sorted_layers(room)],
        # The stroke dicts themselves are never mutated after they are
        # committed; the list they live in is appended to constantly.
        "strokes": list(room.strokes),
        "undone": list(room.undone),
        "aiState": ai_state,
    }


def _sanitize_points(
    raw: Any,
    canvas_size: int = CANVAS_SIZE,
    offset_x: float = 0,
    offset_y: float = 0,
) -> List[Point]:
    """Clean a run of points, clamped in *world* space.

    Points arrive in layer space, and a layer's offset reaches +-2 canvases, so
    clamping the layer-space number is clamping the wrong quantity: on a moved
    layer it silently drags the player's mark somewhere else (draw at world 100
    on a layer at offset 1500 and the browser sends -1400, which a layer-space
    clamp turns into -1024 - world 476). The limit belongs where the drawing
    is, so it is applied to `local + offset` and converted back.
    """
    if not isinstance(raw, list):
        return []
    out: List[Point] = []
    for p in raw[:20000]:
        if not isinstance(p, dict):
            continue
        x, y, pressure = p.get("x"), p.get("y"), p.get("p")
        if not _finite(x) or not _finite(y):
            continue
        point: Point = {
            "x": _clamp(x + offset_x, -canvas_size, 2 * canvas_size) - offset_x,
            "y": _clamp(y + offset_y, -canvas_size, 2 * canvas_size) - offset_y,
        }
        if _finite(pressure):
            point["p"] = _clamp(pressure, 0, 1)
        out.append(point)
    return out


def _world_bbox(room: RoomState, stroke: Stroke) -> Rect:
    """A committed stroke's box in world space (its layer may have been moved)."""
    layer = find_layer(room, stroke["layerId"])
    return translate_rect(
        stroke["bbox"],
        (layer or {}).get("offsetX", 0) or 0,
        (layer or {}).get("offsetY", 0) or 0,
    )


def _layer_dirty(room: RoomState, layer: Layer) -> List[Rect]:
    if layer["kind"] == "reference":
        w = (layer.get("imageWidth") or 0) * (layer.get("scale") or 1)
        h = (layer.get("imageHeight") or 0) * (layer.get("scale") or 1)
        if w <= 0 or h <= 0:
            return []
        return [{"x": layer.get("x") or 0, "y": layer.get("y") or 0, "width": w, "height": h}]
    boxes = [
        translate_rect(s["bbox"], layer.get("offsetX") or 0, layer.get("offsetY") or 0)
        for s in room.strokes
        if s["layerId"] == layer["id"] and s["id"] not in room.undone
    ]
    u = union_rects(boxes)
    return [u] if u else []


def apply_client_message(room: RoomState, user_id: str, msg: Message) -> ApplyResult:
    t = msg["t"]

    if t == "cursor":
        if not _finite(msg.get("x")) or not _finite(msg.get("y")):
            return _empty()
        return ApplyResult(relay=[{"t": "cursor", "userId": user_id, "x": msg["x"], "y": msg["y"]}])

    if t == "stroke_start":
        init = msg["stroke"]
        stroke_id = qualify_stroke_id(user_id, init["id"])
        # Every refusal echoes a stroke_cancel for this id so the sender drops
        # the preview it already started drawing locally.
        layer = find_layer(room, init["layerId"])
        if layer is None or layer["kind"] != "draw" or layer["locked"]:
            return _refuse_stroke(user_id, stroke_id, "cannot draw on that layer")
        if stroke_id in room.pending or any(s["id"] == stroke_id for s in room.strokes):
            return _refuse_stroke(user_id, stroke_id, "duplicate stroke id")
        mine = sum(1 for p in room.pending.values() if p.user_id == user_id)
        if mine >= MAX_PENDING_PER_USER:
            return _refuse_stroke(user_id, stroke_id, "too many strokes in progress")
        if len(room.strokes) >= MAX_STROKES_PER_ROOM:
            log.warning(
                "[room %s] stroke log full (%d); refusing new strokes", room.id, len(room.strokes)
            )
            return _refuse_stroke(user_id, stroke_id, "this room has reached its stroke limit")
        clean: StrokeInit = {
            "id": stroke_id,
            "layerId": layer["id"],
            "tool": init["tool"],
            "color": init["color"] if _is_hex_color(init.get("color")) else "#000000",
            "width": _clamp(init["width"], 1, 128),
            # The eraser has no opacity: it removes, or it is a different tool.
            "alpha": 1
            if init["tool"] == "eraser"
            else _clamp(
                init.get("alpha", DEFAULT_STROKE_ALPHA), MIN_STROKE_ALPHA, MAX_STROKE_ALPHA
            ),
            "points": _sanitize_points(
                init.get("points"),
                room.canvas_size,
                layer.get("offsetX") or 0,
                layer.get("offsetY") or 0,
            ),
        }
        if room_is_full(room, len(clean["points"])):
            return _refuse_stroke(user_id, stroke_id, "quota")
        now = now_ms()
        room.pending[stroke_id] = PendingStroke(
            user_id=user_id,
            init=clean,
            points=list(clean["points"]),
            started_at=now,
            last_activity_at=now,
        )
        room.pending_points += len(clean["points"])
        return ApplyResult(relay=[{"t": "stroke_start", "userId": user_id, "stroke": clean}])

    if t == "stroke_chunk":
        stroke_id = qualify_stroke_id(user_id, msg["strokeId"])
        p = room.pending.get(stroke_id)
        if p is None:
            return _empty()
        chunk_layer = find_layer(room, p.init["layerId"])
        points = _sanitize_points(
            msg.get("points"),
            room.canvas_size,
            (chunk_layer.get("offsetX") or 0) if chunk_layer else 0,
            (chunk_layer.get("offsetY") or 0) if chunk_layer else 0,
        )
        if (
            len(p.points) + len(points) > MAX_STROKE_POINTS
            or now_ms() - p.started_at > MAX_STROKE_MS
        ):
            drop_pending(room, stroke_id)
            return ApplyResult(
                broadcast=[
                    {
                        "t": "stroke_cancel",
                        "userId": user_id,
                        "strokeId": stroke_id,
                        "reason": "stroke too long",
                    }
                ],
                to_sender=[{"t": "error", "message": "stroke exceeded the point or time limit"}],
            )
        if room_is_full(room, len(points)):
            # Refused before the points are held, not after: the budget is
            # about what the room is allowed to occupy, and an unfinished
            # stroke occupies it exactly as much as a committed one.
            drop_pending(room, stroke_id)
            return ApplyResult(
                broadcast=[
                    {
                        "t": "stroke_cancel",
                        "userId": user_id,
                        "strokeId": stroke_id,
                        "reason": "quota",
                    }
                ]
            )
        p.points.extend(points)
        room.pending_points += len(points)
        p.last_activity_at = now_ms()
        return ApplyResult(
            relay=[
                {"t": "stroke_chunk", "userId": user_id, "strokeId": stroke_id, "points": points}
            ]
        )

    if t == "stroke_end":
        stroke_id = qualify_stroke_id(user_id, msg["strokeId"])
        p = room.pending.get(stroke_id)
        if p is None:
            return _empty()

        def _end(reason: str) -> ApplyResult:
            """Every way this stroke ends without committing."""
            drop_pending(room, stroke_id)
            return ApplyResult(
                broadcast=[
                    {
                        "t": "stroke_cancel",
                        "userId": user_id,
                        "strokeId": stroke_id,
                        "reason": reason,
                    }
                ]
            )

        layer = find_layer(room, p.init["layerId"])
        if layer is None or layer["kind"] != "draw":
            return _end("layer removed")
        # The layer may have been locked while this stroke was being drawn.
        if layer["locked"]:
            return _end("layer locked")

        tail = _sanitize_points(
            msg.get("points"),
            room.canvas_size,
            layer.get("offsetX") or 0,
            layer.get("offsetY") or 0,
        )
        # The tail has never been charged, and this stroke's own points still
        # are. Everyone else's pending points count too: checking only against
        # the committed log let a 100-point room hold 80 committed and 40
        # pending at once.
        if room_is_full(room, len(tail)):
            return _end("quota")
        # Past the check: the stroke's points stop being pending here, whether
        # they are committed below or refused for another reason.
        drop_pending(room, stroke_id)
        p.points.extend(tail)
        if len(p.points) == 0 or len(p.points) > MAX_STROKE_POINTS:
            return ApplyResult(
                broadcast=[
                    {
                        "t": "stroke_cancel",
                        "userId": user_id,
                        "strokeId": stroke_id,
                        "reason": "empty or oversized stroke",
                    }
                ]
            )
        stroke: Stroke = {
            "id": stroke_id,
            "userId": user_id,
            "layerId": p.init["layerId"],
            "tool": p.init["tool"],
            "color": p.init["color"],
            "width": p.init["width"],
            "alpha": p.init.get("alpha", DEFAULT_STROKE_ALPHA),
            "points": p.points,
            "revision": room.human_revision + 1,
            "bbox": stroke_bbox(p.points, p.init["width"]),
        }
        added_bytes = stroke_snapshot_bytes(stroke)
        if (
            room.committed_points + room.pending_points + len(p.points) > room.max_points
            or room.snapshot_bytes + added_bytes > room.max_snapshot_bytes
        ):
            # The room is full. Cancelling is the honest answer: the client
            # drops its optimistic preview instead of showing a stroke the
            # server does not have. Nothing has been mutated at this point -
            # not the log, not the revision, and the pending charge is already
            # released, which is what ends the stroke.
            return ApplyResult(
                broadcast=[
                    {
                        "t": "stroke_cancel",
                        "userId": user_id,
                        "strokeId": stroke_id,
                        "reason": "quota",
                    }
                ]
            )
        room.human_revision += 1
        room.strokes.append(stroke)
        room.committed_points += len(p.points)
        room.snapshot_bytes += added_bytes
        return ApplyResult(
            broadcast=[
                {"t": "stroke_committed", "stroke": stroke, "humanRevision": room.human_revision}
            ],
            relay=[
                {"t": "stroke_end", "userId": user_id, "strokeId": stroke_id, "points": tail}
            ],
            dirty=[
                translate_rect(
                    stroke["bbox"], layer.get("offsetX") or 0, layer.get("offsetY") or 0
                )
            ],
        )

    if t == "undo":
        target: Optional[Stroke] = None
        for s in reversed(room.strokes):
            if s["userId"] == user_id and s["id"] not in room.undone:
                target = s
                break
        if target is None:
            return _empty()
        room.undone.add(target["id"])
        room.human_revision += 1
        return ApplyResult(
            broadcast=[
                {
                    "t": "undo_applied",
                    "strokeId": target["id"],
                    "layerId": target["layerId"],
                    "humanRevision": room.human_revision,
                }
            ],
            dirty=[_world_bbox(room, target)],
        )

    if t == "clear_layer":
        layer = find_layer(room, msg["layerId"])
        if layer is None:
            return _refuse("unknown layer")
        removed = [s for s in room.strokes if s["layerId"] == layer["id"]]
        # union of what was actually *visible*, computed before `undone` changes
        visible = [
            translate_rect(s["bbox"], layer.get("offsetX") or 0, layer.get("offsetY") or 0)
            for s in removed
            if s["id"] not in room.undone
        ]
        u = union_rects(visible)
        room.strokes = [s for s in room.strokes if s["layerId"] != layer["id"]]
        room.committed_points -= sum(len(s["points"]) for s in removed)
        room.snapshot_bytes -= sum(stroke_snapshot_bytes(s) for s in removed)
        for s in removed:
            room.undone.discard(s["id"])
        cancels = _cancel_pending_on_layer(room, layer["id"])
        room.human_revision += 1
        return ApplyResult(
            broadcast=[
                {"t": "clear_applied", "layerId": layer["id"], "humanRevision": room.human_revision},
                *cancels,
            ],
            dirty=[u] if u else [],
        )

    if t == "layer_create":
        if len(room.layers) >= MAX_LAYERS:
            return _refuse("layer limit reached")
        kind = msg["layer"]["kind"]
        max_order = max([l["order"] for l in room.layers], default=-1)
        image = (
            room.images.get(msg["layer"]["imageId"])
            if kind == "reference" and msg["layer"].get("imageId")
            else None
        )
        if kind == "reference" and image is None:
            return _refuse("unknown imageId")
        name = msg["layer"].get("name")
        layer: Layer = {
            "id": short_id(6),
            "name": (name[:32] if isinstance(name, str) and name else None)
            or ("Reference" if kind == "reference" else f"Layer {len(room.layers) + 1}"),
            "kind": kind,
            "visible": True,
            "locked": False,
            "opacity": 1,
            "order": max_order + 1,
            "includeInAI": kind == "draw",
        }
        if image is not None:
            layer["imageId"] = image.id
            layer["imageWidth"] = image.width
            layer["imageHeight"] = image.height
            layer["x"] = msg["layer"]["x"] if _finite(msg["layer"].get("x")) else 0
            layer["y"] = msg["layer"]["y"] if _finite(msg["layer"].get("y")) else 0
            layer["scale"] = _clamp(
                msg["layer"]["scale"] if _finite(msg["layer"].get("scale")) else 1, 0.05, 8
            )
        room.layers.append(layer)
        room.human_revision += 1
        return ApplyResult(
            broadcast=[{"t": "layer_created", "layer": layer, "humanRevision": room.human_revision}],
            dirty=_layer_dirty(room, layer) if layer["includeInAI"] else [],
        )

    if t == "layer_update":
        layer = find_layer(room, msg["id"])
        if layer is None:
            return _refuse("unknown layer")
        patch = msg["patch"]
        was_included_in_ai = layer["includeInAI"]
        before_dirty = _layer_dirty(room, layer)

        # `name` and `locked` change nothing about the pixels. Renaming a layer
        # must not spend a generation.
        renders_differently = False

        def set_render(key: str, value: Any) -> None:
            nonlocal renders_differently
            if layer.get(key) == value:
                return
            layer[key] = value
            renders_differently = True

        # Validate the WHOLE patch before touching the layer. A refusal must
        # leave nothing behind: `{locked: true, offsetX: 10}` used to lock the
        # layer and then refuse, with no revision and no broadcast, so every
        # client disagreed with the server about the lock.
        # Unlocking in the same update is still allowed - the prospective lock
        # is what the patch asks for, not what the layer currently says.
        transforms = ("x", "y", "scale", "offsetX", "offsetY")
        will_be_locked = (
            patch["locked"] if isinstance(patch.get("locked"), bool) else layer["locked"]
        )
        if will_be_locked and any(_finite(patch.get(k)) for k in transforms):
            return _refuse("layer is locked")

        if isinstance(patch.get("name"), str):
            layer["name"] = patch["name"][:32]
        if isinstance(patch.get("locked"), bool):
            layer["locked"] = patch["locked"]
        if isinstance(patch.get("visible"), bool):
            set_render("visible", patch["visible"])
        if _finite(patch.get("opacity")):
            set_render("opacity", _clamp(patch["opacity"], 0, 1))
        if isinstance(patch.get("includeInAI"), bool) and layer["kind"] == "reference":
            set_render("includeInAI", patch["includeInAI"])
        offset_moved = False
        if layer["kind"] == "draw":
            # Offsets move existing strokes; the log keeps its coordinates.
            limit = 2 * CANVAS_SIZE
            before = {"x": layer.get("offsetX") or 0, "y": layer.get("offsetY") or 0}
            if _finite(patch.get("offsetX")):
                set_render("offsetX", _clamp(patch["offsetX"], -limit, limit))
            if _finite(patch.get("offsetY")):
                set_render("offsetY", _clamp(patch["offsetY"], -limit, limit))
            offset_moved = (layer.get("offsetX") or 0) != before["x"] or (
                layer.get("offsetY") or 0
            ) != before["y"]
        if layer["kind"] == "reference":
            if _finite(patch.get("x")):
                set_render("x", patch["x"])
            if _finite(patch.get("y")):
                set_render("y", patch["y"])
            if _finite(patch.get("scale")):
                set_render("scale", _clamp(patch["scale"], 0.05, 8))

        room.human_revision += 1
        after_dirty = _layer_dirty(room, layer)
        # Turning "AI input" off still has to repaint where the layer used to be.
        affects_ai = renders_differently and (was_included_in_ai or layer["includeInAI"])
        # A stroke in progress was aimed at where the layer WAS, so it is
        # cancelled rather than committed across two coordinate systems.
        cancels = _cancel_pending_on_layer(room, layer["id"], "layer moved") if offset_moved else []
        return ApplyResult(
            broadcast=[
                {"t": "layer_updated", "layer": layer, "humanRevision": room.human_revision},
                *cancels,
            ],
            dirty=[*before_dirty, *after_dirty] if affects_ai else [],
        )

    if t == "layer_delete":
        layer = find_layer(room, msg["id"])
        if layer is None:
            return _refuse("unknown layer")
        if layer["kind"] == "draw" and len([l for l in room.layers if l["kind"] == "draw"]) <= 1:
            return _refuse("cannot delete the last draw layer")
        dirty = _layer_dirty(room, layer)
        cancels = _cancel_pending_on_layer(room, layer["id"])
        room.layers = [l for l in room.layers if l["id"] != layer["id"]]
        dropped = [s for s in room.strokes if s["layerId"] == layer["id"]]
        for s in dropped:
            room.undone.discard(s["id"])
        room.strokes = [s for s in room.strokes if s["layerId"] != layer["id"]]
        # Deleting a layer gives its budget back: those strokes are gone from
        # the log and from every future snapshot.
        room.committed_points -= sum(len(s["points"]) for s in dropped)
        room.snapshot_bytes -= sum(stroke_snapshot_bytes(s) for s in dropped)
        room.human_revision += 1
        return ApplyResult(
            broadcast=[
                {"t": "layer_deleted", "id": layer["id"], "humanRevision": room.human_revision},
                *cancels,
            ],
            dirty=dirty if layer["includeInAI"] else [],
        )

    if t == "layer_reorder":
        known = [i for i in msg["ids"] if find_layer(room, i)]
        if len(known) != len(room.layers):
            return _refuse("layer_reorder must list every layer exactly once")
        for index, layer_id in enumerate(known):
            found = find_layer(room, layer_id)
            assert found is not None
            found["order"] = index
        room.human_revision += 1
        dirty: List[Rect] = []
        for l in room.layers:
            if l["includeInAI"]:
                dirty.extend(_layer_dirty(room, l))
        u = union_rects(dirty)
        return ApplyResult(
            broadcast=[
                {
                    "t": "layers_reordered",
                    "layers": sorted_layers(room),
                    "humanRevision": room.human_revision,
                }
            ],
            dirty=[u] if u else [],
        )

    if t == "set_ai_settings":
        # Both are room-level, so a change behaves exactly like a prompt change.
        if "denoise" in msg and clamp_denoise(msg["denoise"]) > room.max_denoise:
            return _refuse(f"this backend supports denoise up to {_fmt(room.max_denoise)}")
        denoise = room.denoise if "denoise" not in msg else clamp_denoise(msg["denoise"])
        negative_prompt = (
            room.negative_prompt
            if "negativePrompt" not in msg
            else msg["negativePrompt"][:MAX_NEGATIVE_PROMPT]
        )
        # In patch mode the generation size is the crop window, so accepting a
        # change here would broadcast a setting that silently does nothing.
        if "aiResolution" in msg and not room.ai_resolution_adjustable:
            return _refuse("the AI resolution is fixed in patch mode")
        if "seed" in msg and not is_seed(msg["seed"]):
            return _refuse(f"seed must be a whole number between 0 and {MAX_SEED}")
        seed = room.seed if "seed" not in msg else clamp_seed(msg["seed"])
        if "aiProfile" in msg and msg["aiProfile"] not in AI_PROFILES:
            return _refuse("aiProfile must be fast or quality")
        if "aiProfile" in msg and msg["aiProfile"] not in room.ai_profiles:
            return _refuse(
                f"this backend only supports the {' and '.join(room.ai_profiles)} profile"
            )
        if (
            "aiResolution" in msg
            and clamp_resolution(msg["aiResolution"], MAX_AI_RESOLUTION) > room.ai_resolution_max
        ):
            # Clamping silently is how a version-skewed client ends up drawing
            # at a size it never asked for, with no way to tell.
            return _refuse(f"this room can generate at up to {room.ai_resolution_max}")
        ai_profile = msg.get("aiProfile", room.ai_profile)
        # Switching profile also moves the generation size to that profile's
        # default. An explicit aiResolution in the same message still wins.
        profile_switched = ai_profile != room.ai_profile
        requested_resolution = msg.get(
            "aiResolution",
            PROFILE_DEFAULTS[ai_profile]["resolution"] if profile_switched else room.ai_resolution,
        )
        ai_resolution = (
            clamp_resolution(requested_resolution, room.ai_resolution_max)
            if room.ai_resolution_adjustable
            else room.ai_resolution
        )
        if (
            denoise == room.denoise
            and negative_prompt == room.negative_prompt
            and ai_resolution == room.ai_resolution
            and ai_profile == room.ai_profile
            and seed == room.seed
        ):
            return _empty()
        room.seed = seed
        room.denoise = denoise
        room.negative_prompt = negative_prompt
        room.ai_resolution = ai_resolution
        room.ai_profile = ai_profile
        return ApplyResult(
            broadcast=[
                {
                    "t": "ai_settings_changed",
                    "denoise": denoise,
                    "negativePrompt": negative_prompt,
                    "aiResolution": ai_resolution,
                    "aiProfile": ai_profile,
                    "negativePromptActive": room.negative_active.get(ai_profile, True) is not False,
                    "seed": seed,
                }
            ],
            prompt_changed=True,
        )

    if t == "set_prompt":
        prompt = msg["prompt"][:800]
        if prompt == room.prompt:
            return _empty()
        room.prompt = prompt
        return ApplyResult(
            broadcast=[{"t": "prompt_changed", "prompt": prompt}], prompt_changed=True
        )

    return _empty()


def _fmt(value: float) -> str:
    """JS number-to-string, so an error message reads 0.95 rather than 0.950."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(round(value, 12))


@dataclass
class RenderSnapshot:
    """Everything a render needs, captured synchronously so an AI request cannot
    mix state from two different revisions while it awaits."""

    revision: int
    prompt: str
    denoise: float
    negative_prompt: str
    ai_resolution: int
    ai_profile: str
    seed: int
    layers: List[Layer]
    strokes: List[Stroke]
    undone: Set[str]
    images: Dict[str, RoomImage]


def capture_render_snapshot(room: RoomState) -> RenderSnapshot:
    return RenderSnapshot(
        revision=room.human_revision,
        prompt=room.prompt,
        denoise=room.denoise,
        negative_prompt=room.negative_prompt,
        ai_resolution=room.ai_resolution,
        ai_profile=room.ai_profile,
        seed=room.seed,
        layers=[dict(l) for l in sorted_layers(room)],
        strokes=list(room.strokes),
        undone=set(room.undone),
        images=dict(room.images),
    )


def strokes_for_crop(
    source: RenderSnapshot,
    crop: Rect,
    layer_id: Optional[str] = None,
    offset: Optional[Dict[str, float]] = None,
) -> List[Stroke]:
    """Strokes intersecting a crop, in log order, skipping undone ones."""
    offset = offset or {"x": 0, "y": 0}
    out: List[Stroke] = []
    for s in source.strokes:
        if s["id"] in source.undone:
            continue
        if layer_id is not None and s["layerId"] != layer_id:
            continue
        box = translate_rect(s["bbox"], offset["x"], offset["y"])
        if (
            box["x"] < crop["x"] + crop["width"]
            and crop["x"] < box["x"] + box["width"]
            and box["y"] < crop["y"] + crop["height"]
            and crop["y"] < box["y"] + box["height"]
        ):
            out.append(s)
    return out
