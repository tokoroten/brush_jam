"""Mirror of packages/shared/src/protocol.ts.

The wire format is JSON exactly as the TypeScript types describe it, so these
are TypedDicts over plain dicts rather than pydantic models: messages are built
and sent verbatim, and `validate.py` is a hand port of `validate.ts` so the
error strings a client sees are identical too.

Optional fields follow the TS semantics precisely: a key that is `undefined` in
TypeScript is *absent* from the JSON, so it must be absent here as well (a
`None` would serialise as `null`, which the client does not expect).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from .geometry import Point, Rect

Tool = Literal["pen", "eraser", "noise"]
LayerKind = Literal["draw", "reference"]
AIState = Literal["idle", "queued", "generating", "error"]


class Layer(TypedDict, total=False):
    id: str
    name: str
    kind: str
    visible: bool
    locked: bool
    opacity: float
    order: int
    includeInAI: bool
    offsetX: float
    offsetY: float
    imageId: str
    x: float
    y: float
    scale: float
    imageWidth: int
    imageHeight: int


class StrokeInit(TypedDict, total=False):
    id: str
    layerId: str
    tool: str
    color: str
    width: float
    alpha: float
    points: List[Point]


class Stroke(TypedDict, total=False):
    id: str
    userId: str
    layerId: str
    tool: str
    color: str
    width: float
    alpha: float
    points: List[Point]
    revision: int
    bbox: Rect


class Member(TypedDict):
    userId: str
    name: str
    color: str


class RoomSnapshot(TypedDict):
    roomId: str
    youUserId: str
    prompt: str
    humanRevision: int
    aiRevision: int
    aiGeneration: int
    canvasSize: int
    aiWindow: int
    aiApply: int
    denoise: float
    aiProfile: str
    aiProfiles: List[str]
    maxDenoise: float
    negativePromptActive: bool
    negativePrompt: str
    aiResolution: int
    aiResolutionMax: int
    aiResolutionAdjustable: bool
    #: Whether the picker offers the R18 preset group (PRESETS_R18). The server
    #: still accepts any prompt anybody types; this only decides what the UI
    #: puts in front of them.
    r18Presets: bool
    members: List[Member]
    layers: List[Layer]
    strokes: List[Stroke]
    undone: List[str]
    aiState: str


#: A client -> server or server -> client frame, as the JSON dict it is on the
#: wire. Always carries a `t` discriminator.
Message = Dict[str, Any]

CLIENT_MESSAGE_TYPES = (
    "cursor",
    "stroke_start",
    "stroke_chunk",
    "stroke_end",
    "undo",
    "clear_layer",
    "layer_create",
    "layer_update",
    "layer_delete",
    "layer_reorder",
    "set_prompt",
    "set_ai_settings",
)

SERVER_MESSAGE_TYPES = (
    "snapshot",
    "presence",
    "cursor",
    "stroke_start",
    "stroke_chunk",
    "stroke_end",
    "stroke_committed",
    "stroke_cancel",
    "undo_applied",
    "clear_applied",
    "layer_created",
    "layer_updated",
    "layer_deleted",
    "layers_reordered",
    "prompt_changed",
    "ai_settings_changed",
    "ai_capabilities",
    "ai_status",
    "ai_result",
    "error",
)
