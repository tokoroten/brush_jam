"""Mirror of packages/shared/src/constants.ts. Values must not drift."""

from __future__ import annotations

from typing import Dict, List, Literal

#: Logical world size in pixels. Square.
CANVAS_SIZE = 4096
#: Hard cap on layers per room.
MAX_LAYERS = 8

MIN_STROKE_ALPHA = 0.05
MAX_STROKE_ALPHA = 1.0
DEFAULT_STROKE_ALPHA = 1.0

DIRTY_MERGE_PADDING = 256
MASK_DILATE = 48
MASK_FEATHER = 32
MAX_PASTE_SIZE = 2048
MIN_ZOOM = 0.05
MAX_ZOOM = 8

DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, "
    "jpeg artifacts, signature, watermark, blurry"
)
QUALITY_SUFFIX = ", masterpiece, best quality"

MIN_DENOISE = 0.2
MAX_DENOISE = 0.95
DENOISE_STEP = 0.05
MAX_NEGATIVE_PROMPT = 1000

AIProfileName = Literal["fast", "quality"]
AI_PROFILES: List[str] = ["fast", "quality"]

PROFILE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "fast": {"resolution": 768, "denoise": 0.7, "steps": 4},
    "quality": {"resolution": 1024, "denoise": 0.7, "steps": 14},
}

PROFILE_HINT_MS: Dict[str, int] = {"fast": 2400, "quality": 10_300}

AI_RESOLUTIONS = [512, 768, 1024]
MIN_AI_RESOLUTION = 512
MAX_AI_RESOLUTION = 2048
