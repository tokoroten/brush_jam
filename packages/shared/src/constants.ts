/** Logical world size in pixels. Square. */
export const CANVAS_SIZE = 4096;
/** Hard cap on layers per room (context doc: "roughly 4-8"). */
export const MAX_LAYERS = 8;

/**
 * Stroke opacity. Zero would be an invisible stroke that still costs a
 * generation, so the slider stops short of it.
 */
export const MIN_STROKE_ALPHA = 0.05;
export const MAX_STROKE_ALPHA = 1;
export const DEFAULT_STROKE_ALPHA = 1;
/** Two dirty regions merge when their bboxes, expanded by this, intersect. */
export const DIRTY_MERGE_PADDING = 256;
/** Mask dilation around dirty strokes, in world px. */
export const MASK_DILATE = 48;
/** Mask feather (blur radius), in world px. */
export const MASK_FEATHER = 32;
/** Max long-side of a pasted reference image before upload. */
export const MAX_PASTE_SIZE = 2048;
export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 8;
export const DEFAULT_NEGATIVE_PROMPT =
  'lowres, bad anatomy, bad hands, text, error, worst quality, low quality, jpeg artifacts, signature, watermark, blurry';
export const QUALITY_SUFFIX = ', masterpiece, best quality';
/** Room-adjustable img2img strength. */
export const MIN_DENOISE = 0.2;
export const MAX_DENOISE = 0.95;
export const DENOISE_STEP = 0.05;
export const MAX_NEGATIVE_PROMPT = 1000;
/**
 * Room-level speed/quality choice. Measured on an RTX 3070 at 1024:
 * quality (14-step euler_a) ~10 s, fast (4-step LCM) ~5.7 s, and fast at 768
 * ~3.7 s - see docs/experiments/2026-09-05-comfyui/REPORT.md.
 */
export const AI_PROFILES = ['fast', 'quality'] as const;
export type AIProfileName = (typeof AI_PROFILES)[number];

/** Per-profile defaults a new room starts from. */
export const PROFILE_DEFAULTS: Record<AIProfileName, { resolution: number; denoise: number; steps: number }> = {
  fast: { resolution: 768, denoise: 0.7, steps: 4 },
  quality: { resolution: 1024, denoise: 0.7, steps: 14 },
};

/**
 * Fallback hints shown before this room has measured anything, in ms. Measured
 * end-to-end medians: stream worker at 768 (docs/experiments/2026-09-05-stream/
 * REPORT.md) and ComfyUI 14-step at 1024 (docs/experiments/2026-09-05-comfyui/
 * REPORT.md). The room replaces these with its own timings after one run.
 */
export const PROFILE_HINT_MS: Record<AIProfileName, number> = { fast: 2400, quality: 10_300 };

/** Generation resolutions a room may pick from (bounded by AI_WINDOW). */
export const AI_RESOLUTIONS = [512, 768, 1024] as const;
export const MIN_AI_RESOLUTION = 512;
export const MAX_AI_RESOLUTION = 2048;

/**
 * Close code for a connection replaced by a newer one with the same identity.
 * In the application range on purpose: a takeover is not a transport failure,
 * and reconnecting on it is how two tabs sharing a session token evict each
 * other for as long as they are both open.
 */
export const CLOSE_SUPERSEDED = 4001;
/** Close code for "this server is full" - retried, but slowly. */
export const CLOSE_CAPACITY = 1013;
