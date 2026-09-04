/** Logical world size in pixels. Square. */
export const CANVAS_SIZE = 4096;
/** Hard cap on layers per room (context doc: "roughly 4-8"). */
export const MAX_LAYERS = 8;
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
