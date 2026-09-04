import { MASK_DILATE, MASK_FEATHER } from './constants.js';
import { expandRect, intersectRect, type Rect } from './geometry.js';

export interface MaskPlan {
  /** Rects to paint white, in crop-local pixel coordinates. */
  shapes: Rect[];
  /** The authoritative area, in crop-local pixel coordinates. */
  apply: Rect;
  /** Blur radius in crop-local pixels. */
  feather: number;
  /** True when nothing at all would be regenerated. */
  empty: boolean;
}

/**
 * Pure mask geometry: dirty stroke bboxes -> dilated, crop-local rects clipped to
 * the crop, plus the apply rect the result is limited to (world space in, crop
 * space out). Rasterisation happens on the server; only the geometry is shared.
 */
export function planMask(
  dirty: readonly Rect[],
  crop: Rect,
  apply: Rect,
  dilate: number = MASK_DILATE,
  feather: number = MASK_FEATHER,
): MaskPlan {
  const shapes: Rect[] = [];
  for (const d of dirty) {
    const clipped = intersectRect(expandRect(d, dilate), crop);
    if (!clipped) continue;
    shapes.push({ x: clipped.x - crop.x, y: clipped.y - crop.y, width: clipped.width, height: clipped.height });
  }
  const localApply = intersectRect(apply, crop) ?? { x: crop.x, y: crop.y, width: 0, height: 0 };
  return {
    shapes,
    apply: { x: localApply.x - crop.x, y: localApply.y - crop.y, width: localApply.width, height: localApply.height },
    feather,
    empty: shapes.length === 0 || localApply.width <= 0 || localApply.height <= 0,
  };
}
