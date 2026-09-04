import { MASK_DILATE, MASK_FEATHER } from './constants.js';
import { applyRect } from './crop.js';
import { expandRect, intersectRect, type Rect } from './geometry.js';

export interface MaskPlan {
  /** Rects to paint white, in crop-local pixel coordinates. */
  shapes: Rect[];
  /** The authoritative central area, in crop-local pixel coordinates. */
  apply: Rect;
  /** Blur radius in crop-local pixels. */
  feather: number;
  /** True when nothing at all would be regenerated. */
  empty: boolean;
}

/**
 * Pure mask geometry: dirty stroke bboxes -> dilated, crop-local rects clipped to
 * the crop, plus the central apply rect the result is limited to. Rasterisation
 * (blur, compositing) happens on the server; only the geometry is shared/tested.
 */
export function planMask(
  dirty: readonly Rect[],
  crop: Rect,
  applySize: number,
  dilate: number = MASK_DILATE,
  feather: number = MASK_FEATHER,
): MaskPlan {
  const apply = applyRect(crop, applySize);
  const shapes: Rect[] = [];
  for (const d of dirty) {
    const clipped = intersectRect(expandRect(d, dilate), crop);
    if (!clipped) continue;
    shapes.push({ x: clipped.x - crop.x, y: clipped.y - crop.y, width: clipped.width, height: clipped.height });
  }
  return {
    shapes,
    apply: { x: apply.x - crop.x, y: apply.y - crop.y, width: apply.width, height: apply.height },
    feather,
    empty: shapes.length === 0,
  };
}
