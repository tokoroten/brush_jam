import { clampRectInside, rectCenter, roundRectValues, type Rect } from './geometry.js';

/**
 * Choose the square AI window for a dirty region: `size`x`size` centered on the
 * region, clamped to stay inside the canvas. No tiling in MVP - a dirty region
 * larger than the window simply gets its center regenerated first.
 */
export function chooseCrop(region: Rect, size: number, canvasSize: number): Rect {
  const c = rectCenter(region);
  const raw: Rect = { x: c.x - size / 2, y: c.y - size / 2, width: size, height: size };
  return roundRectValues(clampRectInside(raw, canvasSize, canvasSize));
}

/** The central, authoritative sub-rect of a crop (context vs applied region). */
export function applyRect(crop: Rect, applySize: number): Rect {
  const size = Math.min(applySize, crop.width, crop.height);
  const c = rectCenter(crop);
  return roundRectValues({ x: c.x - size / 2, y: c.y - size / 2, width: size, height: size });
}
