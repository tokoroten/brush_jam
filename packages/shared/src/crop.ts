import { rectCenter, roundRectValues, type Rect } from './geometry.js';

/**
 * Choose the square AI window for a dirty region: `size`x`size` centered on the
 * region, clamped to stay inside the canvas. No tiling in MVP - a dirty region
 * larger than the window simply gets its center regenerated first.
 */
export function chooseCrop(region: Rect, size: number, canvasSize: number): Rect {
  const side = Math.min(Math.round(size), Math.floor(canvasSize));
  const c = rectCenter(region);
  const clamp = (v: number): number => Math.min(Math.max(Math.round(v), 0), canvasSize - side);
  return { x: clamp(c.x - side / 2), y: clamp(c.y - side / 2), width: side, height: side };
}

/** The central, authoritative sub-rect of a crop (context vs applied region). */
export function applyRect(crop: Rect, applySize: number): Rect {
  const size = Math.min(applySize, crop.width, crop.height);
  const c = rectCenter(crop);
  return roundRectValues({ x: c.x - size / 2, y: c.y - size / 2, width: size, height: size });
}

/**
 * The apply rect for a specific dirty region. It is centered on the region
 * rather than on the crop, then clamped inside the crop, so a region sitting
 * against a canvas edge is still covered: with a fixed centered apply rect the
 * scheduler would repaint an area that never contains the dirty pixels and
 * would re-run the identical crop forever.
 */
export function applyRectFor(crop: Rect, applySize: number, region?: Rect): Rect {
  if (!region) return applyRect(crop, applySize);
  const size = Math.min(applySize, crop.width, crop.height);
  const c = rectCenter(region);
  const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(Math.round(v), lo), hi);
  return {
    x: clamp(c.x - size / 2, crop.x, crop.x + crop.width - size),
    y: clamp(c.y - size / 2, crop.y, crop.y + crop.height - size),
    width: size,
    height: size,
  };
}
