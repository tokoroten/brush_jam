export interface Rect { x: number; y: number; width: number; height: number }
export type Point = { x: number; y: number; p?: number };

export const rect = (x: number, y: number, width: number, height: number): Rect => ({ x, y, width, height });

export function rectRight(r: Rect): number { return r.x + r.width; }
export function rectBottom(r: Rect): number { return r.y + r.height; }

export function rectCenter(r: Rect): { x: number; y: number } {
  return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
}

export function expandRect(r: Rect, by: number): Rect {
  return { x: r.x - by, y: r.y - by, width: r.width + by * 2, height: r.height + by * 2 };
}

export function rectsIntersect(a: Rect, b: Rect): boolean {
  return a.x < rectRight(b) && b.x < rectRight(a) && a.y < rectBottom(b) && b.y < rectBottom(a);
}

export function rectContains(outer: Rect, inner: Rect): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    rectRight(inner) <= rectRight(outer) &&
    rectBottom(inner) <= rectBottom(outer)
  );
}

export function unionRect(a: Rect, b: Rect): Rect {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return { x, y, width: Math.max(rectRight(a), rectRight(b)) - x, height: Math.max(rectBottom(a), rectBottom(b)) - y };
}

export function unionRects(rects: readonly Rect[]): Rect | null {
  if (rects.length === 0) return null;
  let acc = rects[0]!;
  for (let i = 1; i < rects.length; i++) acc = unionRect(acc, rects[i]!);
  return acc;
}

export function intersectRect(a: Rect, b: Rect): Rect | null {
  const x = Math.max(a.x, b.x);
  const y = Math.max(a.y, b.y);
  const r = Math.min(rectRight(a), rectRight(b));
  const bo = Math.min(rectBottom(a), rectBottom(b));
  if (r <= x || bo <= y) return null;
  return { x, y, width: r - x, height: bo - y };
}

/** Clamp a rect so it stays inside [0,bounds]^2 without changing its size (size is clipped if larger). */
export function clampRectInside(r: Rect, boundsWidth: number, boundsHeight: number): Rect {
  const width = Math.min(r.width, boundsWidth);
  const height = Math.min(r.height, boundsHeight);
  const x = Math.min(Math.max(r.x, 0), boundsWidth - width);
  const y = Math.min(Math.max(r.y, 0), boundsHeight - height);
  return { x, y, width, height };
}

export function roundRectValues(r: Rect): Rect {
  const x = Math.floor(r.x);
  const y = Math.floor(r.y);
  return { x, y, width: Math.ceil(rectRight(r)) - x, height: Math.ceil(rectBottom(r)) - y };
}

/** Bounding box of a stroke, padded by half its width (plus 1 for antialiasing). */
export function strokeBBox(points: readonly Point[], width: number): Rect {
  if (points.length === 0) return { x: 0, y: 0, width: 0, height: 0 };
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const pt of points) {
    if (pt.x < minX) minX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y > maxY) maxY = pt.y;
  }
  const pad = width / 2 + 1;
  return { x: minX - pad, y: minY - pad, width: maxX - minX + pad * 2, height: maxY - minY + pad * 2 };
}
