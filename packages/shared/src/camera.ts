import { MAX_ZOOM, MIN_ZOOM } from './constants.js';

export interface Camera { centerX: number; centerY: number; zoom: number }

export const clampZoom = (z: number): number => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));

export function worldToScreen(cam: Camera, viewW: number, viewH: number, wx: number, wy: number): { x: number; y: number } {
  return { x: (wx - cam.centerX) * cam.zoom + viewW / 2, y: (wy - cam.centerY) * cam.zoom + viewH / 2 };
}

export function screenToWorld(cam: Camera, viewW: number, viewH: number, sx: number, sy: number): { x: number; y: number } {
  return { x: (sx - viewW / 2) / cam.zoom + cam.centerX, y: (sy - viewH / 2) / cam.zoom + cam.centerY };
}

/** Zoom by `factor` keeping the world point under (sx, sy) fixed. */
export function zoomAt(cam: Camera, viewW: number, viewH: number, sx: number, sy: number, factor: number): Camera {
  const zoom = clampZoom(cam.zoom * factor);
  if (zoom === cam.zoom) return cam;
  const before = screenToWorld(cam, viewW, viewH, sx, sy);
  const after = screenToWorld({ ...cam, zoom }, viewW, viewH, sx, sy);
  return { zoom, centerX: cam.centerX + (before.x - after.x), centerY: cam.centerY + (before.y - after.y) };
}

export function panBy(cam: Camera, dxScreen: number, dyScreen: number): Camera {
  return { ...cam, centerX: cam.centerX - dxScreen / cam.zoom, centerY: cam.centerY - dyScreen / cam.zoom };
}

export function fitCamera(viewW: number, viewH: number, canvasSize: number): Camera {
  const zoom = clampZoom(Math.min(viewW / canvasSize, viewH / canvasSize));
  return { centerX: canvasSize / 2, centerY: canvasSize / 2, zoom };
}
