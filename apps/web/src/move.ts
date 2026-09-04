import type { Layer, Point, Rect } from '@brushjam/shared';

/** Natural (unscaled) size of the image a reference layer points at. */
export type SizeLookup = (imageId: string) => { width: number; height: number } | undefined;

/** World-space rectangle a reference layer occupies, or null if unknown. */
export function layerRect(layer: Layer, sizeOf: SizeLookup): Rect | null {
  if (layer.kind !== 'reference' || !layer.imageId) return null;
  const size = sizeOf(layer.imageId);
  if (!size || size.width <= 0 || size.height <= 0) return null;
  const scale = layer.scale ?? 1;
  return { x: layer.x ?? 0, y: layer.y ?? 0, width: size.width * scale, height: size.height * scale };
}

/**
 * The reference layer to move for a pointer press: the topmost unlocked one
 * under the pointer, falling back to the selected reference layer (which is how
 * a freshly pasted image can be dragged before anyone has aimed at it).
 */
export function pickReferenceLayer(layers: readonly Layer[], sizeOf: SizeLookup, at: Point, selectedId?: string | null): Layer | null {
  const candidates = layers.filter((l) => l.kind === 'reference' && !l.locked && l.visible);
  // highest order = frontmost
  const byFront = [...candidates].sort((a, b) => b.order - a.order);
  for (const layer of byFront) {
    const rect = layerRect(layer, sizeOf);
    if (!rect) continue;
    if (at.x >= rect.x && at.x <= rect.x + rect.width && at.y >= rect.y && at.y <= rect.y + rect.height) return layer;
  }
  const selected = candidates.find((l) => l.id === selectedId);
  return selected ?? null;
}

/** Screen-space drag delta applied to a world position (world = screen / zoom). */
export function movedPosition(origin: Point, dxScreen: number, dyScreen: number, zoom: number): Point {
  const z = zoom > 0 ? zoom : 1;
  return { x: origin.x + dxScreen / z, y: origin.y + dyScreen / z };
}

export const MIN_LAYER_SCALE = 0.05;
export const MAX_LAYER_SCALE = 8;

/** Wheel-driven scaling, clamped to the same range as the panel slider. */
export function scaledBy(scale: number | undefined, factor: number): number {
  const next = (scale ?? 1) * factor;
  return Math.min(MAX_LAYER_SCALE, Math.max(MIN_LAYER_SCALE, Math.round(next * 100) / 100));
}
