import { translateRect, unionRects, type Layer, type Point, type Rect, type Stroke } from '@brushjam/shared';

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

/** World-space content box of a draw layer: its strokes, moved by its offset. */
export function drawLayerRect(layer: Layer, strokes: readonly Stroke[], undone: ReadonlySet<string>): Rect | null {
  if (layer.kind !== 'draw') return null;
  const boxes = strokes.filter((s) => s.layerId === layer.id && !undone.has(s.id)).map((s) => s.bbox);
  const union = unionRects(boxes);
  return union ? translateRect(union, layer.offsetX ?? 0, layer.offsetY ?? 0) : null;
}

export interface MoveContext {
  sizeOf: SizeLookup;
  strokes: readonly Stroke[];
  undone: ReadonlySet<string>;
}

/** Where a layer currently sits, whichever kind it is. */
export function layerOrigin(layer: Layer): Point {
  return layer.kind === 'reference' ? { x: layer.x ?? 0, y: layer.y ?? 0 } : { x: layer.offsetX ?? 0, y: layer.offsetY ?? 0 };
}

/** The layer_update patch that puts a layer at `at`. */
export function movePatch(layer: Layer, at: Point): Record<string, number> {
  return layer.kind === 'reference' ? { x: at.x, y: at.y } : { offsetX: at.x, offsetY: at.y };
}

/**
 * Any movable layer under the pointer - a reference by its image box, a draw
 * layer by the box of its strokes - topmost first, falling back to the
 * selection (of either kind) so a freshly pasted or empty layer can still move.
 */
export function pickMovableLayer(layers: readonly Layer[], ctx: MoveContext, at: Point, selectedId?: string | null): Layer | null {
  const movable = layers.filter((l) => !l.locked && l.visible);
  const boxOf = (layer: Layer): Rect | null =>
    layer.kind === 'reference' ? layerRect(layer, ctx.sizeOf) : drawLayerRect(layer, ctx.strokes, ctx.undone);
  for (const layer of [...movable].sort((a, b) => b.order - a.order)) {
    const rect = boxOf(layer);
    if (!rect) continue;
    if (at.x >= rect.x && at.x <= rect.x + rect.width && at.y >= rect.y && at.y <= rect.y + rect.height) return layer;
  }
  return movable.find((l) => l.id === selectedId) ?? null;
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

/**
 * World point -> layer-space point. Strokes are stored in layer space, so a
 * stroke drawn on a moved layer must have the offset removed before it is sent.
 */
export function layerPoint(world: Point, layer: Layer | undefined): Point {
  return { x: world.x - (layer?.offsetX ?? 0), y: world.y - (layer?.offsetY ?? 0) };
}

export const MIN_LAYER_SCALE = 0.05;
export const MAX_LAYER_SCALE = 8;

/** Wheel-driven scaling, clamped to the same range as the panel slider. */
export function scaledBy(scale: number | undefined, factor: number): number {
  const next = (scale ?? 1) * factor;
  return Math.min(MAX_LAYER_SCALE, Math.max(MIN_LAYER_SCALE, Math.round(next * 100) / 100));
}
