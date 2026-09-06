/**
 * The brush cursor: a ring the size of the brush, drawn where the pointer is.
 *
 * The size slider is the one control whose value cannot be read off the screen.
 * It is in world pixels and the canvas is nearly always zoomed out, so "40"
 * means anything between a hair and a fist depending on where the zoom happens
 * to be, and the only way to find out used to be to draw a stroke and undo it.
 * The ring answers it directly: it is drawn in screen space at exactly the size
 * the stroke will land, so the slider becomes something you can see.
 *
 * It replaces the OS cursor rather than joining it (`stageCursor` returns
 * `none` for the drawing tools): two pointers on one canvas, one of them lying
 * about where the paint goes, is worse than either alone.
 *
 * Geometry only - no canvas, no React - so the two rules that are easy to get
 * wrong are tested rather than eyeballed: the radius is half the *screen* width
 * of the brush, and a ring smaller than the line drawing it becomes a cross.
 */

export type BrushCursorTool = 'pen' | 'noise' | 'eraser' | 'move';

/**
 * Below this screen radius the ring is barely wider than its own 1 px stroke:
 * it reads as a dot, its centre is invisible, and it is no longer possible to
 * tell where the next stroke starts. A small cross says the same thing and
 * keeps the exact point.
 */
export const MIN_RING_RADIUS_PX = 3;

/** Half the width of that cross, in screen pixels. */
export const CROSSHAIR_ARM_PX = 4;

export interface BrushCursor {
  kind: 'ring' | 'crosshair';
  /** Screen pixels. Kept on the crosshair too, for the caller that wants it. */
  radius: number;
}

/**
 * The screen radius of a `width`-wide brush at `zoom`.
 *
 * Half, because `width` is a diameter: a stroke is drawn as a round cap of that
 * width, so the footprint under the pointer reaches half of it in every
 * direction. Never negative, and never NaN - a half-typed or missing number
 * must degrade to "no ring", not to a canvas exception in the frame loop.
 */
export function brushScreenRadius(width: number, zoom: number): number {
  if (!Number.isFinite(width) || !Number.isFinite(zoom)) return 0;
  return Math.max(0, (width * zoom) / 2);
}

/** The tools that put paint down, and therefore have a footprint to show. */
export const drawsWithBrush = (tool: BrushCursorTool): boolean =>
  tool === 'pen' || tool === 'noise' || tool === 'eraser';

/**
 * What to draw at the pointer, or null when the pointer is not a brush.
 *
 * Not while panning: Space turns every tool into the hand for as long as it is
 * held, and a ring left on screen would promise a stroke the drag is not going
 * to make.
 */
export function brushCursor(input: {
  tool: BrushCursorTool;
  width: number;
  zoom: number;
  panning: boolean;
}): BrushCursor | null {
  if (input.panning || !drawsWithBrush(input.tool)) return null;
  const radius = brushScreenRadius(input.width, input.zoom);
  return { kind: radius < MIN_RING_RADIUS_PX ? 'crosshair' : 'ring', radius };
}

/**
 * The CSS cursor for the stage.
 *
 * `none` exactly where the ring is drawn, because there the ring *is* the
 * cursor. Everything else keeps a pointer the OS drew: the AI stage and a held
 * Space only ever pan, and the move tool drags a layer, none of which has a
 * footprint to stand in for the arrow.
 */
export function stageCursor(input: {
  tool: BrushCursorTool;
  panning: boolean;
  /** The AI stage: a viewport, so every drag is a pan. */
  viewOnly?: boolean;
}): 'none' | 'grab' | 'default' {
  if (input.viewOnly === true || input.panning) return 'grab';
  return drawsWithBrush(input.tool) ? 'none' : 'default';
}
