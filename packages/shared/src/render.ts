import type { Point } from './geometry.js';
import { MAX_STROKE_ALPHA, MIN_STROKE_ALPHA } from './constants.js';
import { fnv1a, noiseHash } from './noise.js';

/** Structural subset of CanvasRenderingContext2D used by the shared renderer. */
export interface Ctx2DLike {
  save(): void;
  restore(): void;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, r: number, a0: number, a1: number): void;
  stroke(): void;
  fill(): void;
  globalCompositeOperation: string;
  globalAlpha: number;
  strokeStyle: unknown;
  fillStyle: unknown;
  lineWidth: number;
  lineCap: unknown;
  lineJoin: unknown;
  /** Present on both browser and @napi-rs contexts. */
  canvas?: { width: number; height: number };
  drawImage(image: never, dx: number, dy: number): void;
  getImageData(x: number, y: number, w: number, h: number): { data: { length: number; [i: number]: number } };
  putImageData(data: never, x: number, y: number): void;
  createImageData?(w: number, h: number): unknown;
}

/** Minimal canvas factory the noise pen needs (OffscreenCanvas / napi canvas). */
export interface CanvasLike {
  width: number;
  height: number;
  getContext(id: '2d'): unknown;
}

export interface RenderableStroke {
  id: string;
  tool: 'pen' | 'eraser' | 'noise';
  color: string;
  width: number;
  /** Opacity of the whole stroke, 0.05-1; absent means opaque. */
  alpha?: number;
  points: Point[];
}

/**
 * The eraser always removes fully - a half-strength eraser is a different
 * tool, not a setting - and anything else is clamped into range.
 */
export function alphaOf(stroke: { tool: string; alpha?: number }): number {
  if (stroke.tool === 'eraser') return 1;
  const raw = typeof stroke.alpha === 'number' ? stroke.alpha : 1;
  if (!Number.isFinite(raw)) return 1;
  return Math.max(MIN_STROKE_ALPHA, Math.min(MAX_STROKE_ALPHA, raw));
}

export interface RenderStrokesOptions {
  /** Stroke ids to skip (undone). */
  undone?: ReadonlySet<string>;
  /** Subtracted from every point, e.g. the crop origin. */
  offsetX?: number;
  offsetY?: number;
  /**
   * Required to draw noise strokes: they are rasterised into a temporary
   * canvas so each pixel can be replaced with its deterministic value.
   * Without it a noise stroke falls back to a plain stroke.
   */
  createCanvas?: (width: number, height: number) => CanvasLike;
  /**
   * Target size in *drawing* units, used to clip the noise pen's temporary
   * canvas. Only pass it when the context is untransformed and its pixels are
   * the drawing units (a layer raster): the screen stage is scaled and
   * translated, so its pixel size says nothing about which world coordinates
   * are visible - clamping against it silently dropped strokes on the right.
   */
  bounds?: { width: number; height: number };
}

/** World-space bounding box of a stroke's painted area. */
export function strokeBounds(stroke: RenderableStroke): { x: number; y: number; width: number; height: number } {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of stroke.points) {
    const r = Math.max(0.5, (stroke.width * pressureOf(p)) / 2) + 1;
    minX = Math.min(minX, p.x - r);
    minY = Math.min(minY, p.y - r);
    maxX = Math.max(maxX, p.x + r);
    maxY = Math.max(maxY, p.y + r);
  }
  return { x: Math.floor(minX), y: Math.floor(minY), width: Math.ceil(maxX - minX), height: Math.ceil(maxY - minY) };
}

const pressureOf = (p: Point): number => (typeof p.p === 'number' ? Math.max(0.05, Math.min(1, p.p)) : 1);

/**
 * Draw strokes onto any Canvas2D-compatible context. Deterministic: the same
 * strokes produce the same pixels on the server (@napi-rs/canvas) and in the
 * browser, which is what keeps the AI input identical to what people see.
 */
export function renderStrokes(
  ctx: Ctx2DLike,
  strokes: readonly RenderableStroke[],
  options: RenderStrokesOptions = {},
): void {
  const { undone, offsetX = 0, offsetY = 0, createCanvas, bounds } = options;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const s of strokes) {
    if (undone?.has(s.id)) continue;
    if (s.points.length === 0) continue;
    if (s.tool === 'noise' && createCanvas) {
      drawNoiseStroke(ctx, s, offsetX, offsetY, createCanvas, bounds);
      continue;
    }
    ctx.globalCompositeOperation = s.tool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    const alpha = alphaOf(s);
    if (alpha >= 1) {
      ctx.globalAlpha = 1;
      strokePath(ctx, s, offsetX, offsetY);
      continue;
    }
    // A stroke is one mark, so its opacity applies once. Drawing the segments
    // straight onto the target at globalAlpha would darken every join and
    // every place the stroke crosses itself - and the segments cannot simply
    // be merged into a single path, because pressure varies their width. So
    // the stroke is rasterised opaque and composited once, which is also
    // exactly what the live preview does.
    if (createCanvas) {
      drawFlattened(ctx, s, offsetX, offsetY, createCanvas, alpha, bounds);
      continue;
    }
    // No temp canvas available: the joins will darken, but the stroke is at
    // least the right colour and roughly the right strength.
    ctx.globalAlpha = alpha;
    strokePath(ctx, s, offsetX, offsetY);
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

/** The temp canvas a stroke is rasterised into, in target space. */
function tempFor(
  stroke: RenderableStroke,
  offsetX: number,
  offsetY: number,
  createCanvas: (width: number, height: number) => CanvasLike,
  bounds?: { width: number; height: number },
): { canvas: CanvasLike; ctx: Ctx2DLike; left: number; top: number; width: number; height: number } | null {
  const box = strokeBounds(stroke);
  let left = box.x - offsetX;
  let top = box.y - offsetY;
  let right = left + box.width;
  let bottom = top + box.height;
  if (bounds) {
    // Clipped to what the target can show, but with a margin: a pixel's
    // coverage depends on the geometry around it, so cutting the shape exactly
    // at the edge gave the boundary row a different value than the same stroke
    // rendered without a crop. One stroke width plus a pixel is all it takes,
    // and the target clips the excess when this is composited.
    const pad = Math.ceil(stroke.width) + 2;
    left = Math.max(-pad, left);
    top = Math.max(-pad, top);
    right = Math.min(bounds.width + pad, right);
    bottom = Math.min(bounds.height + pad, bottom);
  }
  const width = Math.ceil(right - left);
  const height = Math.ceil(bottom - top);
  if (width <= 0 || height <= 0) return null;
  const canvas = createCanvas(width, height);
  return { canvas, ctx: canvas.getContext('2d') as Ctx2DLike, left, top, width, height };
}

/**
 * Rasterise the stroke at full strength into its own canvas, then composite
 * that once at `alpha`. Overlaps within the stroke merge before the opacity is
 * applied, so a self-crossing stroke has one uniform strength throughout.
 */
function drawFlattened(
  ctx: Ctx2DLike,
  stroke: RenderableStroke,
  offsetX: number,
  offsetY: number,
  createCanvas: (width: number, height: number) => CanvasLike,
  alpha: number,
  bounds?: { width: number; height: number },
): void {
  const temp = tempFor(stroke, offsetX, offsetY, createCanvas, bounds);
  if (!temp) return;
  temp.ctx.save();
  temp.ctx.lineCap = 'round';
  temp.ctx.lineJoin = 'round';
  temp.ctx.globalAlpha = 1;
  temp.ctx.strokeStyle = stroke.color;
  temp.ctx.fillStyle = stroke.color;
  strokePath(temp.ctx, stroke, offsetX + temp.left, offsetY + temp.top);
  temp.ctx.restore();

  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = alpha;
  ctx.drawImage(temp.canvas as never, temp.left, temp.top);
  ctx.globalAlpha = 1;
}

/** The shape itself, with whatever style/composite the caller has set. */
function strokePath(ctx: Ctx2DLike, s: RenderableStroke, offsetX: number, offsetY: number): void {
  if (s.points.length === 1) {
    const p = s.points[0]!;
    ctx.beginPath();
    ctx.arc(p.x - offsetX, p.y - offsetY, Math.max(0.5, (s.width * pressureOf(p)) / 2), 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  for (let i = 1; i < s.points.length; i++) {
    const a = s.points[i - 1]!;
    const b = s.points[i]!;
    ctx.lineWidth = Math.max(0.5, s.width * ((pressureOf(a) + pressureOf(b)) / 2));
    ctx.beginPath();
    ctx.moveTo(a.x - offsetX, a.y - offsetY);
    ctx.lineTo(b.x - offsetX, b.y - offsetY);
    ctx.stroke();
  }
}

/**
 * Noise pen: rasterise the stroke shape into a temporary canvas, replace every
 * covered pixel's RGB with hash(seed, worldX, worldY) while keeping the shape's
 * antialiased alpha, then composite that onto the target. Seeded by the stroke
 * id and addressed in world coordinates, so a translated render (the server's
 * crop) yields identical pixels.
 */
function drawNoiseStroke(
  ctx: Ctx2DLike,
  stroke: RenderableStroke,
  offsetX: number,
  offsetY: number,
  createCanvas: (width: number, height: number) => CanvasLike,
  bounds?: { width: number; height: number },
): void {
  const temp = tempFor(stroke, offsetX, offsetY, createCanvas, bounds);
  if (!temp) return;
  const { ctx: tctx, left, top, width, height } = temp;
  tctx.save();
  tctx.lineCap = 'round';
  tctx.lineJoin = 'round';
  tctx.globalAlpha = 1;
  tctx.strokeStyle = '#000000';
  tctx.fillStyle = '#000000';
  // the temp canvas starts at (left, top) in target space
  strokePath(tctx, stroke, offsetX + left, offsetY + top);
  tctx.restore();

  const image = tctx.getImageData(0, 0, width, height);
  const data = image.data;
  const seed = fnv1a(stroke.id);
  // world coordinate of the temp canvas origin
  const worldX = Math.round(left + offsetX);
  const worldY = Math.round(top + offsetY);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      if (data[i + 3] === 0) continue;
      const h = noiseHash(seed, worldX + x, worldY + y);
      data[i] = h & 0xff;
      data[i + 1] = (h >>> 8) & 0xff;
      data[i + 2] = (h >>> 16) & 0xff;
    }
  }
  tctx.putImageData(image as never, 0, 0);

  // Composited once, at the stroke's own opacity: the noise is already
  // flattened here, so overlaps inside the stroke cannot accumulate.
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = alphaOf(stroke);
  ctx.drawImage(temp.canvas as never, left, top);
  ctx.globalAlpha = 1;
}
