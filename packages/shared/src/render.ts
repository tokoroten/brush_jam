import type { Point } from './geometry.js';
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
  points: Point[];
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
    strokePath(ctx, s, offsetX, offsetY);
  }
  ctx.restore();
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
  const box = strokeBounds(stroke);
  // target-space placement, clipped only when the caller declared the bounds
  let left = box.x - offsetX;
  let top = box.y - offsetY;
  let right = left + box.width;
  let bottom = top + box.height;
  if (bounds) {
    left = Math.max(0, left);
    top = Math.max(0, top);
    right = Math.min(bounds.width, right);
    bottom = Math.min(bounds.height, bottom);
  }
  const width = Math.ceil(right - left);
  const height = Math.ceil(bottom - top);
  if (width <= 0 || height <= 0) return;

  const temp = createCanvas(width, height);
  const tctx = temp.getContext('2d') as Ctx2DLike;
  tctx.save();
  tctx.lineCap = 'round';
  tctx.lineJoin = 'round';
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

  ctx.globalCompositeOperation = 'source-over';
  // drawImage (not putImageData) so the noise composites with what is below
  ctx.drawImage(temp as never, left, top);
}
