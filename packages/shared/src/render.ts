import type { Point } from './geometry.js';

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
}

export interface RenderableStroke {
  id: string;
  tool: 'pen' | 'eraser';
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
  const { undone, offsetX = 0, offsetY = 0 } = options;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const s of strokes) {
    if (undone?.has(s.id)) continue;
    if (s.points.length === 0) continue;
    ctx.globalCompositeOperation = s.tool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    if (s.points.length === 1) {
      const p = s.points[0]!;
      ctx.beginPath();
      ctx.arc(p.x - offsetX, p.y - offsetY, Math.max(0.5, (s.width * pressureOf(p)) / 2), 0, Math.PI * 2);
      ctx.fill();
      continue;
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
  ctx.restore();
}
