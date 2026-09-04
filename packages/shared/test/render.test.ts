import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it } from 'vitest';
import { renderStrokes, type RenderableStroke } from '../src/index.js';

const strokes: RenderableStroke[] = [
  { id: 'a', tool: 'pen', color: '#ff0000', width: 24, points: [{ x: 20, y: 20 }, { x: 100, y: 60 }, { x: 140, y: 30 }] },
  { id: 'b', tool: 'pen', color: '#0000ff', width: 8, points: [{ x: 40, y: 100 }] },
  { id: 'c', tool: 'eraser', color: '#000000', width: 20, points: [{ x: 20, y: 20 }, { x: 60, y: 40 }] },
];

function paint(list: RenderableStroke[], undone?: Set<string>): Buffer {
  const canvas = createCanvas(200, 160);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 200, 160);
  renderStrokes(ctx as never, list, undone ? { undone } : {});
  return canvas.toBuffer('image/png');
}

describe('renderStrokes', () => {
  it('is deterministic for identical input', () => {
    expect(paint(strokes).equals(paint(strokes))).toBe(true);
  });

  it('actually draws something', () => {
    expect(paint(strokes).equals(paint([]))).toBe(false);
  });

  it('skips undone strokes', () => {
    expect(paint(strokes, new Set(['a', 'b', 'c'])).equals(paint([]))).toBe(true);
  });

  it('erases through earlier strokes', () => {
    const withEraser = paint(strokes);
    const withoutEraser = paint(strokes.filter((s) => s.tool !== 'eraser'));
    expect(withEraser.equals(withoutEraser)).toBe(false);
  });

  it('honours the crop offset', () => {
    const canvas = createCanvas(200, 160);
    const ctx = canvas.getContext('2d');
    renderStrokes(ctx as never, strokes, { offsetX: 1000, offsetY: 1000 });
    const data = ctx.getImageData(0, 0, 200, 160).data;
    expect(data.some((v) => v !== 0)).toBe(false);
  });
});
