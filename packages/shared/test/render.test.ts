import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it } from 'vitest';
import { MIN_STROKE_ALPHA, alphaOf, renderStrokes, type RenderableStroke } from '../src/index.js';

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

/**
 * Stroke opacity. A stroke is one mark, so its alpha applies once: where it
 * crosses itself it must not be darker, and the client and the server must
 * agree pixel for pixel or the AI would be fed a different picture than the
 * one on screen.
 */
describe('stroke alpha', () => {
  const SIZE = 120;
  /** A stroke that doubles back through its own middle. */
  const crossing = (alpha?: number): RenderableStroke => ({
    id: 'x',
    tool: 'pen',
    color: '#000000',
    width: 16,
    ...(alpha === undefined ? {} : { alpha }),
    points: [
      { x: 20, y: 20 },
      { x: 100, y: 100 },
      { x: 100, y: 20 },
      { x: 20, y: 100 },
    ],
  });

  function pixels(list: RenderableStroke[], options: Record<string, unknown> = {}): Uint8ClampedArray {
    const canvas = createCanvas(SIZE, SIZE);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, SIZE, SIZE);
    renderStrokes(ctx as never, list, {
      createCanvas: (w, h) => createCanvas(w, h) as never,
      bounds: { width: SIZE, height: SIZE },
      ...options,
    });
    return ctx.getImageData(0, 0, SIZE, SIZE).data as unknown as Uint8ClampedArray;
  }

  const at = (data: Uint8ClampedArray, x: number, y: number): number => data[(y * SIZE + x) * 4]!;

  it('alphaOf defaults to opaque and clamps into range', () => {
    expect(alphaOf({ tool: 'pen' })).toBe(1);
    expect(alphaOf({ tool: 'pen', alpha: 0.5 })).toBe(0.5);
    expect(alphaOf({ tool: 'pen', alpha: 0 })).toBe(MIN_STROKE_ALPHA);
    expect(alphaOf({ tool: 'pen', alpha: 4 })).toBe(1);
    expect(alphaOf({ tool: 'pen', alpha: Number.NaN })).toBe(1);
  });

  it('ignores alpha on the eraser, which always removes fully', () => {
    expect(alphaOf({ tool: 'eraser', alpha: 0.2 })).toBe(1);
  });

  it('draws a half-alpha stroke lighter than an opaque one', () => {
    const opaque = at(pixels([crossing()]), 60, 60);
    const half = at(pixels([crossing(0.5)]), 60, 60);
    expect(opaque).toBeLessThan(20);
    expect(half).toBeGreaterThan(100);
    expect(half).toBeLessThan(160);
  });

  it('has uniform coverage where the stroke crosses itself', () => {
    const data = pixels([crossing(0.5)]);
    // (60,60) is the crossing point; (40,40) and (80,40) are single-pass
    // stretches of the same stroke.
    const cross = at(data, 60, 60);
    const single = at(data, 40, 40);
    const other = at(data, 80, 40);
    expect(Math.abs(cross - single)).toBeLessThanOrEqual(2);
    expect(Math.abs(single - other)).toBeLessThanOrEqual(2);
  });

  it('would have been darker at the crossing without the fix', () => {
    // Sanity check that the test point really is an overlap: two SEPARATE
    // half-alpha strokes crossing there do accumulate, as they should.
    const a: RenderableStroke = { id: 'a', tool: 'pen', color: '#000000', width: 16, alpha: 0.5, points: [{ x: 20, y: 20 }, { x: 100, y: 100 }] };
    const b: RenderableStroke = { id: 'b', tool: 'pen', color: '#000000', width: 16, alpha: 0.5, points: [{ x: 100, y: 20 }, { x: 20, y: 100 }] };
    const data = pixels([a, b]);
    expect(at(data, 60, 60)).toBeLessThan(at(data, 40, 40) - 20);
  });

  it('keeps consecutive segments of one stroke from darkening at the joins', () => {
    const data = pixels([crossing(0.5)]);
    // the corner at (100,100) is where two segments meet at a sharp angle
    expect(Math.abs(at(data, 96, 96) - at(data, 40, 40))).toBeLessThanOrEqual(3);
  });

  it('renders identically for a client-style and a server-style pass', () => {
    // The server renders a crop: the same stroke, shifted by the crop origin,
    // into a canvas of the crop's size. The overlapping region must match the
    // client's full-canvas render exactly.
    const stroke = crossing(0.4);
    const full = pixels([stroke]);

    const cropSize = 80;
    const canvas = createCanvas(cropSize, cropSize);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, cropSize, cropSize);
    renderStrokes(ctx as never, [stroke], {
      offsetX: 20,
      offsetY: 20,
      createCanvas: (w, h) => createCanvas(w, h) as never,
      bounds: { width: cropSize, height: cropSize },
    });
    const crop = ctx.getImageData(0, 0, cropSize, cropSize).data;

    let diff = 0;
    for (let y = 0; y < cropSize; y++) {
      for (let x = 0; x < cropSize; x++) {
        const a = full[((y + 20) * SIZE + (x + 20)) * 4]!;
        const b = crop[(y * cropSize + x) * 4]!;
        if (Math.abs(a - b) > 1) diff += 1;
      }
    }
    expect(diff).toBe(0);
  });

  it('is deterministic across repeated renders at the same alpha', () => {
    const once = pixels([crossing(0.35)]);
    const twice = pixels([crossing(0.35)]);
    expect(Buffer.from(once).equals(Buffer.from(twice))).toBe(true);
  });

  it('leaves opaque strokes byte-for-byte as before', () => {
    const withField = pixels([{ ...crossing(1) }]);
    const without = pixels([crossing()]);
    expect(Buffer.from(withField).equals(Buffer.from(without))).toBe(true);
  });

  it('applies alpha to a noise stroke too', () => {
    const noise = (alpha?: number): RenderableStroke => ({
      id: 'n',
      tool: 'noise',
      color: '#000000',
      width: 30,
      ...(alpha === undefined ? {} : { alpha }),
      points: [{ x: 30, y: 60 }, { x: 90, y: 60 }],
    });
    const opaque = pixels([noise()]);
    const faint = pixels([noise(0.2)]);
    // A faint noise stroke sits much closer to the white background.
    let opaqueDist = 0;
    let faintDist = 0;
    for (let x = 40; x < 80; x++) {
      opaqueDist += Math.abs(255 - at(opaque, x, 60));
      faintDist += Math.abs(255 - at(faint, x, 60));
    }
    expect(faintDist).toBeLessThan(opaqueDist / 2);
  });
});
