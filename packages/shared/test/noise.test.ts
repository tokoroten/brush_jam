import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it } from 'vitest';
import { fnv1a, noiseRGB, renderStrokes, strokeBounds, type RenderableStroke } from '../src/index.js';

const stroke = (id: string): RenderableStroke => ({
  id,
  tool: 'noise',
  color: '#000000',
  width: 24,
  points: [
    { x: 100, y: 100 },
    { x: 180, y: 140 },
    { x: 240, y: 120 },
  ],
});

const factory = (w: number, h: number): never => createCanvas(w, h) as never;

/** Render a stroke on a `size` canvas whose origin is (offsetX, offsetY). */
function render(s: RenderableStroke, size: number, offsetX = 0, offsetY = 0): Buffer {
  const canvas = createCanvas(size, size);
  renderStrokes(canvas.getContext('2d') as never, [s], { offsetX, offsetY, createCanvas: factory });
  return canvas.toBuffer('image/png');
}

function pixels(s: RenderableStroke, size: number, offsetX = 0, offsetY = 0): Uint8ClampedArray {
  const canvas = createCanvas(size, size);
  renderStrokes(canvas.getContext('2d') as never, [s], { offsetX, offsetY, createCanvas: factory });
  return canvas.getContext('2d').getImageData(0, 0, size, size).data as unknown as Uint8ClampedArray;
}

describe('noise hashing', () => {
  it('is a pure function of id and world position', () => {
    expect(fnv1a('abc')).toBe(fnv1a('abc'));
    expect(fnv1a('abc')).not.toBe(fnv1a('abd'));
    expect(noiseRGB(1, 10, 20)).toEqual(noiseRGB(1, 10, 20));
    expect(noiseRGB(1, 10, 20)).not.toEqual(noiseRGB(1, 11, 20));
    expect(noiseRGB(1, 10, 20)).not.toEqual(noiseRGB(2, 10, 20));
  });

  it('spreads values over the whole byte range', () => {
    const buckets = new Array(8).fill(0);
    for (let x = 0; x < 64; x++) {
      for (let y = 0; y < 64; y++) buckets[noiseRGB(99, x, y)[0] >> 5]! += 1;
    }
    const expected = (64 * 64) / 8;
    for (const count of buckets) expect(Math.abs(count - expected) / expected).toBeLessThan(0.25);
  });
});

describe('noise stroke rendering', () => {
  it('is byte-identical across renders', () => {
    expect(render(stroke('s1'), 320)).toEqual(render(stroke('s1'), 320));
  });

  it('differs between stroke ids', () => {
    expect(render(stroke('s1'), 320)).not.toEqual(render(stroke('s2'), 320));
  });

  it('produces the same world pixels through a translated context', () => {
    const size = 320;
    const full = pixels(stroke('s1'), size);
    // the "server crop": a 128px window starting at (96, 96) of the same world
    const crop = 128;
    const cropped = pixels(stroke('s1'), crop, 96, 96);
    let compared = 0;
    for (let y = 0; y < crop; y++) {
      for (let x = 0; x < crop; x++) {
        const a = ((y + 96) * size + (x + 96)) * 4;
        const b = (y * crop + x) * 4;
        // Compare fully covered pixels: the noise value is what must match.
        // (Edge antialiasing can differ by a single alpha step between a
        // full-size and a clipped rasterisation, which is not the claim here.)
        if (full[a + 3] !== 255 || cropped[b + 3] !== 255) continue;
        expect([cropped[b], cropped[b + 1], cropped[b + 2]]).toEqual([full[a], full[a + 1], full[a + 2]]);
        compared += 1;
      }
    }
    expect(compared).toBeGreaterThan(500);
  });

  it('keeps the shape antialiasing in the alpha channel', () => {
    const data = pixels(stroke('s1'), 320);
    let opaque = 0;
    let partial = 0;
    let coloured = 0;
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3]!;
      if (a === 255) opaque += 1;
      else if (a > 0) partial += 1;
      if (a > 0 && (data[i] !== data[i + 1] || data[i + 1] !== data[i + 2])) coloured += 1;
    }
    expect(opaque).toBeGreaterThan(100);
    expect(partial).toBeGreaterThan(0);
    // RGB really is noise, not a flat colour
    expect(coloured).toBeGreaterThan(100);
  });

  it('draws nothing outside the stroke bounds', () => {
    const s = stroke('s1');
    const bounds = strokeBounds(s);
    const data = pixels(s, 320);
    for (let y = 0; y < 320; y++) {
      for (let x = 0; x < 320; x++) {
        if (x >= bounds.x && x <= bounds.x + bounds.width && y >= bounds.y && y <= bounds.y + bounds.height) continue;
        expect(data[(y * 320 + x) * 4 + 3]).toBe(0);
      }
    }
  });

  it('falls back to a plain stroke without a canvas factory', () => {
    const canvas = createCanvas(320, 320);
    renderStrokes(canvas.getContext('2d') as never, [stroke('s1')], {});
    const data = canvas.getContext('2d').getImageData(0, 0, 320, 320).data;
    let painted = 0;
    for (let i = 0; i < data.length; i += 4) if (data[i + 3]! > 0) painted += 1;
    expect(painted).toBeGreaterThan(0);
  });

  it('skips an undone noise stroke', () => {
    const canvas = createCanvas(320, 320);
    renderStrokes(canvas.getContext('2d') as never, [stroke('s1')], {
      undone: new Set(['s1']),
      createCanvas: factory,
    });
    const data = canvas.getContext('2d').getImageData(0, 0, 320, 320).data;
    for (let i = 0; i < data.length; i += 4) expect(data[i + 3]).toBe(0);
  });
});
