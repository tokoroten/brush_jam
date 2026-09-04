import { describe, expect, it } from 'vitest';
import { applyRect, chooseCrop } from '../src/index.js';

describe('chooseCrop', () => {
  it('centers the window on the dirty region', () => {
    expect(chooseCrop({ x: 2000, y: 2000, width: 48, height: 48 }, 1024, 4096)).toEqual({ x: 1512, y: 1512, width: 1024, height: 1024 });
  });

  it('clamps against the canvas edges without shrinking', () => {
    expect(chooseCrop({ x: 0, y: 0, width: 10, height: 10 }, 1024, 4096)).toEqual({ x: 0, y: 0, width: 1024, height: 1024 });
    expect(chooseCrop({ x: 4090, y: 4090, width: 6, height: 6 }, 1024, 4096)).toEqual({ x: 3072, y: 3072, width: 1024, height: 1024 });
  });

  it('clips the window to a canvas smaller than the window', () => {
    expect(chooseCrop({ x: 100, y: 100, width: 10, height: 10 }, 1024, 512)).toEqual({ x: 0, y: 0, width: 512, height: 512 });
  });

  it('applyRect is the centered sub-rect of the crop', () => {
    expect(applyRect({ x: 1512, y: 1512, width: 1024, height: 1024 }, 768)).toEqual({ x: 1640, y: 1640, width: 768, height: 768 });
    expect(applyRect({ x: 0, y: 0, width: 512, height: 512 }, 768)).toEqual({ x: 0, y: 0, width: 512, height: 512 });
  });
});

describe('chooseCrop squareness', () => {
  it('always returns an exact square with integer coordinates', () => {
    for (const region of [
      { x: 1000.5, y: 1487.25, width: 25, height: 25 },
      { x: 0.3, y: 4095.7, width: 3, height: 1 },
      { x: 2011, y: 2011, width: 3, height: 900 },
    ]) {
      const c = chooseCrop(region, 1024, 4096);
      expect(c.width).toBe(1024);
      expect(c.height).toBe(1024);
      expect(Number.isInteger(c.x) && Number.isInteger(c.y)).toBe(true);
      expect(c.x).toBeGreaterThanOrEqual(0);
      expect(c.x + c.width).toBeLessThanOrEqual(4096);
      expect(c.y + c.height).toBeLessThanOrEqual(4096);
    }
  });
});
