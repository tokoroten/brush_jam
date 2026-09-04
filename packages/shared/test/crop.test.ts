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
