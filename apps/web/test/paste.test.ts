import { describe, expect, it } from 'vitest';
import { downscaleSize, fitScale, pastePlacement } from '../src/paste.js';

describe('paste helpers', () => {
  it('never upscales small images', () => {
    expect(fitScale(800, 600)).toBe(1);
    expect(downscaleSize(800, 600)).toEqual({ width: 800, height: 600 });
  });

  it('fits huge images to the long side', () => {
    expect(downscaleSize(6000, 3000)).toEqual({ width: 2048, height: 1024 });
    expect(downscaleSize(1000, 8000)).toEqual({ width: 256, height: 2048 });
  });

  it('honours a custom maximum', () => {
    expect(downscaleSize(1000, 500, 100)).toEqual({ width: 100, height: 50 });
  });

  it('never produces a zero dimension', () => {
    expect(downscaleSize(4000, 1, 512)).toEqual({ width: 512, height: 1 });
  });

  it('centers a pasted image on the viewport center', () => {
    expect(pastePlacement({ x: 2048, y: 2048 }, { width: 400, height: 200 })).toEqual({ x: 1848, y: 1948 });
  });
});
