import { describe, expect, it } from 'vitest';
import { MAX_PASTE_SIZE } from '@brushjam/shared';
import { downscaleSize, fitScale, pasteLimit, pastePlacement } from '../src/paste.js';

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

/** Paste must fit the actual canvas, which is 1024 by default. */
describe('paste on a small canvas', () => {
  it('limits the long side to the canvas size', () => {
    expect(pasteLimit(1024)).toBe(1024);
    expect(pasteLimit(4096)).toBe(MAX_PASTE_SIZE);
    expect(pasteLimit(512)).toBe(512);
  });

  it('keeps the pasted image inside the canvas', () => {
    // centred normally...
    expect(pastePlacement({ x: 512, y: 512 }, { width: 200, height: 100 }, 1024)).toEqual({ x: 412, y: 462 });
    // ...and pulled back inside when the viewport centre is near an edge
    expect(pastePlacement({ x: 1000, y: 1000 }, { width: 400, height: 400 }, 1024)).toEqual({ x: 624, y: 624 });
    expect(pastePlacement({ x: 10, y: 10 }, { width: 400, height: 400 }, 1024)).toEqual({ x: 0, y: 0 });
  });

  it('leaves an image larger than the canvas at the origin', () => {
    expect(pastePlacement({ x: 512, y: 512 }, { width: 2000, height: 2000 }, 1024)).toEqual({ x: 0, y: 0 });
  });
});
