import { describe, expect, it } from 'vitest';
import { clampRectInside, expandRect, intersectRect, rectContains, strokeBBox, unionRects } from '../src/index.js';

describe('geometry', () => {
  it('expands and intersects', () => {
    expect(expandRect({ x: 10, y: 10, width: 5, height: 5 }, 5)).toEqual({ x: 5, y: 5, width: 15, height: 15 });
    expect(intersectRect({ x: 0, y: 0, width: 10, height: 10 }, { x: 5, y: 5, width: 10, height: 10 })).toEqual({ x: 5, y: 5, width: 5, height: 5 });
    expect(intersectRect({ x: 0, y: 0, width: 10, height: 10 }, { x: 20, y: 0, width: 5, height: 5 })).toBeNull();
  });

  it('clamps a rect inside bounds without shrinking it', () => {
    expect(clampRectInside({ x: -100, y: 4000, width: 1024, height: 1024 }, 4096, 4096)).toEqual({ x: 0, y: 3072, width: 1024, height: 1024 });
  });

  it('clips the size when the rect is larger than the bounds', () => {
    expect(clampRectInside({ x: -10, y: -10, width: 5000, height: 5000 }, 4096, 4096)).toEqual({ x: 0, y: 0, width: 4096, height: 4096 });
  });

  it('pads stroke bboxes by half the brush width', () => {
    const b = strokeBBox([{ x: 100, y: 100 }, { x: 120, y: 140 }], 20);
    expect(b).toEqual({ x: 89, y: 89, width: 42, height: 62 });
  });

  it('unions and contains', () => {
    const u = unionRects([{ x: 0, y: 0, width: 10, height: 10 }, { x: 20, y: 5, width: 10, height: 10 }])!;
    expect(u).toEqual({ x: 0, y: 0, width: 30, height: 15 });
    expect(rectContains(u, { x: 1, y: 1, width: 2, height: 2 })).toBe(true);
    expect(rectContains(u, { x: 1, y: 1, width: 100, height: 2 })).toBe(false);
    expect(unionRects([])).toBeNull();
  });
});
