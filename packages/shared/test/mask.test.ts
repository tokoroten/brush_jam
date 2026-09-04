import { describe, expect, it } from 'vitest';
import { planMask } from '../src/index.js';

const crop = { x: 1000, y: 1000, width: 1024, height: 1024 };
const apply = { x: 1128, y: 1128, width: 768, height: 768 };

describe('planMask', () => {
  it('produces crop-local, dilated shapes', () => {
    const plan = planMask([{ x: 1500, y: 1500, width: 20, height: 20 }], crop, apply, 48, 32);
    expect(plan.shapes).toEqual([{ x: 452, y: 452, width: 116, height: 116 }]);
    expect(plan.apply).toEqual({ x: 128, y: 128, width: 768, height: 768 });
    expect(plan.feather).toBe(32);
    expect(plan.empty).toBe(false);
  });

  it('clips shapes to the crop and drops those outside it', () => {
    const plan = planMask(
      [{ x: 990, y: 990, width: 20, height: 20 }, { x: 3000, y: 3000, width: 20, height: 20 }],
      crop, apply, 48, 32,
    );
    expect(plan.shapes).toEqual([{ x: 0, y: 0, width: 58, height: 58 }]);
  });

  it('reports empty when nothing intersects the crop', () => {
    expect(planMask([{ x: 3000, y: 3000, width: 4, height: 4 }], crop, apply).empty).toBe(true);
  });

  it('keeps every shape inside the crop bounds', () => {
    const plan = planMask([{ x: 1000, y: 1000, width: 1024, height: 1024 }], crop, apply);
    for (const s of plan.shapes) {
      expect(s.x).toBeGreaterThanOrEqual(0);
      expect(s.y).toBeGreaterThanOrEqual(0);
      expect(s.x + s.width).toBeLessThanOrEqual(crop.width);
      expect(s.y + s.height).toBeLessThanOrEqual(crop.height);
    }
  });
});
