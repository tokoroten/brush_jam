import { describe, expect, it } from 'vitest';
import { fitCamera, panBy, screenToWorld, worldToScreen, zoomAt } from '../src/index.js';

const cam = { centerX: 2048, centerY: 2048, zoom: 1 };

describe('camera', () => {
  it('round-trips world and screen coordinates', () => {
    const s = worldToScreen(cam, 800, 600, 2148, 1948);
    expect(s).toEqual({ x: 500, y: 200 });
    expect(screenToWorld(cam, 800, 600, 500, 200)).toEqual({ x: 2148, y: 1948 });
  });

  it('keeps the point under the cursor fixed while zooming', () => {
    const next = zoomAt(cam, 800, 600, 100, 50, 2);
    const before = screenToWorld(cam, 800, 600, 100, 50);
    const after = screenToWorld(next, 800, 600, 100, 50);
    expect(next.zoom).toBe(2);
    expect(after.x).toBeCloseTo(before.x, 6);
    expect(after.y).toBeCloseTo(before.y, 6);
  });

  it('clamps zoom', () => {
    expect(zoomAt(cam, 800, 600, 0, 0, 1000).zoom).toBe(8);
    expect(zoomAt(cam, 800, 600, 0, 0, 0.0001).zoom).toBe(0.05);
  });

  it('pans in world units', () => {
    expect(panBy({ ...cam, zoom: 2 }, 100, -50)).toEqual({ centerX: 1998, centerY: 2073, zoom: 2 });
  });

  it('fits the whole canvas', () => {
    const f = fitCamera(800, 600, 4096);
    expect(f.centerX).toBe(2048);
    expect(f.zoom).toBeCloseTo(600 / 4096, 10);
  });
});
