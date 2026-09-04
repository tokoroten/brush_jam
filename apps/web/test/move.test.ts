import { describe, expect, it } from 'vitest';
import type { Layer } from '@brushjam/shared';
import { layerActions } from '../src/LayerPanel.js';
import {
  drawLayerRect,
  layerOrigin,
  layerPoint,
  layerRect,
  movePatch,
  movedPosition,
  pickMovableLayer,
  pickReferenceLayer,
  scaledBy,
  MAX_LAYER_SCALE,
  MIN_LAYER_SCALE,
} from '../src/move.js';

const ref = (id: string, extra: Partial<Layer> = {}): Layer => ({
  id,
  name: id,
  kind: 'reference',
  visible: true,
  locked: false,
  opacity: 1,
  order: 0,
  includeInAI: false,
  imageId: `img-${id}`,
  x: 0,
  y: 0,
  scale: 1,
  ...extra,
});

const draw = (id: string, order = 5): Layer => ({
  id,
  name: id,
  kind: 'draw',
  visible: true,
  locked: false,
  opacity: 1,
  order,
  includeInAI: true,
});

const sizes: Record<string, { width: number; height: number }> = {
  'img-a': { width: 100, height: 50 },
  'img-b': { width: 200, height: 200 },
};
const sizeOf = (id: string): { width: number; height: number } | undefined => sizes[id];

describe('layerRect', () => {
  it('applies the layer scale and position', () => {
    expect(layerRect(ref('a', { x: 10, y: 20, scale: 2 }), sizeOf)).toEqual({ x: 10, y: 20, width: 200, height: 100 });
  });

  it('is null for a draw layer or an image we have not loaded', () => {
    expect(layerRect(draw('d'), sizeOf)).toBeNull();
    expect(layerRect(ref('z', { imageId: 'img-missing' }), sizeOf)).toBeNull();
  });
});

describe('pickReferenceLayer', () => {
  it('picks the reference under the pointer', () => {
    const layers = [draw('d'), ref('a', { x: 0, y: 0 })];
    expect(pickReferenceLayer(layers, sizeOf, { x: 50, y: 25 })?.id).toBe('a');
    expect(pickReferenceLayer(layers, sizeOf, { x: 500, y: 500 })).toBeNull();
  });

  it('prefers the frontmost of two overlapping references', () => {
    const layers = [ref('a', { order: 1 }), ref('b', { order: 3, x: 0, y: 0 })];
    expect(pickReferenceLayer(layers, sizeOf, { x: 10, y: 10 })?.id).toBe('b');
  });

  it('never picks a locked or hidden layer', () => {
    expect(pickReferenceLayer([ref('a', { locked: true })], sizeOf, { x: 10, y: 10 })).toBeNull();
    expect(pickReferenceLayer([ref('a', { visible: false })], sizeOf, { x: 10, y: 10 })).toBeNull();
  });

  it('falls back to the selected reference when the pointer is elsewhere', () => {
    const layers = [ref('a'), ref('b', { x: 1000, y: 1000 })];
    expect(pickReferenceLayer(layers, sizeOf, { x: 5000, y: 5000 }, 'b')?.id).toBe('b');
    // ...but not to a locked one
    expect(pickReferenceLayer([ref('b', { locked: true })], sizeOf, { x: 5000, y: 5000 }, 'b')).toBeNull();
  });
});

describe('movedPosition', () => {
  it('converts a screen delta to world units through the zoom', () => {
    expect(movedPosition({ x: 100, y: 100 }, 40, -20, 0.5)).toEqual({ x: 180, y: 60 });
    expect(movedPosition({ x: 100, y: 100 }, 40, -20, 2)).toEqual({ x: 120, y: 90 });
  });

  it('treats a nonsense zoom as 1', () => {
    expect(movedPosition({ x: 0, y: 0 }, 10, 10, 0)).toEqual({ x: 10, y: 10 });
  });
});

describe('scaledBy', () => {
  it('multiplies and clamps', () => {
    expect(scaledBy(1, 2)).toBe(2);
    expect(scaledBy(undefined, 0.5)).toBe(0.5);
    expect(scaledBy(1, 100)).toBe(MAX_LAYER_SCALE);
    expect(scaledBy(1, 0.0001)).toBe(MIN_LAYER_SCALE);
  });
});

/** Destructive layer actions must ask first. */
describe('layer actions confirmation', () => {
  const layer = ref('a', { name: 'Sky' });

  function fakeClient(): { sent: unknown[]; send: (msg: unknown) => void } {
    const sent: unknown[] = [];
    return { sent, send: (msg) => sent.push(msg) };
  }

  it('sends nothing when the user cancels', () => {
    const client = fakeClient();
    const asked: string[] = [];
    const actions = layerActions(client as never, (m) => {
      asked.push(m);
      return false;
    });
    actions.clear(layer);
    actions.remove(layer);
    expect(client.sent).toHaveLength(0);
    expect(asked[0]).toContain("Clear all strokes on 'Sky'");
    expect(asked[1]).toContain("Delete layer 'Sky'");
  });

  it('sends the message when the user confirms', () => {
    const client = fakeClient();
    const actions = layerActions(client as never, () => true);
    actions.clear(layer);
    actions.remove(layer);
    expect(client.sent).toEqual([
      { t: 'clear_layer', layerId: 'a' },
      { t: 'layer_delete', id: 'a' },
    ]);
  });
});

/** Move also applies to draw layers, via a render-time offset. */
describe('draw layer moves', () => {
  const stroke = (id: string, layerId: string, box: { x: number; y: number; width: number; height: number }): never =>
    ({
      id,
      userId: 'u',
      layerId,
      tool: 'pen',
      color: '#000',
      width: 4,
      points: [{ x: box.x, y: box.y }],
      revision: 1,
      bbox: box,
    }) as never;

  const drawLayer = (id: string, extra: Partial<Layer> = {}): Layer => ({ ...draw(id, 1), ...extra });
  const none = new Set<string>();

  it('boxes a draw layer by its strokes plus the offset', () => {
    const layer = drawLayer('d1', { offsetX: 100, offsetY: 50 });
    const strokes = [stroke('s1', 'd1', { x: 10, y: 10, width: 40, height: 20 })];
    expect(drawLayerRect(layer, strokes, none)).toEqual({ x: 110, y: 60, width: 40, height: 20 });
    // undone strokes do not count, and an empty layer has no box
    expect(drawLayerRect(layer, strokes, new Set(['s1']))).toBeNull();
    expect(drawLayerRect(layer, [], none)).toBeNull();
  });

  it('picks the draw layer under the pointer', () => {
    const layer = drawLayer('d1', { offsetX: 100, offsetY: 0 });
    const strokes = [stroke('s1', 'd1', { x: 0, y: 0, width: 50, height: 50 })];
    const ctx = { sizeOf, strokes, undone: none };
    expect(pickMovableLayer([layer], ctx, { x: 120, y: 20 })?.id).toBe('d1');
    // the *old* position is no longer where the content is
    expect(pickMovableLayer([layer], ctx, { x: 20, y: 20 })).toBeNull();
  });

  it('prefers the frontmost layer and never a locked one', () => {
    const back = drawLayer('back', { order: 1 });
    const front = ref('front', { order: 9, imageId: 'img-a' });
    const strokes = [stroke('s1', 'back', { x: 0, y: 0, width: 80, height: 40 })];
    const ctx = { sizeOf, strokes, undone: none };
    expect(pickMovableLayer([back, front], ctx, { x: 20, y: 20 })?.id).toBe('front');
    expect(pickMovableLayer([{ ...back, locked: true }], ctx, { x: 20, y: 20 })).toBeNull();
  });

  it('sends offsetX/offsetY for a draw layer and x/y for a reference', () => {
    expect(movePatch(drawLayer('d1'), { x: 5, y: 6 })).toEqual({ offsetX: 5, offsetY: 6 });
    expect(movePatch(ref('r1'), { x: 5, y: 6 })).toEqual({ x: 5, y: 6 });
    expect(layerOrigin(drawLayer('d1', { offsetX: 3, offsetY: 4 }))).toEqual({ x: 3, y: 4 });
    expect(layerOrigin(ref('r1', { x: 7, y: 8 }))).toEqual({ x: 7, y: 8 });
  });

  it('records stroke points in layer space so the line lands under the pointer', () => {
    const layer = drawLayer('d1', { offsetX: 120, offsetY: -30 });
    expect(layerPoint({ x: 200, y: 100 }, layer)).toEqual({ x: 80, y: 130 });
    expect(layerPoint({ x: 200, y: 100 }, undefined)).toEqual({ x: 200, y: 100 });
  });
});

/** Wheel scaling shares the slider's range and keeps full precision. */
describe('wheel scaling precision', () => {
  it('accumulates small steps instead of rounding them away', () => {
    let scale = 1;
    for (let i = 0; i < 10; i++) scale = scaledBy(scale, Math.exp(-1 * 0.0015));
    // 10 tiny steps must move the value, not vanish to 1.00 each time
    expect(scale).toBeLessThan(1);
    expect(scale).toBeGreaterThan(0.98);
    expect(scale).not.toBe(Math.round(scale * 100) / 100);
  });

  it('clamps to the same range the panel slider offers', () => {
    expect(scaledBy(MAX_LAYER_SCALE, 2)).toBe(MAX_LAYER_SCALE);
    expect(scaledBy(MIN_LAYER_SCALE, 0.5)).toBe(MIN_LAYER_SCALE);
  });
});
