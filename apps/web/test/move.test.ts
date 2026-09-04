import { describe, expect, it } from 'vitest';
import type { Layer } from '@brushjam/shared';
import { layerActions } from '../src/LayerPanel.js';
import { layerRect, movedPosition, pickReferenceLayer, scaledBy, MAX_LAYER_SCALE, MIN_LAYER_SCALE } from '../src/move.js';

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
