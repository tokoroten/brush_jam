import { describe, expect, it } from 'vitest';
import { validateClientMessage } from '../src/validate.js';
import { applyClientMessage, addMember, createRoom } from '../src/room.js';

/** Finding 1: a malformed frame must be rejected, never crash the process. */
describe('validateClientMessage', () => {
  const rejects = [
    null,
    'nope',
    42,
    [],
    {},
    { t: 42 },
    { t: 'nope' },
    { t: 'layer_create' },
    { t: 'layer_create', layer: null },
    { t: 'layer_create', layer: 'draw' },
    { t: 'layer_create', layer: {} },
    { t: 'layer_create', layer: { kind: 'folder' } },
    { t: 'layer_create', layer: { kind: 'draw', opacity: 'half' } },
    { t: 'layer_update' },
    { t: 'layer_update', id: 'a' },
    { t: 'layer_update', id: 'a', patch: null },
    { t: 'layer_update', id: 'a', patch: { visible: 'yes' } },
    { t: 'layer_delete' },
    { t: 'layer_reorder' },
    { t: 'layer_reorder', ids: [] },
    { t: 'layer_reorder', ids: ['a', 'a'] },
    { t: 'layer_reorder', ids: [1, 2] },
    { t: 'clear_layer' },
    { t: 'cursor' },
    { t: 'cursor', x: 'a', y: 1 },
    { t: 'cursor', x: Number.NaN, y: 1 },
    { t: 'set_prompt' },
    { t: 'set_prompt', prompt: 12 },
    { t: 'stroke_start' },
    { t: 'stroke_start', stroke: null },
    { t: 'stroke_start', stroke: { id: '', layerId: 'l', tool: 'pen', color: '#000000', width: 1, points: [] } },
    { t: 'stroke_start', stroke: { id: 'a', layerId: 'l', tool: 'brush', color: '#000000', width: 1, points: [] } },
    { t: 'stroke_start', stroke: { id: 'a', layerId: 'l', tool: 'pen', color: '#000000', width: 1, points: [{ x: 1 }] } },
    { t: 'stroke_start', stroke: { id: 'a', layerId: 'l', tool: 'pen', color: '#000000', width: Infinity, points: [] } },
    { t: 'stroke_chunk', strokeId: 'a' },
    { t: 'stroke_chunk', points: [] },
    { t: 'stroke_end', strokeId: 'a', points: 'lots' },
  ];

  it.each(rejects.map((raw, i) => [i, raw] as const))('rejects malformed message #%i', (_i, raw) => {
    const result = validateClientMessage(raw);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toBeTruthy();
  });

  it('accepts well-formed messages', () => {
    const accepted = [
      { t: 'undo' },
      { t: 'cursor', x: 1.5, y: -2 },
      { t: 'clear_layer', layerId: 'l1' },
      { t: 'set_prompt', prompt: 'a town' },
      { t: 'layer_create', layer: { kind: 'draw' } },
      { t: 'layer_create', layer: { kind: 'reference', imageId: 'img1', x: 1, y: 2, scale: 0.5 } },
      { t: 'layer_update', id: 'l1', patch: { name: 'Sky', opacity: 0.5, visible: false } },
      { t: 'layer_delete', id: 'l1' },
      { t: 'layer_reorder', ids: ['a', 'b'] },
      { t: 'stroke_start', stroke: { id: 's', layerId: 'l', tool: 'eraser', color: '#ffffff', width: 4, points: [{ x: 1, y: 2, p: 0.5 }] } },
      { t: 'stroke_chunk', strokeId: 's', points: [] },
      { t: 'stroke_end', strokeId: 's', points: [{ x: 3, y: 4 }] },
    ];
    for (const raw of accepted) expect(validateClientMessage(raw).ok).toBe(true);
  });

  it('strips unknown layer fields instead of trusting them', () => {
    const result = validateClientMessage({ t: 'layer_update', id: 'l1', patch: { kind: 'reference', order: 99, name: 'ok' } });
    expect(result.ok).toBe(true);
    if (result.ok && result.msg.t === 'layer_update') {
      expect(result.msg.patch).toEqual({ name: 'ok' });
    }
  });

  it('rejects an oversized point batch', () => {
    const points = Array.from({ length: 20_001 }, () => ({ x: 1, y: 1 }));
    expect(validateClientMessage({ t: 'stroke_chunk', strokeId: 's', points }).ok).toBe(false);
  });

  it('the reducer survives every rejected payload if one slipped through', () => {
    // Belt and braces: the reducer is only ever called with validated input, but
    // the shape that crashed before (layer_create without a layer) must not throw.
    const state = createRoom('guard');
    const userId = addMember(state, 'A').userId;
    expect(() => applyClientMessage(state, userId, { t: 'layer_create', layer: { kind: 'draw' } })).not.toThrow();
  });
});

describe('set_ai_settings', () => {
  it('accepts a denoise inside the range', () => {
    expect(validateClientMessage({ t: 'set_ai_settings', denoise: 0.75 })).toEqual({ ok: true, msg: { t: 'set_ai_settings', denoise: 0.75 } });
  });

  it('accepts a negative prompt on its own', () => {
    const out = validateClientMessage({ t: 'set_ai_settings', negativePrompt: 'blurry' });
    expect(out).toEqual({ ok: true, msg: { t: 'set_ai_settings', negativePrompt: 'blurry' } });
  });

  it('accepts an empty negative prompt (means "use the default")', () => {
    expect(validateClientMessage({ t: 'set_ai_settings', negativePrompt: '' }).ok).toBe(true);
  });

  it.each([
    { t: 'set_ai_settings' },
    { t: 'set_ai_settings', denoise: 0.1 },
    { t: 'set_ai_settings', denoise: 1.5 },
    { t: 'set_ai_settings', denoise: Number.NaN },
    { t: 'set_ai_settings', denoise: Number.POSITIVE_INFINITY },
    { t: 'set_ai_settings', denoise: '0.5' },
    { t: 'set_ai_settings', negativePrompt: 42 },
    { t: 'set_ai_settings', negativePrompt: 'x'.repeat(1001) },
  ])('rejects %j', (payload) => {
    expect(validateClientMessage(payload).ok).toBe(false);
  });
});

describe('set_ai_settings aiResolution', () => {
  it('accepts a size on the 64 grid', () => {
    expect(validateClientMessage({ t: 'set_ai_settings', aiResolution: 768 })).toEqual({
      ok: true,
      msg: { t: 'set_ai_settings', aiResolution: 768 },
    });
  });

  it.each([
    { t: 'set_ai_settings', aiResolution: 256 },
    { t: 'set_ai_settings', aiResolution: 4096 },
    { t: 'set_ai_settings', aiResolution: 700 },
    { t: 'set_ai_settings', aiResolution: '768' },
    { t: 'set_ai_settings', aiResolution: Number.NaN },
  ])('rejects %j', (payload) => {
    expect(validateClientMessage(payload).ok).toBe(false);
  });
});

/**
 * The browser found this one: the reducer handled aiProfile but the validator
 * silently dropped it, so the segmented control did nothing over a real socket
 * while every unit test passed.
 */
describe('set_ai_settings aiProfile', () => {
  it('accepts both profiles', () => {
    for (const aiProfile of ['fast', 'quality']) {
      const out = validateClientMessage({ t: 'set_ai_settings', aiProfile });
      expect(out.ok).toBe(true);
      expect(out.ok && out.msg).toMatchObject({ t: 'set_ai_settings', aiProfile });
    }
  });

  it('rejects anything else', () => {
    const out = validateClientMessage({ t: 'set_ai_settings', aiProfile: 'turbo' });
    expect(out.ok).toBe(false);
  });

  it('rejects a non-string profile', () => {
    expect(validateClientMessage({ t: 'set_ai_settings', aiProfile: 7 }).ok).toBe(false);
  });

  it('still requires at least one field', () => {
    expect(validateClientMessage({ t: 'set_ai_settings' }).ok).toBe(false);
  });

  it('carries the profile alongside the other settings', () => {
    const out = validateClientMessage({ t: 'set_ai_settings', denoise: 0.8, aiResolution: 512, aiProfile: 'quality' });
    expect(out.ok && out.msg).toMatchObject({ denoise: 0.8, aiResolution: 512, aiProfile: 'quality' });
  });
});
