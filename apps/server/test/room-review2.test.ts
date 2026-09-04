import { describe, expect, it } from 'vitest';
import { CANVAS_SIZE, MAX_DENOISE, MAX_NEGATIVE_PROMPT, MIN_DENOISE } from '@brushjam/shared';
import {
  captureRenderSnapshot,
  snapshot,
  addMember,
  applyClientMessage,
  createRoom,
  expirePendingStrokes,
  joinMember,
  qualifyStrokeId,
  removeMember,
  MAX_PENDING_PER_USER,
  MAX_STROKES_PER_ROOM,
  PENDING_STROKE_IDLE_MS,
  type RoomState,
} from '../src/room.js';

function room(): { state: RoomState; alice: string; bob: string; layerId: string } {
  const state = createRoom('review2');
  const alice = addMember(state, 'Alice').userId;
  const bob = addMember(state, 'Bob').userId;
  return { state, alice, bob, layerId: state.layers[0]!.id };
}

const begin = (state: RoomState, userId: string, layerId: string, id: string): ReturnType<typeof applyClientMessage> =>
  applyClientMessage(state, userId, {
    t: 'stroke_start',
    stroke: { id, layerId, tool: 'pen', color: '#000000', width: 8, points: [{ x: 10, y: 10 }] },
  });

/** Finding B8: renaming or locking a layer must never spend a generation. */
describe('metadata-only layer updates', () => {
  it('does not dirty anything when only the name changes', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 's');
    applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 's', points: [{ x: 90, y: 90 }] });

    const renamed = applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { name: 'Sky' } });
    expect(renamed.dirty).toHaveLength(0);
    expect(state.layers[0]!.name).toBe('Sky');
  });

  it('does not dirty anything when only the lock changes', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 's');
    applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 's', points: [{ x: 90, y: 90 }] });
    expect(applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { locked: true } }).dirty).toHaveLength(0);
  });

  it('does not dirty anything when a render property is set to the value it already has', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 's');
    applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 's', points: [{ x: 90, y: 90 }] });
    expect(applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { opacity: 1, visible: true } }).dirty).toHaveLength(0);
  });

  it('still dirties a real visual change', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 's');
    applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 's', points: [{ x: 90, y: 90 }] });
    expect(applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { opacity: 0.3 } }).dirty.length).toBeGreaterThan(0);
    expect(applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { visible: false } }).dirty.length).toBeGreaterThan(0);
  });
});

/** Finding 14 (completing): pending strokes expire on their own. */
describe('pending stroke expiry', () => {
  it('cancels a stroke that has had no activity for the idle window', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'quiet');
    const now = Date.now();

    expect(expirePendingStrokes(state, now + PENDING_STROKE_IDLE_MS - 1)).toHaveLength(0);
    const cancels = expirePendingStrokes(state, now + PENDING_STROKE_IDLE_MS + 1);
    expect(cancels).toHaveLength(1);
    expect(cancels[0]).toMatchObject({ t: 'stroke_cancel', strokeId: qualifyStrokeId(alice, 'quiet'), reason: 'stroke abandoned' });
    expect(state.pending.size).toBe(0);
  });

  it('also cancels a stroke that has run past the absolute time limit', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'endless');
    const p = state.pending.get(qualifyStrokeId(alice, 'endless'))!;
    const now = Date.now();
    p.lastActivityAt = now + 120_000; // still active...
    p.startedAt = now; // ...but started far too long ago
    expect(expirePendingStrokes(state, now + 120_000)).toHaveLength(1);
  });

  it('keeps a stroke alive while chunks keep arriving', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'busy');
    applyClientMessage(state, alice, { t: 'stroke_chunk', strokeId: 'busy', points: [{ x: 20, y: 20 }] });
    expect(expirePendingStrokes(state, Date.now() + PENDING_STROKE_IDLE_MS - 100)).toHaveLength(0);
    expect(state.pending.size).toBe(1);
  });

  it('leaves other users strokes alone', () => {
    const { state, alice, bob, layerId } = room();
    begin(state, alice, layerId, 'old');
    const now = Date.now();
    begin(state, bob, layerId, 'new');
    const busy = state.pending.get(qualifyStrokeId(bob, 'new'))!;
    busy.lastActivityAt = now + PENDING_STROKE_IDLE_MS;
    busy.startedAt = now + PENDING_STROKE_IDLE_MS;
    const cancels = expirePendingStrokes(state, now + PENDING_STROKE_IDLE_MS + 1);
    expect(cancels).toHaveLength(1);
    expect(cancels[0]).toMatchObject({ userId: alice });
  });
});

/** Finding 14 (completing): the lock is rechecked when the stroke commits. */
describe('lock recheck at commit', () => {
  it('cancels a stroke whose layer was locked mid-draw', () => {
    const { state, alice, bob, layerId } = room();
    begin(state, alice, layerId, 'x');
    applyClientMessage(state, bob, { t: 'layer_update', id: layerId, patch: { locked: true } });

    const revisionBefore = state.humanRevision;
    const end = applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 'x', points: [{ x: 40, y: 40 }] });
    expect(end.broadcast[0]).toMatchObject({ t: 'stroke_cancel', reason: 'layer locked' });
    expect(state.strokes).toHaveLength(0);
    expect(state.humanRevision).toBe(revisionBefore);
  });

  it('commits normally when the layer is still unlocked', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'x');
    expect(applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 'x', points: [{ x: 40, y: 40 }] }).broadcast[0]).toMatchObject({
      t: 'stroke_committed',
    });
  });
});

/** Finding 6 (completing): bounded pending strokes and stroke log. */
describe('stroke limits', () => {
  it('caps simultaneous pending strokes per user', () => {
    const { state, alice, bob, layerId } = room();
    for (let i = 0; i < MAX_PENDING_PER_USER; i++) expect(begin(state, alice, layerId, `s${i}`).relay).toHaveLength(1);

    const refused = begin(state, alice, layerId, 'one-too-many');
    expect(refused.relay).toHaveLength(0);
    expect(refused.toSender?.[1]).toMatchObject({ t: 'error', message: 'too many strokes in progress' });
    expect(state.pending.size).toBe(MAX_PENDING_PER_USER);

    // the cap is per user, not per room
    expect(begin(state, bob, layerId, 'b0').relay).toHaveLength(1);
  });

  it('frees a slot when a stroke ends', () => {
    const { state, alice, layerId } = room();
    for (let i = 0; i < MAX_PENDING_PER_USER; i++) begin(state, alice, layerId, `s${i}`);
    applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 's0', points: [{ x: 30, y: 30 }] });
    expect(begin(state, alice, layerId, 'next').relay).toHaveLength(1);
  });

  it('refuses new strokes once the room stroke log is full', () => {
    const { state, alice, layerId } = room();
    const bbox = { x: 0, y: 0, width: 1, height: 1 };
    state.strokes = Array.from({ length: MAX_STROKES_PER_ROOM }, (_, i) => ({
      id: `filler${i}`,
      userId: alice,
      layerId,
      tool: 'pen' as const,
      color: '#000000',
      width: 1,
      points: [{ x: 0, y: 0 }],
      revision: 1,
      bbox,
    }));
    const refused = begin(state, alice, layerId, 'overflow');
    expect(refused.relay).toHaveLength(0);
    expect(refused.toSender?.[1]).toMatchObject({ t: 'error', message: 'this room has reached its stroke limit' });
  });
});

/** Finding 11 (completing): the session table never drops a live participant. */
describe('session table', () => {
  it('evicts a disconnected session rather than dropping the new token', () => {
    const state = createRoom('sessions');
    const tokens = Array.from({ length: 64 }, (_, i) => `token-${i}`);
    const members = tokens.map((t, i) => joinMember(state, `user${i}`, t));
    expect(state.sessions.size).toBe(64);

    // user0 leaves, so their session is the one that should make way
    removeMember(state, members[0]!.userId);
    const fresh = joinMember(state, 'newcomer', 'token-new');
    expect(state.sessions.get('token-new')?.userId).toBe(fresh.userId);
    expect(state.sessions.has('token-0')).toBe(false);
    expect(state.sessions.has('token-1')).toBe(true);
    expect(state.sessions.size).toBe(64);
  });

  it('keeps resuming an identity that is over the cap boundary', () => {
    const state = createRoom('sessions2');
    const first = joinMember(state, 'Alice', 'token-abc');
    removeMember(state, first.userId);
    for (let i = 0; i < 70; i++) {
      const m = joinMember(state, `user${i}`, `token-${i}`);
      removeMember(state, m.userId);
    }
    // Alice's token may have been evicted, but she still gets a working identity
    expect(joinMember(state, 'Alice', 'token-abc').userId).toBeTruthy();
    expect(state.sessions.size).toBeLessThanOrEqual(64);
  });
});

/** Round 3, finding 4: a full session table never costs a live participant. */
describe('session table under pressure', () => {
  it('refuses to record a new token rather than evicting a connected member', () => {
    const state = createRoom('sessions3');
    const tokens = Array.from({ length: 64 }, (_, i) => `full-${i}`);
    for (const t of tokens) joinMember(state, 'Someone', t);
    expect(state.sessions.size).toBe(64);

    const latecomer = joinMember(state, 'Late', 'full-new');
    // they are in the room...
    expect(state.members.has(latecomer.userId)).toBe(true);
    // ...but nobody else's identity was sacrificed for their token
    expect(state.sessions.size).toBe(64);
    expect(state.sessions.has('full-new')).toBe(false);
    for (const t of tokens) expect(state.sessions.has(t)).toBe(true);
  });

  it('still recycles a slot as soon as one of them disconnects', () => {
    const state = createRoom('sessions4');
    const members = Array.from({ length: 64 }, (_, i) => joinMember(state, `U${i}`, `full-${i}`));
    removeMember(state, members[3]!.userId);

    const latecomer = joinMember(state, 'Late', 'full-new');
    expect(state.sessions.get('full-new')?.userId).toBe(latecomer.userId);
    expect(state.sessions.has('full-3')).toBe(false);
    expect(state.sessions.size).toBe(64);
  });
});

/** Feature: room-level AI settings behave like the prompt. */
describe('ai settings', () => {
  it('applies both values and asks the AI to re-run', () => {
    const { state, alice } = room();
    const out = applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 0.8, negativePrompt: 'blurry, jpeg' });
    expect(out.broadcast[0]).toEqual({ t: 'ai_settings_changed', denoise: 0.8, negativePrompt: 'blurry, jpeg' });
    expect(out.promptChanged).toBe(true);
    // a settings change is not a canvas edit
    expect(out.dirty).toHaveLength(0);
    expect(state.humanRevision).toBe(0);
    expect(state.denoise).toBe(0.8);
    expect(state.negativePrompt).toBe('blurry, jpeg');
  });

  it('changes only what was sent', () => {
    const { state, alice } = room();
    applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 0.8, negativePrompt: 'keep me' });
    applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 0.3 });
    expect(state.negativePrompt).toBe('keep me');
    expect(state.denoise).toBe(0.3);
  });

  it('snaps denoise to the slider grid and clamps it to the range', () => {
    const { state, alice } = room();
    applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 0.77 });
    expect(state.denoise).toBe(0.75);
    applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 99 });
    expect(state.denoise).toBe(MAX_DENOISE);
    applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: -5 });
    expect(state.denoise).toBe(MIN_DENOISE);
  });

  it('does nothing when the values are unchanged', () => {
    const { state, alice } = room();
    applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 0.8 });
    const again = applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 0.8, negativePrompt: '' });
    expect(again.broadcast).toHaveLength(0);
    expect(again.promptChanged).toBeFalsy();
  });

  it('truncates an over-long negative prompt', () => {
    const { state, alice } = room();
    applyClientMessage(state, alice, { t: 'set_ai_settings', negativePrompt: 'z'.repeat(2000) });
    expect(state.negativePrompt).toHaveLength(MAX_NEGATIVE_PROMPT);
  });

  it('carries the settings in the render snapshot and the room snapshot', () => {
    const { state, alice } = room();
    applyClientMessage(state, alice, { t: 'set_ai_settings', denoise: 0.9, negativePrompt: 'no text' });
    expect(captureRenderSnapshot(state)).toMatchObject({ denoise: 0.9, negativePrompt: 'no text' });
    expect(snapshot(state, alice, 'idle', { window: 1024, apply: 768 })).toMatchObject({ denoise: 0.9, negativePrompt: 'no text' });
  });

  it('defaults denoise to the configured value', () => {
    expect(createRoom('cfg', 0.4).denoise).toBe(0.4);
    expect(createRoom('cfg2').negativePrompt).toBe('');
  });
});

/** Draw layers can be moved: the offset is render-time only. */
describe('draw layer offsets', () => {
  function withStroke(): { state: RoomState; alice: string; layerId: string } {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 's');
    applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 's', points: [{ x: 90, y: 90 }] });
    return { state, alice, layerId };
  }

  it('stores the offset and leaves stroke coordinates alone', () => {
    const { state, alice, layerId } = withStroke();
    const before = JSON.stringify(state.strokes[0]!.points);
    const out = applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { offsetX: 120, offsetY: -40 } });
    expect(state.layers[0]).toMatchObject({ offsetX: 120, offsetY: -40 });
    expect(JSON.stringify(state.strokes[0]!.points)).toBe(before);
    // both the old and the new position have to be regenerated
    expect(out.dirty.length).toBeGreaterThanOrEqual(2);
  });

  it('clamps the offset to +/- 2 canvases', () => {
    const { state, alice, layerId } = withStroke();
    applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { offsetX: 999999, offsetY: -999999 } });
    expect(state.layers[0]!.offsetX).toBe(2 * CANVAS_SIZE);
    expect(state.layers[0]!.offsetY).toBe(-2 * CANVAS_SIZE);
  });

  it('ignores an offset on a reference layer', () => {
    const { state, alice } = withStroke();
    state.layers.push({
      id: 'refl',
      name: 'ref',
      kind: 'reference',
      visible: true,
      locked: false,
      opacity: 1,
      order: 5,
      includeInAI: false,
      imageId: 'img',
      x: 0,
      y: 0,
    });
    applyClientMessage(state, alice, { t: 'layer_update', id: 'refl', patch: { offsetX: 50 } });
    expect(state.layers.find((l) => l.id === 'refl')?.offsetX).toBeUndefined();
  });

  it('reports dirty regions and undo in world space', () => {
    const { state, alice, layerId } = withStroke();
    applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { offsetX: 200, offsetY: 100 } });
    const bbox = state.strokes[0]!.bbox;
    const undo = applyClientMessage(state, alice, { t: 'undo' });
    expect(undo.dirty[0]).toMatchObject({ x: bbox.x + 200, y: bbox.y + 100 });
  });

  it('accepts points outside the canvas on a moved layer', () => {
    const { state, alice, layerId } = withStroke();
    applyClientMessage(state, alice, { t: 'layer_update', id: layerId, patch: { offsetX: 2000 } });
    const out = applyClientMessage(state, alice, {
      t: 'stroke_start',
      stroke: { id: 'far', layerId, tool: 'pen', color: '#000000', width: 4, points: [{ x: -800, y: 10 }] },
    });
    expect(out.relay).toHaveLength(1);
    expect((out.relay[0] as { stroke: { points: Array<{ x: number }> } }).stroke.points[0]!.x).toBe(-800);
  });
});
