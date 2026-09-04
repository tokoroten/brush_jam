import { describe, expect, it } from 'vitest';
import {
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
