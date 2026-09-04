import { describe, expect, it } from 'vitest';
import {
  addMember,
  applyClientMessage,
  captureRenderSnapshot,
  createRoom,
  joinMember,
  qualifyStrokeId,
  removeMember,
  strokesForCrop,
  type RoomState,
} from '../src/room.js';

/** Regression tests for the review findings that live in the room reducer. */
function room(): { state: RoomState; alice: string; bob: string; layerId: string } {
  const state = createRoom('reviewroom');
  const alice = addMember(state, 'Alice').userId;
  const bob = addMember(state, 'Bob').userId;
  return { state, alice, bob, layerId: state.layers[0]!.id };
}

function draw(state: RoomState, userId: string, layerId: string, id: string, x = 100, y = 100): void {
  applyClientMessage(state, userId, {
    t: 'stroke_start',
    stroke: { id, layerId, tool: 'pen', color: '#112233', width: 10, points: [{ x, y }] },
  });
  applyClientMessage(state, userId, { t: 'stroke_end', strokeId: id, points: [{ x: x + 20, y: y + 20 }] });
}

function begin(state: RoomState, userId: string, layerId: string, id: string): void {
  applyClientMessage(state, userId, {
    t: 'stroke_start',
    stroke: { id, layerId, tool: 'pen', color: '#000000', width: 8, points: [{ x: 10, y: 10 }] },
  });
}

describe('stroke identity (finding 13)', () => {
  it('namespaces ids so two users cannot collide on the same raw id', () => {
    const { state, alice, bob, layerId } = room();
    draw(state, alice, layerId, 'same');
    draw(state, bob, layerId, 'same');
    expect(state.strokes.map((s) => s.id)).toEqual([qualifyStrokeId(alice, 'same'), qualifyStrokeId(bob, 'same')]);

    applyClientMessage(state, alice, { t: 'undo' });
    expect(state.undone.has(qualifyStrokeId(alice, 'same'))).toBe(true);
    expect(state.undone.has(qualifyStrokeId(bob, 'same'))).toBe(false);
    expect(strokesForCrop(state, { x: 0, y: 0, width: 512, height: 512 }).map((s) => s.userId)).toEqual([bob]);
  });

  it('rejects a duplicate id from the same user', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 'dup');
    const res = applyClientMessage(state, alice, {
      t: 'stroke_start',
      stroke: { id: 'dup', layerId, tool: 'pen', color: '#000000', width: 4, points: [{ x: 1, y: 1 }] },
    });
    expect(res.toSender?.[0]).toMatchObject({ t: 'stroke_cancel', strokeId: qualifyStrokeId(alice, 'dup') });
    expect(res.toSender?.[1]).toMatchObject({ t: 'error', message: 'duplicate stroke id' });
    expect(state.pending.size).toBe(0);
  });

  it('rejects a second stroke_start while the same id is pending', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'p');
    const again = applyClientMessage(state, alice, {
      t: 'stroke_start',
      stroke: { id: 'p', layerId, tool: 'pen', color: '#000000', width: 4, points: [{ x: 1, y: 1 }] },
    });
    expect(again.toSender?.[0]).toMatchObject({ t: 'stroke_cancel' });
    expect(again.toSender?.[1]).toMatchObject({ t: 'error' });
    expect(state.pending.size).toBe(1);
  });
});

describe('pending stroke lifecycle (finding 14)', () => {
  it('cancels pending strokes when their layer is deleted and refuses the late end', () => {
    const { state, alice } = room();
    const created = applyClientMessage(state, alice, { t: 'layer_create', layer: { kind: 'draw' } });
    const newId = (created.broadcast[0] as { layer: { id: string } }).layer.id;
    begin(state, alice, newId, 'x');

    const del = applyClientMessage(state, alice, { t: 'layer_delete', id: newId });
    expect(del.broadcast.some((m) => m.t === 'stroke_cancel')).toBe(true);
    expect(state.pending.size).toBe(0);

    const revisionBefore = state.humanRevision;
    const end = applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 'x', points: [{ x: 20, y: 20 }] });
    expect(end.broadcast).toHaveLength(0);
    expect(state.strokes).toHaveLength(0);
    expect(state.humanRevision).toBe(revisionBefore);
  });

  it('cancels pending strokes when the author leaves', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'y');
    const cancels = removeMember(state, alice);
    expect(cancels).toHaveLength(1);
    expect(cancels[0]).toMatchObject({ t: 'stroke_cancel', strokeId: qualifyStrokeId(alice, 'y') });
    expect(state.pending.size).toBe(0);
  });

  it('cancels a stroke that exceeds the point limit', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'z');
    const many = Array.from({ length: 20000 }, (_, i) => ({ x: i % 1000, y: 5 }));
    let cancelled = false;
    for (let i = 0; i < 4 && !cancelled; i++) {
      const res = applyClientMessage(state, alice, { t: 'stroke_chunk', strokeId: 'z', points: many });
      cancelled = res.broadcast.some((m) => m.t === 'stroke_cancel');
    }
    expect(cancelled).toBe(true);
    expect(state.pending.size).toBe(0);
  });

  it('cancels pending strokes on a cleared layer', () => {
    const { state, alice, layerId } = room();
    begin(state, alice, layerId, 'c');
    expect(applyClientMessage(state, alice, { t: 'clear_layer', layerId }).broadcast.some((m) => m.t === 'stroke_cancel')).toBe(true);
  });
});

describe('clear_layer dirty union (finding 16)', () => {
  it('ignores strokes that were already undone', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 'far', 3000, 3000);
    applyClientMessage(state, alice, { t: 'undo' });
    draw(state, alice, layerId, 'near', 100, 100);

    const res = applyClientMessage(state, alice, { t: 'clear_layer', layerId });
    expect(res.dirty).toHaveLength(1);
    expect(res.dirty[0]!.width).toBeLessThan(200);
    expect(res.dirty[0]!.x).toBeLessThan(200);
  });

  it('reports no dirty area when every removed stroke was undone', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 'only');
    applyClientMessage(state, alice, { t: 'undo' });
    expect(applyClientMessage(state, alice, { t: 'clear_layer', layerId }).dirty).toHaveLength(0);
  });
});

describe('reference layer AI input (finding 9)', () => {
  const withReference = (): { state: RoomState; alice: string; refId: string } => {
    const { state, alice } = room();
    state.images.set('img1', { id: 'img1', mime: 'image/png', bytes: Buffer.alloc(4), width: 200, height: 100, createdAt: Date.now() });
    const created = applyClientMessage(state, alice, {
      t: 'layer_create',
      layer: { kind: 'reference', imageId: 'img1', x: 500, y: 500 },
    });
    return { state, alice, refId: (created.broadcast[0] as { layer: { id: string } }).layer.id };
  };

  it('dirties the previous bounds when AI input is switched off', () => {
    const { state, alice, refId } = withReference();
    applyClientMessage(state, alice, { t: 'layer_update', id: refId, patch: { includeInAI: true } });
    const off = applyClientMessage(state, alice, { t: 'layer_update', id: refId, patch: { includeInAI: false } });
    expect(off.dirty.length).toBeGreaterThan(0);
    expect(off.dirty[0]).toMatchObject({ x: 500, y: 500, width: 200, height: 100 });
  });

  it('stays clean while the layer is not AI input in either state', () => {
    const { state, alice, refId } = withReference();
    expect(applyClientMessage(state, alice, { t: 'layer_update', id: refId, patch: { x: 800 } }).dirty).toHaveLength(0);
  });
});

describe('reconnect identity (finding 11)', () => {
  it('resumes the same userId for a returning token so undo still works', () => {
    const state = createRoom('resume');
    const first = joinMember(state, 'Alice', 'token-abc');
    const layerId = state.layers[0]!.id;
    draw(state, first.userId, layerId, 's1');

    removeMember(state, first.userId);
    const second = joinMember(state, 'Alice', 'token-abc');
    expect(second.userId).toBe(first.userId);
    expect(applyClientMessage(state, second.userId, { t: 'undo' }).broadcast[0]).toMatchObject({ t: 'undo_applied' });
  });

  it('rebinds the identity when the token returns while the old socket looks alive', () => {
    // A dead TCP connection the server has not noticed yet must not fork the
    // participant into two identities; the caller replaces the stale socket.
    const state = createRoom('resume2');
    const first = joinMember(state, 'Alice', 'token-abc');
    expect(joinMember(state, 'Alice', 'token-abc').userId).toBe(first.userId);
    expect(state.members.size).toBe(1);
  });

  it('mints a fresh identity without a token', () => {
    const state = createRoom('resume3');
    const first = joinMember(state, 'Alice');
    removeMember(state, first.userId);
    expect(joinMember(state, 'Alice').userId).not.toBe(first.userId);
  });
});

describe('captureRenderSnapshot (finding 10)', () => {
  it('is unaffected by later mutations', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 'first');
    const snap = captureRenderSnapshot(state);
    expect(snap.revision).toBe(1);

    draw(state, alice, layerId, 'second', 900, 900);
    applyClientMessage(state, alice, { t: 'set_prompt', prompt: 'changed' });
    applyClientMessage(state, alice, { t: 'undo' });

    expect(snap.strokes).toHaveLength(1);
    expect(snap.undone.size).toBe(0);
    expect(snap.prompt).not.toBe('changed');
    expect(state.strokes).toHaveLength(2);
  });
});
