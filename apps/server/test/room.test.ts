import { describe, expect, it } from 'vitest';
import {
  addMember,
  applyClientMessage,
  captureRenderSnapshot,
  createRoom,
  joinMember,
  qualifyStrokeId,
  removeMember,
  snapshot,
  strokesForCrop,
  type RoomState,
} from '../src/room.js';

function room(): { state: RoomState; alice: string; bob: string; layerId: string } {
  const state = createRoom('testroom');
  const alice = addMember(state, 'Alice').userId;
  const bob = addMember(state, 'Bob').userId;
  return { state, alice, bob, layerId: state.layers[0]!.id };
}

function draw(state: RoomState, userId: string, layerId: string, id: string, x = 100, y = 100): void {
  applyClientMessage(state, userId, { t: 'stroke_start', stroke: { id, layerId, tool: 'pen', color: '#112233', width: 10, points: [{ x, y }] } });
  applyClientMessage(state, userId, { t: 'stroke_chunk', strokeId: id, points: [{ x: x + 10, y: y + 10 }] });
  applyClientMessage(state, userId, { t: 'stroke_end', strokeId: id, points: [{ x: x + 20, y: y + 20 }] });
}

describe('room reducer', () => {
  it('starts with one draw layer and no strokes', () => {
    const { state } = room();
    expect(state.layers).toHaveLength(1);
    expect(state.layers[0]!.kind).toBe('draw');
    expect(state.humanRevision).toBe(0);
  });

  it('gives members distinct ids and colors', () => {
    const { state, alice, bob } = room();
    expect(alice).not.toBe(bob);
    expect(state.members.get(alice)!.color).not.toBe(state.members.get(bob)!.color);
  });

  it('commits a stroke and bumps the revision exactly once', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 's1');
    expect(state.strokes).toHaveLength(1);
    expect(state.strokes[0]!.points).toHaveLength(3);
    expect(state.humanRevision).toBe(1);
    expect(state.strokes[0]!.userId).toBe(alice);
    expect(state.strokes[0]!.id).toBe(qualifyStrokeId(alice, 's1'));
  });

  it('reports a dirty rect covering the stroke', () => {
    const { state, alice, layerId } = room();
    applyClientMessage(state, alice, { t: 'stroke_start', stroke: { id: 'x', layerId, tool: 'pen', color: '#000000', width: 10, points: [{ x: 500, y: 500 }] } });
    const res = applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 'x', points: [{ x: 520, y: 540 }] });
    expect(res.dirty).toHaveLength(1);
    expect(res.dirty[0]!.x).toBeLessThan(500);
    expect(res.dirty[0]!.width).toBeGreaterThan(20);
  });

  it('ignores stroke chunks from another user', () => {
    const { state, alice, bob, layerId } = room();
    applyClientMessage(state, alice, { t: 'stroke_start', stroke: { id: 'a', layerId, tool: 'pen', color: '#000000', width: 4, points: [{ x: 1, y: 1 }] } });
    const res = applyClientMessage(state, bob, { t: 'stroke_chunk', strokeId: 'a', points: [{ x: 9, y: 9 }] });
    expect(res.relay).toHaveLength(0);
    expect(state.pending.size).toBe(1);
  });

  it('rejects strokes on a locked or missing layer', () => {
    const { state, alice, layerId } = room();
    state.layers[0]!.locked = true;
    const res = applyClientMessage(state, alice, { t: 'stroke_start', stroke: { id: 'a', layerId, tool: 'pen', color: '#000000', width: 4, points: [] } });
    expect(res.relay).toHaveLength(0);
    expect(res.toSender?.[0]).toMatchObject({ t: 'error' });
    expect(state.pending.size).toBe(0);
  });

  it('sanitises color and width', () => {
    const { state, alice, layerId } = room();
    applyClientMessage(state, alice, { t: 'stroke_start', stroke: { id: 'a', layerId, tool: 'pen', color: 'red', width: 9999, points: [{ x: 0, y: 0 }] } });
    applyClientMessage(state, alice, { t: 'stroke_end', strokeId: 'a', points: [] });
    const s = state.strokes[0]!;
    expect(s.tool).toBe('pen');
    expect(s.color).toBe('#000000');
    expect(s.width).toBe(128);
  });

  it('undoes the sender own last stroke (Alice/Bob/Alice)', () => {
    const { state, alice, bob, layerId } = room();
    draw(state, alice, layerId, '100');
    draw(state, bob, layerId, '101');
    draw(state, alice, layerId, '102');
    const res = applyClientMessage(state, alice, { t: 'undo' });
    expect(res.broadcast[0]).toMatchObject({ t: 'undo_applied', strokeId: qualifyStrokeId(alice, '102') });
    expect([...state.undone]).toEqual([qualifyStrokeId(alice, '102')]);
    expect(state.humanRevision).toBe(4);
    const second = applyClientMessage(state, alice, { t: 'undo' });
    expect(second.broadcast[0]).toMatchObject({ strokeId: qualifyStrokeId(alice, '100') });
  });

  it('is a no-op when a user has nothing to undo', () => {
    const { state, alice, bob, layerId } = room();
    draw(state, alice, layerId, '100');
    const before = state.humanRevision;
    expect(applyClientMessage(state, bob, { t: 'undo' }).broadcast).toHaveLength(0);
    expect(state.humanRevision).toBe(before);
  });

  it('clears a layer for everyone and is not undoable', () => {
    const { state, alice, bob, layerId } = room();
    draw(state, alice, layerId, '1');
    draw(state, bob, layerId, '2');
    const res = applyClientMessage(state, alice, { t: 'clear_layer', layerId });
    expect(res.broadcast[0]).toMatchObject({ t: 'clear_applied', layerId });
    expect(state.strokes).toHaveLength(0);
    expect(applyClientMessage(state, alice, { t: 'undo' }).broadcast).toHaveLength(0);
  });

  it('creates, updates, reorders and deletes layers', () => {
    const { state, alice, layerId } = room();
    const created = applyClientMessage(state, alice, { t: 'layer_create', layer: { kind: 'draw' } });
    expect(state.layers).toHaveLength(2);
    const newId = (created.broadcast[0] as { layer: { id: string } }).layer.id;

    applyClientMessage(state, alice, { t: 'layer_update', id: newId, patch: { name: 'Sky', opacity: 0.5, visible: false } });
    const layer = state.layers.find((l) => l.id === newId)!;
    expect(layer).toMatchObject({ name: 'Sky', opacity: 0.5, visible: false });

    applyClientMessage(state, alice, { t: 'layer_reorder', ids: [newId, layerId] });
    expect(state.layers.find((l) => l.id === newId)!.order).toBe(0);

    applyClientMessage(state, alice, { t: 'layer_delete', id: newId });
    expect(state.layers).toHaveLength(1);
  });

  it('refuses to delete the last draw layer and caps layer count', () => {
    const { state, alice, layerId } = room();
    expect(applyClientMessage(state, alice, { t: 'layer_delete', id: layerId }).broadcast).toHaveLength(0);
    for (let i = 0; i < 20; i++) applyClientMessage(state, alice, { t: 'layer_create', layer: { kind: 'draw' } });
    expect(state.layers).toHaveLength(8);
  });

  it('adds reference layers that are excluded from AI input by default', () => {
    const { state, alice } = room();
    state.images.set('img1', { id: 'img1', mime: 'image/png', bytes: Buffer.alloc(4), width: 100, height: 80 });
    const res = applyClientMessage(state, alice, { t: 'layer_create', layer: { kind: 'reference', imageId: 'img1', x: 10, y: 20 } });
    const layer = (res.broadcast[0] as { layer: { includeInAI: boolean; imageWidth: number } }).layer;
    expect(layer.includeInAI).toBe(false);
    expect(layer.imageWidth).toBe(100);
    expect(res.dirty).toHaveLength(0);
  });

  it('changes the prompt once and flags a re-run', () => {
    const { state, alice } = room();
    const res = applyClientMessage(state, alice, { t: 'set_prompt', prompt: 'a cat' });
    expect(res.promptChanged).toBe(true);
    expect(state.prompt).toBe('a cat');
    expect(applyClientMessage(state, alice, { t: 'set_prompt', prompt: 'a cat' }).promptChanged).toBeUndefined();
  });

  it('selects only strokes intersecting a crop', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 'near', 100, 100);
    draw(state, alice, layerId, 'far', 3000, 3000);
    const picked = strokesForCrop(state, { x: 0, y: 0, width: 512, height: 512 });
    expect(picked.map((s) => s.id)).toEqual([qualifyStrokeId(alice, 'near')]);
  });

  it('excludes undone strokes from crop selection', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 'a', 100, 100);
    applyClientMessage(state, alice, { t: 'undo' });
    expect(strokesForCrop(state, { x: 0, y: 0, width: 512, height: 512 })).toHaveLength(0);
  });

  it('snapshots the full recoverable state', () => {
    const { state, alice, layerId } = room();
    draw(state, alice, layerId, 'a');
    applyClientMessage(state, alice, { t: 'undo' });
    const snap = snapshot(state, alice, 'idle', { window: 1024, apply: 768 });
    expect(snap).toMatchObject({ roomId: 'testroom', youUserId: alice, undone: [qualifyStrokeId(alice, 'a')], humanRevision: 2, aiWindow: 1024, aiApply: 768 });
    expect(snap.strokes).toHaveLength(1);
    expect(snap.members).toHaveLength(2);
  });
});
