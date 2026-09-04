import { describe, expect, it } from 'vitest';
import { selectUndoTarget, shouldAcceptResult } from '../src/index.js';

describe('shouldAcceptResult', () => {
  it('drops results older than the last accepted AI revision', () => {
    expect(shouldAcceptResult(142, 146)).toBe(false);
    expect(shouldAcceptResult(146, 146)).toBe(true);
    expect(shouldAcceptResult(150, 146)).toBe(true);
  });
});

describe('selectUndoTarget', () => {
  const strokes = [
    { id: '100', userId: 'alice' },
    { id: '101', userId: 'bob' },
    { id: '102', userId: 'alice' },
  ];

  it('undoes the sender own latest stroke, not the global latest', () => {
    expect(selectUndoTarget(strokes, new Set(), 'alice')?.id).toBe('102');
    expect(selectUndoTarget(strokes, new Set(), 'bob')?.id).toBe('101');
  });

  it('skips already-undone strokes and walks further back', () => {
    expect(selectUndoTarget(strokes, new Set(['102']), 'alice')?.id).toBe('100');
    expect(selectUndoTarget(strokes, new Set(['100', '102']), 'alice')).toBeNull();
  });

  it('returns null for a user with no strokes', () => {
    expect(selectUndoTarget(strokes, new Set(), 'carol')).toBeNull();
  });
});
