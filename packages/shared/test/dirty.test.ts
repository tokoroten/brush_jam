import { describe, expect, it } from 'vitest';
import { mergeDirty, mergeDirtyAll } from '../src/index.js';

const R = (x: number, y: number, w = 10, h = 10) => ({ x, y, width: w, height: h });

describe('mergeDirty', () => {
  it('keeps far-apart regions separate', () => {
    const out = mergeDirtyAll([], [R(0, 0), R(3000, 3000)]);
    expect(out).toHaveLength(2);
  });

  it('merges regions within the padding', () => {
    const out = mergeDirtyAll([], [R(0, 0), R(100, 100)]);
    expect(out).toEqual([{ x: 0, y: 0, width: 110, height: 110 }]);
  });

  it('collapses a chain transitively in a single call', () => {
    const start = [R(0, 0), R(3000, 0)];
    expect(mergeDirty(start, R(1500, 0))).toHaveLength(3);
    const bridged = mergeDirty(start, { x: 0, y: 0, width: 3010, height: 10 });
    expect(bridged).toEqual([{ x: 0, y: 0, width: 3010, height: 10 }]);
  });

  it('appends the merged region last so it is the most recent', () => {
    const out = mergeDirty([R(0, 0)], R(3000, 3000));
    expect(out[out.length - 1]).toEqual(R(3000, 3000));
  });
});
