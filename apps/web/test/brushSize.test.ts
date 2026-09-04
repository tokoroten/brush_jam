import { describe, expect, it } from 'vitest';
import {
  BRUSH_SIZE_KEY,
  DEFAULT_SIZES,
  MAX_BRUSH,
  MIN_BRUSH,
  isSizedTool,
  loadBrushSizes,
  saveBrushSizes,
  sizeForTool,
  withSize,
  type BrushSizes,
} from '../src/brushSize.js';
import type { StorageLike } from '../src/session.js';

/** In-memory stand-in for localStorage, optionally a hostile one. */
function fakeStorage(initial: Record<string, string> = {}, throws = false): StorageLike & { data: Record<string, string> } {
  const data = { ...initial };
  return {
    data,
    getItem(key) {
      if (throws) throw new Error('blocked');
      return data[key] ?? null;
    },
    setItem(key, value) {
      if (throws) throw new Error('quota');
      data[key] = value;
    },
  };
}

describe('per-tool brush sizes', () => {
  it('starts each tool at a size that suits it', () => {
    expect(DEFAULT_SIZES).toEqual({ pen: 14, eraser: 32, noise: 64 });
  });

  it('remembers a size per tool rather than globally', () => {
    let sizes: BrushSizes = { ...DEFAULT_SIZES };
    sizes = withSize(sizes, 'pen', 8);
    sizes = withSize(sizes, 'noise', 96);
    expect(sizes).toEqual({ pen: 8, eraser: 32, noise: 96 });
    // switching back to a tool restores what it was left on
    expect(sizeForTool(sizes, 'pen', 'pen')).toBe(8);
    expect(sizeForTool(sizes, 'noise', 'pen')).toBe(96);
    expect(sizeForTool(sizes, 'eraser', 'pen')).toBe(32);
  });

  it('keeps showing the last drawing size while the move tool is active', () => {
    const sizes = withSize({ ...DEFAULT_SIZES }, 'noise', 96);
    expect(sizeForTool(sizes, 'move', 'noise')).toBe(96);
    expect(sizeForTool(sizes, 'move', 'pen')).toBe(14);
  });

  it('knows which tools have a size at all', () => {
    expect(isSizedTool('pen')).toBe(true);
    expect(isSizedTool('noise')).toBe(true);
    expect(isSizedTool('eraser')).toBe(true);
    expect(isSizedTool('move')).toBe(false);
  });

  it('clamps a size to the slider range', () => {
    expect(withSize({ ...DEFAULT_SIZES }, 'pen', 0).pen).toBe(MIN_BRUSH);
    expect(withSize({ ...DEFAULT_SIZES }, 'pen', 9999).pen).toBe(MAX_BRUSH);
    expect(withSize({ ...DEFAULT_SIZES }, 'pen', 12.7).pen).toBe(13);
  });
});

describe('persistence', () => {
  it('round-trips through storage', () => {
    const storage = fakeStorage();
    const sizes = withSize(withSize({ ...DEFAULT_SIZES }, 'noise', 100), 'eraser', 48);
    saveBrushSizes(storage, sizes);
    expect(loadBrushSizes(storage)).toEqual({ pen: 14, eraser: 48, noise: 100 });
    expect(Object.keys(storage.data)).toEqual([BRUSH_SIZE_KEY]);
  });

  it('uses the defaults when nothing is stored', () => {
    expect(loadBrushSizes(fakeStorage())).toEqual(DEFAULT_SIZES);
    expect(loadBrushSizes(undefined)).toEqual(DEFAULT_SIZES);
  });

  it('survives malformed or partial stored data', () => {
    expect(loadBrushSizes(fakeStorage({ [BRUSH_SIZE_KEY]: 'not json' }))).toEqual(DEFAULT_SIZES);
    expect(loadBrushSizes(fakeStorage({ [BRUSH_SIZE_KEY]: 'null' }))).toEqual(DEFAULT_SIZES);
    expect(loadBrushSizes(fakeStorage({ [BRUSH_SIZE_KEY]: '"14"' }))).toEqual(DEFAULT_SIZES);
    // a value written by an older build: keep what is usable, default the rest
    expect(loadBrushSizes(fakeStorage({ [BRUSH_SIZE_KEY]: '{"pen":20}' }))).toEqual({ pen: 20, eraser: 32, noise: 64 });
  });

  it('clamps stored values that are out of range', () => {
    const stored = JSON.stringify({ pen: -5, eraser: 5000, noise: 'wide' });
    expect(loadBrushSizes(fakeStorage({ [BRUSH_SIZE_KEY]: stored }))).toEqual({
      pen: MIN_BRUSH,
      eraser: MAX_BRUSH,
      noise: DEFAULT_SIZES.noise,
    });
  });

  it('never lets blocked storage break the editor', () => {
    const hostile = fakeStorage({}, true);
    expect(loadBrushSizes(hostile)).toEqual(DEFAULT_SIZES);
    expect(() => saveBrushSizes(hostile, DEFAULT_SIZES)).not.toThrow();
  });
});
