import { MIN_STROKE_ALPHA } from '@brushjam/shared';
import { describe, expect, it } from 'vitest';
import {
  BRUSH_ALPHA_KEY,
  alphaForTool,
  hasAlpha,
  loadBrushAlphas,
  saveBrushAlphas,
  withAlpha,
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

/**
 * Opacity is remembered per tool for the same reason size is: someone shading
 * with a 30% pen still wants the noise pen at full strength.
 */
describe('per-tool alpha', () => {
  const store = (initial: Record<string, string> = {}): StorageLike & { data: Record<string, string> } => {
    const data = { ...initial };
    return {
      data,
      getItem: (k: string) => data[k] ?? null,
      setItem: (k: string, v: string) => void (data[k] = v),
    };
  };

  it('starts every tool fully opaque', () => {
    const alphas = loadBrushAlphas(store());
    expect(alphas).toEqual({ pen: 1, eraser: 1, noise: 1 });
  });

  it('remembers a value per tool', () => {
    const s = store();
    const next = withAlpha(loadBrushAlphas(s), 'pen', 0.3);
    saveBrushAlphas(s, next);
    const reloaded = loadBrushAlphas(s);
    expect(reloaded.pen).toBe(0.3);
    expect(reloaded.noise).toBe(1);
  });

  it('clamps a stored value that is out of range', () => {
    const s = store({ [BRUSH_ALPHA_KEY]: JSON.stringify({ pen: 4, noise: -1 }) });
    const alphas = loadBrushAlphas(s);
    expect(alphas.pen).toBe(1);
    expect(alphas.noise).toBe(MIN_STROKE_ALPHA);
  });

  it('falls back to the defaults for malformed storage', () => {
    expect(loadBrushAlphas(store({ [BRUSH_ALPHA_KEY]: 'not json' })).pen).toBe(1);
    expect(loadBrushAlphas(store({ [BRUSH_ALPHA_KEY]: '"a string"' })).pen).toBe(1);
    expect(loadBrushAlphas(undefined).pen).toBe(1);
  });

  it('survives storage that throws', () => {
    const hostile: StorageLike = {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
    };
    expect(loadBrushAlphas(hostile).pen).toBe(1);
    expect(() => saveBrushAlphas(hostile, { pen: 0.5, eraser: 1, noise: 1 })).not.toThrow();
  });

  it('never gives the eraser an opacity, however it got stored', () => {
    const s = store({ [BRUSH_ALPHA_KEY]: JSON.stringify({ eraser: 0.2 }) });
    expect(loadBrushAlphas(s).eraser).toBe(1);
    expect(withAlpha(loadBrushAlphas(s), 'eraser', 0.2).eraser).toBe(1);
    expect(alphaForTool({ pen: 0.3, eraser: 1, noise: 0.5 }, 'eraser', 'pen')).toBe(1);
  });

  it('shows the last drawing tool while move is selected', () => {
    const alphas = { pen: 0.3, eraser: 1, noise: 0.8 };
    expect(alphaForTool(alphas, 'move', 'noise')).toBe(0.8);
    expect(alphaForTool(alphas, 'move', 'pen')).toBe(0.3);
  });

  it('offers the slider for pen and noise only', () => {
    expect(hasAlpha('pen')).toBe(true);
    expect(hasAlpha('noise')).toBe(true);
    expect(hasAlpha('eraser')).toBe(false);
    expect(hasAlpha('move')).toBe(false);
  });

  it('keeps size and alpha in separate storage keys', () => {
    const s = store();
    saveBrushAlphas(s, { pen: 0.4, eraser: 1, noise: 1 });
    saveBrushSizes(s, { pen: 20, eraser: 32, noise: 64 });
    expect(loadBrushAlphas(s).pen).toBe(0.4);
    expect(loadBrushSizes(s).pen).toBe(20);
  });
});
