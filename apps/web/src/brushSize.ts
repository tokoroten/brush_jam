import { DEFAULT_STROKE_ALPHA, MAX_STROKE_ALPHA, MIN_STROKE_ALPHA } from '@brushjam/shared';
import type { StorageLike } from './session.js';

/**
 * Brush size is remembered per tool, because the tools want very different
 * widths: a 14 px noise stroke is useless as an "invent something here" seed
 * (it downsamples to nothing before the model sees it), and an eraser sized
 * for line work is painful for clearing an area.
 */
export type SizedTool = 'pen' | 'eraser' | 'noise';

export const DEFAULT_SIZES: Record<SizedTool, number> = { pen: 14, eraser: 32, noise: 64 };

export const MIN_BRUSH = 1;
export const MAX_BRUSH = 128;

export const BRUSH_SIZE_KEY = 'brushjam.brushSizes';
export const BRUSH_ALPHA_KEY = 'brushjam.brushAlphas';

/**
 * Opacity is remembered per tool for the same reason size is: someone building
 * up soft shading with a 30% pen still wants the noise pen at full strength,
 * where it is a seed for the model rather than a mark.
 */
export const DEFAULT_ALPHAS: Record<SizedTool, number> = {
  pen: DEFAULT_STROKE_ALPHA,
  eraser: 1,
  noise: DEFAULT_STROKE_ALPHA,
};

/** localStorage when the browser allows it; undefined in tests and SSR. */
export function brushStorage(): StorageLike | undefined {
  try {
    return typeof localStorage === 'undefined' ? undefined : localStorage;
  } catch {
    return undefined;
  }
}

export type BrushSizes = Record<SizedTool, number>;
export type BrushAlphas = Record<SizedTool, number>;

/** `move` has no width of its own; it keeps showing the last drawing tool's. */
export function isSizedTool(tool: string): tool is SizedTool {
  return tool === 'pen' || tool === 'eraser' || tool === 'noise';
}

function clamp(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(MAX_BRUSH, Math.max(MIN_BRUSH, Math.round(value)));
}

/**
 * Read the remembered sizes. Anything missing or nonsensical falls back to the
 * default for that tool rather than failing: storage can hold a value written
 * by an older build, and it can throw outright (private mode, blocked site
 * data), which must never stop the editor from opening.
 */
export function loadBrushSizes(storage: StorageLike | undefined): BrushSizes {
  const sizes = { ...DEFAULT_SIZES };
  try {
    const raw = storage?.getItem(BRUSH_SIZE_KEY);
    if (!raw) return sizes;
    const parsed = JSON.parse(raw) as Partial<Record<SizedTool, unknown>>;
    if (typeof parsed !== 'object' || parsed === null) return sizes;
    for (const tool of Object.keys(sizes) as SizedTool[]) {
      sizes[tool] = clamp(parsed[tool], DEFAULT_SIZES[tool]);
    }
  } catch {
    /* unreadable or malformed: the defaults are a perfectly good answer */
  }
  return sizes;
}

/** Best effort: a full or blocked quota must not break drawing. */
export function saveBrushSizes(storage: StorageLike | undefined, sizes: BrushSizes): void {
  try {
    storage?.setItem(BRUSH_SIZE_KEY, JSON.stringify(sizes));
  } catch {
    /* the size is still correct in memory for this session */
  }
}

/** The width to draw with, given the active tool and the remembered sizes. */
export function sizeForTool(sizes: BrushSizes, tool: string, lastSized: SizedTool): number {
  return sizes[isSizedTool(tool) ? tool : lastSized];
}

export function withSize(sizes: BrushSizes, tool: SizedTool, size: number): BrushSizes {
  return { ...sizes, [tool]: clamp(size, DEFAULT_SIZES[tool]) };
}

/** Alpha is a fraction, not a rounded pixel count, so it clamps differently. */
function clampAlpha(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(MAX_STROKE_ALPHA, Math.max(MIN_STROKE_ALPHA, value));
}

export function loadBrushAlphas(storage: StorageLike | undefined): BrushAlphas {
  const alphas = { ...DEFAULT_ALPHAS };
  try {
    const raw = storage?.getItem(BRUSH_ALPHA_KEY);
    if (!raw) return alphas;
    const parsed = JSON.parse(raw) as Partial<Record<SizedTool, unknown>>;
    if (typeof parsed !== 'object' || parsed === null) return alphas;
    for (const tool of Object.keys(alphas) as SizedTool[]) {
      alphas[tool] = clampAlpha(parsed[tool], DEFAULT_ALPHAS[tool]);
    }
  } catch {
    /* unreadable or malformed: the defaults are a perfectly good answer */
  }
  // The eraser has no opacity setting, whatever storage happens to hold.
  alphas.eraser = 1;
  return alphas;
}

export function saveBrushAlphas(storage: StorageLike | undefined, alphas: BrushAlphas): void {
  try {
    storage?.setItem(BRUSH_ALPHA_KEY, JSON.stringify(alphas));
  } catch {
    /* the value is still correct in memory for this session */
  }
}

/** The opacity to draw with, given the active tool. */
export function alphaForTool(alphas: BrushAlphas, tool: string, lastSized: SizedTool): number {
  const which = isSizedTool(tool) ? tool : lastSized;
  return which === 'eraser' ? 1 : alphas[which];
}

export function withAlpha(alphas: BrushAlphas, tool: SizedTool, alpha: number): BrushAlphas {
  if (tool === 'eraser') return alphas;
  return { ...alphas, [tool]: clampAlpha(alpha, DEFAULT_ALPHAS[tool]) };
}

/** Whether the alpha slider applies to this tool at all. */
export function hasAlpha(tool: string): boolean {
  return tool === 'pen' || tool === 'noise';
}
