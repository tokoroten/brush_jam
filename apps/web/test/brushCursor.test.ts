import { describe, expect, it } from 'vitest';
import {
  CROSSHAIR_ARM_PX,
  MIN_RING_RADIUS_PX,
  brushCursor,
  brushScreenRadius,
  drawsWithBrush,
  stageCursor,
  strokePressure,
  type BrushCursorTool,
} from '../src/brushCursor.js';

const base = { tool: 'pen' as BrushCursorTool, width: 40, zoom: 1, panning: false };

describe('how big the ring is', () => {
  it('is half the brush width, in screen pixels', () => {
    // The point of the whole thing: the number on the slider is in world
    // pixels, and at 25% zoom a 40-wide brush lands as a 10-wide mark.
    expect(brushScreenRadius(40, 1)).toBe(20);
    expect(brushScreenRadius(40, 0.25)).toBe(5);
    expect(brushScreenRadius(40, 2)).toBe(40);
    expect(brushScreenRadius(1, 1)).toBe(0.5);
  });

  it('never goes negative or NaN, whatever it is handed', () => {
    // This runs inside the frame loop; a bad number has to become no ring,
    // not an exception sixty times a second.
    expect(brushScreenRadius(Number.NaN, 1)).toBe(0);
    expect(brushScreenRadius(40, Number.NaN)).toBe(0);
    expect(brushScreenRadius(40, Number.POSITIVE_INFINITY)).toBe(0);
    expect(brushScreenRadius(-8, 1)).toBe(0);
  });
});

describe('ring or crosshair', () => {
  it('draws a ring once it is bigger than the line drawing it', () => {
    expect(brushCursor({ ...base, width: 40, zoom: 1 })).toEqual({ kind: 'ring', radius: 20 });
    // exactly at the threshold: a ring, because the rule is "below this"
    expect(brushCursor({ ...base, width: MIN_RING_RADIUS_PX * 2, zoom: 1 })).toEqual({
      kind: 'ring',
      radius: MIN_RING_RADIUS_PX,
    });
  });

  it('falls back to a crosshair when the ring would read as a dot', () => {
    expect(brushCursor({ ...base, width: 4, zoom: 1 })).toEqual({ kind: 'crosshair', radius: 2 });
    // ...which is what a wide brush looks like when the canvas is zoomed out
    expect(brushCursor({ ...base, width: 40, zoom: 0.1 })).toEqual({ kind: 'crosshair', radius: 2 });
    expect(CROSSHAIR_ARM_PX).toBeGreaterThan(0);
  });

  it('is shown for every tool that puts paint down', () => {
    for (const tool of ['pen', 'noise', 'eraser'] as const) {
      expect(drawsWithBrush(tool)).toBe(true);
      expect(brushCursor({ ...base, tool })).not.toBeNull();
    }
    // move drags a layer; there is no footprint to show.
    expect(drawsWithBrush('move')).toBe(false);
    expect(brushCursor({ ...base, tool: 'move' })).toBeNull();
  });

  it('disappears while Space turns the tool into the hand', () => {
    // A ring left up during a pan promises a stroke the drag will not make.
    expect(brushCursor({ ...base, panning: true })).toBeNull();
    expect(brushCursor({ ...base, tool: 'eraser', panning: true })).toBeNull();
  });
});

describe('what the OS cursor does', () => {
  it('gets out of the way exactly where the ring is drawn', () => {
    for (const tool of ['pen', 'noise', 'eraser'] as const) {
      expect(stageCursor({ tool, panning: false })).toBe('none');
    }
  });

  it('comes back for panning and for the move tool', () => {
    expect(stageCursor({ tool: 'move', panning: false })).toBe('default');
    expect(stageCursor({ tool: 'pen', panning: true })).toBe('grab');
    // The AI stage is a viewport: every drag there is a pan, whatever the
    // toolbar says.
    expect(stageCursor({ tool: 'pen', panning: false, viewOnly: true })).toBe('grab');
  });
});

describe('what pressure a point records', () => {
  it('takes a pen at its word', () => {
    expect(strokePressure({ pointerType: 'pen', pressure: 0.3 })).toBe(0.3);
    expect(strokePressure({ pointerType: 'pen', pressure: 1.5 })).toBe(1);
  });
  it('treats a resting pen as full width rather than nothing', () => {
    expect(strokePressure({ pointerType: 'pen', pressure: 0 })).toBe(1);
  });
  it('ignores the constant 0.5 a mouse reports, so the line is as wide as the ring', () => {
    expect(strokePressure({ pointerType: 'mouse', pressure: 0.5 })).toBe(1);
    expect(strokePressure({ pointerType: 'touch', pressure: 0.5 })).toBe(1);
  });
});
