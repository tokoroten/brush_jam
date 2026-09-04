import { describe, expect, it } from 'vitest';
import { pointerIntent } from '../src/Room.js';

/**
 * Review 10 finding 5: the AI viewport accepted drawing input. The stroke was
 * sent, previewed, and then wiped by the next AI result - which reads as the
 * app losing your work rather than as a viewport that does not draw.
 */
const base = { viewOnly: false, button: 0, space: false, shift: false, tool: 'pen' as const };

describe('pointerIntent', () => {
  it('draws on the human canvas with a pen', () => {
    expect(pointerIntent(base)).toBe('draw');
  });

  it('only ever pans on the AI canvas', () => {
    for (const tool of ['pen', 'eraser', 'noise', 'move'] as const) {
      expect(pointerIntent({ ...base, viewOnly: true, tool })).toBe('pan');
    }
  });

  it('still pans on the human canvas with space, shift or the middle button', () => {
    expect(pointerIntent({ ...base, space: true })).toBe('pan');
    expect(pointerIntent({ ...base, shift: true })).toBe('pan');
    expect(pointerIntent({ ...base, button: 1 })).toBe('pan');
  });

  it('moves a layer with the move tool on the human canvas', () => {
    expect(pointerIntent({ ...base, tool: 'move' })).toBe('move');
  });

  it('does not let the move tool drag layers from the AI viewport', () => {
    expect(pointerIntent({ ...base, viewOnly: true, tool: 'move' })).toBe('pan');
  });
});
