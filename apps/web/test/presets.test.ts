import { describe, expect, it } from 'vitest';
import { DEFAULT_NEGATIVE_PROMPT, MINOR_EXCLUSION_NEGATIVE, PROMPT_PRESETS } from '@brushjam/shared';
import { applyPromptPreset, onPresetChange } from '../src/presetPicker.js';
import { DRAFT_DEBOUNCE_MS, initialDraft, sentDraft, editDraft, type DraftState } from '../src/sharedDraft.js';
import type { SharedDraft } from '../src/sharedDraft.js';

/** A SharedDraft that records what it sent, without React. */
function field(server: string): SharedDraft<string> & { sent: string[] } {
  let state: DraftState<string> = initialDraft(server);
  const sent: string[] = [];
  return {
    sent,
    get value() {
      return state.draft;
    },
    get dirty() {
      return state.dirty;
    },
    get foreign() {
      return state.foreign;
    },
    set(value: string) {
      state = editDraft(state, value);
    },
    flush() {
      if (!state.dirty) return;
      if (state.pending !== null && state.pending === state.draft) return;
      sent.push(state.draft);
      state = sentDraft(state, state.draft, server);
    },
    adopt() {},
    onFocus() {},
    onBlur() {},
  };
}

describe('prompt presets', () => {
  it('has unique ids and a prompt in every one', () => {
    expect(PROMPT_PRESETS.length).toBeGreaterThan(1);
    expect(new Set(PROMPT_PRESETS.map((p) => p.id)).size).toBe(PROMPT_PRESETS.length);
    for (const preset of PROMPT_PRESETS) {
      expect(preset.prompt.trim().length, preset.id).toBeGreaterThan(0);
      expect(preset.label.trim().length, preset.id).toBeGreaterThan(0);
    }
  });

  it('keeps the R18 one last, named exactly, and excluding minors', () => {
    const r18 = PROMPT_PRESETS.at(-1)!;
    expect(r18.id).toBe('r18');
    expect(r18.label).toBe('R18 (NSFW)');
    for (const tag of ['child', 'loli', 'shota', 'underage', 'teen']) {
      expect(r18.negative, tag).toContain(tag);
    }
    expect(r18.negative).toContain(MINOR_EXCLUSION_NEGATIVE);
    // Prepended, not instead of: the ordinary quality negatives stay.
    expect(r18.negative).toContain(DEFAULT_NEGATIVE_PROMPT);
    // And the adult tags are in the POSITIVE prompt, because the negative one
    // is inert on the fast profile (CFG 1.0).
    expect(r18.prompt).toContain('adult woman');
    expect(r18.prompt).toContain('mature female');
  });

  it('leaves the other presets on the built-in negative list', () => {
    for (const preset of PROMPT_PRESETS.filter((p) => p.id !== 'r18')) {
      expect(preset.negative, preset.id).toBe('');
    }
  });

  it('sends the prompt and the negative prompt once each, immediately', () => {
    const prompt = field('old prompt');
    const negative = field('old negative');
    expect(applyPromptPreset('landscape', { prompt, negative })).toBe(true);

    const preset = PROMPT_PRESETS.find((p) => p.id === 'landscape')!;
    expect(prompt.sent).toEqual([preset.prompt]);
    expect(negative.sent).toEqual([preset.negative]);
    // Immediately: nothing is waiting on the debounce.
    expect(prompt.value).toBe(preset.prompt);
    expect(DRAFT_DEBOUNCE_MS).toBeGreaterThan(0);
  });

  it('resets the picker to blank, so it never claims to be room state', () => {
    const prompt = field('');
    const negative = field('');
    const target = { value: 'bishoujo' };
    onPresetChange({ target }, { prompt, negative });
    expect(target.value).toBe('');
    expect(prompt.sent).toHaveLength(1);
  });

  it('ignores the blank option and anything unknown', () => {
    const prompt = field('');
    const negative = field('');
    expect(applyPromptPreset('', { prompt, negative })).toBe(false);
    expect(applyPromptPreset('no-such-preset', { prompt, negative })).toBe(false);
    expect(prompt.sent).toEqual([]);
    expect(negative.sent).toEqual([]);
  });

  it('leaves the prompt editable afterwards', () => {
    const prompt = field('');
    const negative = field('');
    applyPromptPreset('bishoujo', { prompt, negative });
    prompt.set('1girl, holding a cat');
    prompt.flush();
    expect(prompt.sent.at(-1)).toBe('1girl, holding a cat');
  });
});
