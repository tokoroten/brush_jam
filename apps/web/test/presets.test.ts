import { describe, expect, it } from 'vitest';
import {
  DEFAULT_NEGATIVE_PROMPT,
  DENOISE_STEP,
  MAX_DENOISE,
  MINOR_EXCLUSION_NEGATIVE,
  MIN_DENOISE,
  PRESET_GROUPS,
  PROMPT_PRESETS,
  randomPresets,
  type AIProfileName,
} from '@brushjam/shared';
import {
  applyPromptPreset,
  applyRandomPreset,
  clampPresetDenoise,
  denoiseCeiling,
  presetsInGroup,
  visiblePresetGroups,
  onPresetChange,
  type PresetTarget,
} from '../src/presetPicker.js';
import { editDraft, initialDraft, sentDraft, type DraftState, type SharedDraft } from '../src/sharedDraft.js';

/** A SharedDraft that records what it sent, without React. */
function field<T>(server: T): SharedDraft<T> & { sent: T[] } {
  let state: DraftState<T> = initialDraft(server);
  const sent: T[] = [];
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
    set(value: T) {
      state = editDraft(state, value);
    },
    flush() {
      if (!state.dirty) return;
      if (state.pending !== null && Object.is(state.pending, state.draft)) return;
      sent.push(state.draft);
      state = sentDraft(state, state.draft, server);
    },
    adopt() {},
    onFocus() {},
    onBlur() {},
  };
}

function target(overrides: Partial<PresetTarget> = {}) {
  const prompt = field('old prompt');
  const negative = field('old negative');
  const denoise = field(0.55);
  const profiles: AIProfileName[] = [];
  const t: PresetTarget & { profiles: AIProfileName[] } = {
    prompt,
    negative,
    denoise,
    maxDenoise: MAX_DENOISE,
    profiles: ['fast', 'quality'],
    send: (profile) => profiles.push(profile),
    ...overrides,
  } as PresetTarget & { profiles: AIProfileName[] };
  return { t, prompt, negative, denoise, profiles };
}

describe('prompt presets', () => {
  it('has unique ids, a label, a prompt and a known group in every one', () => {
    expect(PROMPT_PRESETS.length).toBeGreaterThan(20);
    expect(new Set(PROMPT_PRESETS.map((p) => p.id)).size).toBe(PROMPT_PRESETS.length);
    for (const preset of PROMPT_PRESETS) {
      expect(preset.prompt.trim().length, preset.id).toBeGreaterThan(0);
      expect(preset.label.trim().length, preset.id).toBeGreaterThan(0);
      expect(PRESET_GROUPS, preset.id).toContain(preset.group);
    }
  });

  it('every group in the picker has something in it', () => {
    for (const group of PRESET_GROUPS) {
      expect(PROMPT_PRESETS.filter((p) => p.group === group).length, group).toBeGreaterThan(0);
    }
  });

  it('recommends a denoise the slider can actually show', () => {
    const withDenoise = PROMPT_PRESETS.filter((p) => p.denoise !== undefined);
    expect(withDenoise.length).toBeGreaterThan(10);
    for (const preset of withDenoise) {
      const value = preset.denoise!;
      expect(value, preset.id).toBeGreaterThanOrEqual(MIN_DENOISE);
      expect(value, preset.id).toBeLessThanOrEqual(0.9);
      // On the 0.05 grid, to within floating point.
      expect(Math.abs(value / DENOISE_STEP - Math.round(value / DENOISE_STEP)), preset.id).toBeLessThan(1e-9);
    }
  });

  it('keeps the R18 one last, in its own group, named exactly, excluding minors', () => {
    const r18 = PROMPT_PRESETS.at(-1)!;
    expect(r18.id).toBe('r18');
    expect(r18.label).toBe('R18 (NSFW)');
    expect(r18.group).toBe('R18');
    expect(PRESET_GROUPS.at(-1)).toBe('R18');
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

  it('leaves every other preset on the built-in negative list', () => {
    for (const preset of PROMPT_PRESETS.filter((p) => p.id !== 'r18')) {
      expect(preset.negative, preset.id).toBe('');
    }
  });

  it('sends the prompt and the negative prompt once each, immediately', () => {
    const { t, prompt, negative } = target();
    expect(applyPromptPreset('landscape', t)).toBe(true);

    const preset = PROMPT_PRESETS.find((p) => p.id === 'landscape')!;
    expect(prompt.sent).toEqual([preset.prompt]);
    expect(negative.sent).toEqual([preset.negative]);
    expect(prompt.value).toBe(preset.prompt);
  });

  it('sends the denoise a preset asks for, once', () => {
    const { t, denoise } = target();
    applyPromptPreset('sumi-e', t);
    expect(denoise.sent).toEqual([0.65]);
  });

  it('sends no denoise for a preset that does not carry one', () => {
    const { t, denoise } = target();
    applyPromptPreset('bishoujo', t);
    expect(denoise.sent).toEqual([]);
  });

  it('clamps a recommendation to what the backend allows', () => {
    const { t, denoise } = target({ maxDenoise: 0.7 });
    applyPromptPreset('nebula', t); // wants 0.85
    expect(denoise.sent).toEqual([0.7]);

    expect(clampPresetDenoise(0.85, 0.9)).toBe(0.85);
    expect(clampPresetDenoise(0.99, MAX_DENOISE)).toBe(MAX_DENOISE);
    expect(clampPresetDenoise(0.01, MAX_DENOISE)).toBe(MIN_DENOISE);
    // Off the grid, and free of floating-point litter.
    expect(clampPresetDenoise(0.63, MAX_DENOISE)).toBe(0.65);
  });

  it('asks for a profile only when the backend has it', () => {
    const both = target();
    applyPromptPreset('photoreal', both.t);
    expect(both.profiles).toEqual(['quality']);

    const fastOnly = target({ profiles: ['fast'] });
    applyPromptPreset('photoreal', fastOnly.t);
    expect(fastOnly.profiles).toEqual([]);
    // The look still lands, minus the recommendation.
    expect(fastOnly.prompt.sent).toHaveLength(1);
  });

  it('sends no profile for a preset that does not carry one', () => {
    const { t, profiles } = target();
    applyPromptPreset('ukiyo-e', t);
    expect(profiles).toEqual([]);
  });

  it('resets the picker to blank, so it never claims to be room state', () => {
    const { t, prompt } = target();
    const select = { value: 'bishoujo' };
    onPresetChange({ target: select }, t);
    expect(select.value).toBe('');
    expect(prompt.sent).toHaveLength(1);
  });

  it('ignores the blank option and anything unknown', () => {
    const { t, prompt, negative } = target();
    expect(applyPromptPreset('', t)).toBe(false);
    expect(applyPromptPreset('no-such-preset', t)).toBe(false);
    expect(prompt.sent).toEqual([]);
    expect(negative.sent).toEqual([]);
  });

  it('never rolls R18 on the dice', () => {
    // Every position the picker can reach, including both ends.
    for (let i = 0; i < PROMPT_PRESETS.length + 5; i++) {
      const { t } = target();
      const picked = applyRandomPreset(t, () => i / (PROMPT_PRESETS.length + 5));
      expect(picked.id).not.toBe('r18');
    }
    // And with a real random source, many times over.
    const seen = new Set<string>();
    for (let i = 0; i < 500; i++) seen.add(applyRandomPreset(target().t).id);
    expect(seen.has('r18')).toBe(false);
    expect(seen.size).toBeGreaterThan(5);
  });

  it('applies everything the rolled preset carries', () => {
    const { t, prompt, denoise } = target();
    const picked = applyRandomPreset(t, () => 0.999999);
    expect(prompt.sent).toEqual([picked.prompt]);
    if (picked.denoise !== undefined) expect(denoise.sent).toHaveLength(1);
  });

  it('leaves the prompt editable afterwards', () => {
    const { t, prompt } = target();
    applyPromptPreset('bishoujo', t);
    prompt.set('1girl, holding a cat');
    prompt.flush();
    expect(prompt.sent.at(-1)).toBe('1girl, holding a cat');
  });
});

/**
 * Review 2 finding 4: the ceiling was applied after the grid rounding, so a
 * backend cap that is not a multiple of 0.05 produced a value the server
 * either refuses (0.83 rounds up to 0.85, above the cap) or silently changes
 * (0.82 becomes 0.80, and this side waits for an echo of 0.82 forever).
 */
describe('a backend cap that is off the slider grid', () => {
  it('is floored onto the grid, not rounded onto it', () => {
    expect(denoiseCeiling(0.83)).toBe(0.8);
    expect(denoiseCeiling(0.82)).toBe(0.8);
    expect(denoiseCeiling(0.9)).toBe(0.9);
    // 0.95 / 0.05 is 18.999999999999996 in binary floating point.
    expect(denoiseCeiling(MAX_DENOISE)).toBe(MAX_DENOISE);
    expect(denoiseCeiling(0)).toBe(MAX_DENOISE); // unknown means the default
    expect(denoiseCeiling(2)).toBe(MAX_DENOISE); // never above the protocol max
    expect(denoiseCeiling(0.01)).toBe(MIN_DENOISE);
  });

  it('sends a value the room can hold, for either awkward ceiling', () => {
    expect(clampPresetDenoise(0.9, 0.83)).toBe(0.8);
    expect(clampPresetDenoise(0.9, 0.82)).toBe(0.8);
    expect(clampPresetDenoise(0.75, 0.83)).toBe(0.75);
    expect(clampPresetDenoise(0.95, MAX_DENOISE)).toBe(MAX_DENOISE);
  });
});

/**
 * The R18 group is offered only when the server says so (PRESETS_R18). It is
 * a gate on the menu, not on the room: the server accepts any prompt from any
 * client, and the tests on that side say so out loud.
 */
describe('the R18 gate', () => {
  it('hides the group and its presets by default', () => {
    expect(visiblePresetGroups(false)).not.toContain('R18');
    expect(visiblePresetGroups(false)).toEqual(PRESET_GROUPS.filter((g) => g !== 'R18'));
    expect(presetsInGroup('R18', false)).toEqual([]);
  });

  it('shows every group when the server allows it', () => {
    expect(visiblePresetGroups(true)).toEqual(PRESET_GROUPS);
    expect(presetsInGroup('R18', true).length).toBeGreaterThan(0);
  });

  it('leaves the ordinary groups alone either way', () => {
    for (const group of PRESET_GROUPS.filter((g) => g !== 'R18')) {
      expect(presetsInGroup(group, false)).toEqual(presetsInGroup(group, true));
      expect(presetsInGroup(group, false).length).toBeGreaterThan(0);
    }
  });

  it('is not what keeps the dice away from R18 - that is unconditional', () => {
    expect(randomPresets().some((p) => p.group === 'R18')).toBe(false);
  });
});
