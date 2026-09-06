import {
  DENOISE_STEP,
  MAX_DENOISE,
  MIN_DENOISE,
  PROMPT_PRESETS,
  randomPresets,
  type AIProfileName,
  type PromptPreset,
} from '@brushjam/shared';
import type { SharedDraft } from './sharedDraft.js';

/** The fields a preset fills, and what the room will currently accept. */
export interface PresetTarget {
  prompt: SharedDraft<string>;
  negative: SharedDraft<string>;
  denoise: SharedDraft<number>;
  /** The room's ceiling, which the backend may have lowered. */
  maxDenoise: number;
  /** Profiles the running backend actually has. */
  profiles: readonly AIProfileName[];
  send(profile: AIProfileName): void;
}

/**
 * The room's denoise ceiling, as a value the 0.05 slider can actually hold.
 *
 * A backend may cap denoise anywhere - INPROC_MAX_DENOISE=0.83 is a perfectly
 * ordinary setting - and the server rounds whatever it is sent onto the grid
 * before comparing it with that cap. So the ceiling itself has to be floored
 * onto the grid here, or the two disagree: 0.83 sent back is rounded to 0.85
 * and refused, and 0.82 is accepted as 0.80 while this side waits for an echo
 * of 0.82 that never comes.
 */
export function denoiseCeiling(maxDenoise: number): number {
  const cap = Math.min(MAX_DENOISE, maxDenoise > 0 ? maxDenoise : MAX_DENOISE);
  // The epsilon is not decoration: 0.95 / 0.05 is 18.999999999999996, and
  // flooring that would quietly lower every room's ceiling to 0.9.
  const steps = Math.floor(cap / DENOISE_STEP + 1e-9);
  return Math.max(MIN_DENOISE, Math.round(steps * DENOISE_STEP * 100) / 100);
}

/**
 * A preset's denoise, as this room can express it.
 *
 * The slider is a 0.05 grid between 0.2 and 0.95, and the backend may cap it
 * lower still; sending a value off the grid would be accepted, then shown as a
 * slider position nobody can reproduce.
 */
export function clampPresetDenoise(value: number, maxDenoise: number): number {
  const stepped = Math.round(value / DENOISE_STEP) * DENOISE_STEP;
  const bounded = Math.min(denoiseCeiling(maxDenoise), Math.max(MIN_DENOISE, stepped));
  // 0.65 must not arrive as 0.6500000000000001.
  return Math.round(bounded * 100) / 100;
}

/**
 * Applying a prompt preset.
 *
 * A preset is a one-shot fill, not a setting: it writes the fields and is then
 * forgotten. Everything it touches is sent immediately rather than after the
 * usual debounce - picking from a list is a finished decision, and waiting half
 * a second to show the other players what you chose only looks broken.
 *
 * Denoise and profile are recommendations, not part of the look: a preset that
 * carries them says "this style wants a lighter touch" or "this one needs the
 * slower sampler". A profile the backend does not have is simply not sent.
 */
export function applyPreset(preset: PromptPreset, target: PresetTarget): void {
  target.prompt.set(preset.prompt);
  target.prompt.flush();
  target.negative.set(preset.negative);
  target.negative.flush();

  if (preset.denoise !== undefined) {
    target.denoise.set(clampPresetDenoise(preset.denoise, target.maxDenoise));
    target.denoise.flush();
  }
  if (preset.profile !== undefined && target.profiles.includes(preset.profile)) {
    target.send(preset.profile);
  }
}

export function applyPromptPreset(id: string, target: PresetTarget): boolean {
  const preset = PROMPT_PRESETS.find((p) => p.id === id);
  if (!preset) return false;
  applyPreset(preset, target);
  return true;
}

/**
 * The picker's change handler: apply, then go back to blank. The select shows
 * what you may apply next, never what the room is set to - the room's prompt is
 * the prompt field beside it, which stays editable.
 */
export function onPresetChange(event: { target: { value: string } }, target: PresetTarget): void {
  applyPromptPreset(event.target.value, target);
  event.target.value = '';
}

/**
 * The dice. Never lands on R18: that one is a decision, not a surprise, and
 * nobody wants it arriving in a room of strangers because someone was curious
 * what the button did.
 */
export function applyRandomPreset(target: PresetTarget, pick = Math.random): PromptPreset {
  const choices = randomPresets();
  const preset = choices[Math.min(choices.length - 1, Math.floor(pick() * choices.length))]!;
  applyPreset(preset, target);
  return preset;
}
