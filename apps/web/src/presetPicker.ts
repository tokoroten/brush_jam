import { PROMPT_PRESETS } from '@brushjam/shared';
import type { SharedDraft } from './sharedDraft.js';

/**
 * Applying a prompt preset.
 *
 * A preset is a one-shot fill, not a setting: it writes both prompt fields and
 * is then forgotten. Both are sent immediately rather than after the usual
 * debounce - picking from a list is a finished decision, and waiting half a
 * second to show the other players what you chose only looks broken.
 */
export function applyPromptPreset(
  id: string,
  fields: { prompt: SharedDraft<string>; negative: SharedDraft<string> },
): boolean {
  const preset = PROMPT_PRESETS.find((p) => p.id === id);
  if (!preset) return false;
  fields.prompt.set(preset.prompt);
  fields.prompt.flush();
  fields.negative.set(preset.negative);
  fields.negative.flush();
  return true;
}

/**
 * The picker's change handler: apply, then go back to blank. The select shows
 * what you may apply next, never what the room is set to - the room's prompt
 * is the prompt field beside it, which stays editable.
 */
export function onPresetChange(
  event: { target: { value: string } },
  fields: { prompt: SharedDraft<string>; negative: SharedDraft<string> },
): void {
  applyPromptPreset(event.target.value, fields);
  event.target.value = '';
}
