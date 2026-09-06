import { DEFAULT_NEGATIVE_PROMPT } from './constants.js';

/**
 * A starting point for the room prompt.
 *
 * Presets only *fill the fields*: they are not room state, nothing on the
 * server knows about them, and the moment one is applied it is an ordinary
 * prompt that anyone can edit. A blank `negative` means "leave the built-in
 * default alone".
 *
 * The tags are Danbooru-style because the checkpoint this was built against is
 * Illustrious-class, which is trained on them; on a photographic SDXL model
 * they still read as a sensible description.
 */
export interface PromptPreset {
  id: string;
  /** Japanese / English, in that order - the room is played in both. */
  label: string;
  prompt: string;
  /** Empty means the server's default negative list is used unchanged. */
  negative: string;
}

/**
 * Tags that keep the R18 preset away from anything resembling a minor.
 *
 * They go in front of the default negative list rather than replacing it.
 */
export const MINOR_EXCLUSION_NEGATIVE =
  'child, loli, shota, underage, teen, small body, flat chest, school uniform, lowres, bad anatomy, bad hands, watermark, text';

export const PROMPT_PRESETS: PromptPreset[] = [
  {
    id: 'bishoujo',
    label: '美少女 / anime girl',
    prompt:
      'masterpiece, best quality, 1girl, solo, beautiful detailed eyes, anime style, vibrant colors, detailed background',
    negative: '',
  },
  {
    id: 'landscape',
    label: '風景画 / landscape',
    prompt:
      'masterpiece, best quality, scenery, landscape, no humans, wide shot, detailed sky, mountains, river, painterly',
    negative: '',
  },
  {
    id: 'impressionist',
    label: '印象派 / impressionist',
    prompt:
      'impressionism, oil painting, visible brush strokes, soft natural light, pastel palette, scenery, no humans',
    negative: '',
  },
  {
    id: 'architecture',
    label: '建物 / architecture',
    prompt:
      'masterpiece, best quality, architecture, building, detailed structure, correct perspective, street, daylight, no humans',
    negative: '',
  },
  {
    id: 'background',
    label: '背景 / background art',
    prompt:
      'masterpiece, best quality, anime background, background art, detailed environment, concept art, wide shot, no humans',
    negative: '',
  },
  {
    // Last in the list, and the only one whose negative prompt is set
    // explicitly. "adult woman, mature female" are in the POSITIVE prompt on
    // purpose: on the fast profile the sampler runs at CFG 1.0 and never
    // evaluates the negative branch at all, so the exclusion tags below do
    // nothing there. What steers a distilled 4-step model is what you ask for,
    // not what you ask against - the negative list is the belt for `quality`,
    // and the positive tags are the braces that hold on both.
    id: 'r18',
    label: 'R18 (NSFW)',
    prompt:
      'nsfw, explicit, 1girl, solo, adult woman, mature female, nude, masterpiece, best quality',
    negative: `${MINOR_EXCLUSION_NEGATIVE}, ${DEFAULT_NEGATIVE_PROMPT}`,
  },
];
