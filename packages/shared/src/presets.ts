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
  /** Which optgroup it appears under. */
  group: PresetGroup;
  prompt: string;
  /** Empty means the server's default negative list is used unchanged. */
  negative: string;
  /**
   * A denoise that suits this look, applied with the prompt. Ink wash wants a
   * light touch (the drawing IS the picture); a nebula wants nearly all of it
   * (the drawing is only a hint of where things go). Clamped to whatever the
   * room and the backend allow.
   */
  denoise?: number;
  /** Only when the look genuinely needs the slower sampler. */
  profile?: 'fast' | 'quality';
}

export type PresetGroup = '基本 / basic' | '画風 / style' | '題材 / subject' | '雰囲気 / mood' | 'R18';

/** Group order in the picker. R18 is last, on its own. */
export const PRESET_GROUPS: PresetGroup[] = [
  '基本 / basic',
  '画風 / style',
  '題材 / subject',
  '雰囲気 / mood',
  'R18',
];

/**
 * Tags that keep the R18 preset away from anything resembling a minor.
 *
 * They go in front of the default negative list rather than replacing it.
 */
export const MINOR_EXCLUSION_NEGATIVE =
  'child, loli, shota, underage, teen, small body, flat chest, school uniform, lowres, bad anatomy, bad hands, watermark, text';

export const PROMPT_PRESETS: PromptPreset[] = [
  // ------------------------------------------------------------ 基本 / basic
  {
    id: 'bishoujo',
    label: '美少女 / anime girl',
    group: '基本 / basic',
    prompt:
      'masterpiece, best quality, 1girl, solo, beautiful detailed eyes, anime style, vibrant colors, detailed background',
    negative: '',
  },
  {
    id: 'landscape',
    label: '風景画 / landscape',
    group: '基本 / basic',
    prompt:
      'masterpiece, best quality, scenery, landscape, no humans, wide shot, detailed sky, mountains, river, painterly',
    negative: '',
  },
  {
    id: 'impressionist',
    label: '印象派 / impressionist',
    group: '基本 / basic',
    prompt:
      'impressionism, oil painting, visible brush strokes, soft natural light, pastel palette, scenery, no humans',
    negative: '',
  },
  {
    id: 'architecture',
    label: '建物 / architecture',
    group: '基本 / basic',
    prompt:
      'masterpiece, best quality, architecture, building, detailed structure, correct perspective, street, daylight, no humans',
    negative: '',
  },
  {
    id: 'background',
    label: '背景 / background art',
    group: '基本 / basic',
    prompt:
      'masterpiece, best quality, anime background, background art, detailed environment, concept art, wide shot, no humans',
    negative: '',
  },

  // ------------------------------------------------------------ 画風 / style
  {
    id: 'sumi-e',
    label: '水墨画 / sumi-e',
    group: '画風 / style',
    prompt:
      'sumi-e, ink wash painting, monochrome, black ink on white paper, visible brush strokes, minimal, negative space, traditional japanese art',
    negative: '',
    // The drawing is most of the picture here, so the model is given least room.
    denoise: 0.65,
  },
  {
    id: 'ukiyo-e',
    label: '浮世絵 / ukiyo-e',
    group: '画風 / style',
    prompt:
      'ukiyo-e, woodblock print, flat colors, bold outlines, traditional japanese, edo period, waves, mount fuji',
    negative: '',
    denoise: 0.75,
  },
  {
    id: 'stained-glass',
    label: 'ステンドグラス / stained glass',
    group: '画風 / style',
    prompt: 'stained glass, lead lines, glowing colored glass, cathedral window, geometric, backlit',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'pixel-art',
    label: 'ドット絵 / pixel art',
    group: '画風 / style',
    prompt: 'pixel art, 16-bit, retro game, limited palette, dithering, sprite sheet style',
    negative: '',
    denoise: 0.75,
  },
  {
    id: 'watercolor-book',
    label: '水彩絵本 / watercolor picture book',
    group: '画風 / style',
    prompt:
      "watercolor, children's book illustration, soft edges, paper texture, warm, whimsical, storybook",
    negative: '',
    denoise: 0.7,
  },
  {
    id: 'claymation',
    label: '粘土アニメ / claymation',
    group: '画風 / style',
    prompt:
      'claymation, clay figures, stop motion, plasticine texture, fingerprints, miniature set, studio lighting',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'papercraft',
    label: '切り絵 / papercraft',
    group: '画風 / style',
    prompt: 'papercraft, paper cutout, layered paper, kirigami, soft shadows, handmade, diorama',
    negative: '',
    denoise: 0.8,
  },

  // ---------------------------------------------------------- 題材 / subject
  {
    id: 'fantasy-map',
    label: 'ファンタジー地図 / fantasy map',
    group: '題材 / subject',
    prompt:
      'fantasy map, cartography, parchment, hand drawn map, rivers, roads, forests, mountains, town icons, compass rose, top-down',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'creature',
    label: 'クリーチャー図鑑 / creature design',
    group: '題材 / subject',
    prompt:
      'creature design, monster, bestiary illustration, fantasy creature, detailed anatomy, concept art, full body, white background',
    negative: '',
    denoise: 0.85,
  },
  {
    id: 'nebula',
    label: '宇宙 / nebula',
    group: '題材 / subject',
    prompt:
      'outer space, nebula, galaxy, stars, cosmic clouds, vivid colors, astrophotography, no humans',
    negative: '',
    // The drawing is a hint about where things go, nothing more.
    denoise: 0.85,
  },
  {
    id: 'food',
    label: '料理 / food photo',
    group: '題材 / subject',
    prompt:
      'food photography, delicious dish on a plate, appetizing, close-up, restaurant lighting, no humans',
    negative: '',
    denoise: 0.85,
  },
  {
    id: 'satellite',
    label: '航空写真 / satellite view',
    group: '題材 / subject',
    prompt:
      'satellite view, aerial photograph, top-down city, roads, rooftops, fields, rivers, no humans',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'mecha-blueprint',
    label: 'メカ設計図 / mecha blueprint',
    group: '題材 / subject',
    prompt:
      'mecha, blueprint, technical drawing, schematic lines, cyan background, annotations, mechanical details',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'botanical',
    label: '植物図鑑 / botanical',
    group: '題材 / subject',
    prompt:
      'botanical illustration, flowers, leaves, scientific illustration, watercolor, white background, no humans',
    negative: '',
    denoise: 0.75,
  },

  // ------------------------------------------------------------- 雰囲気 / mood
  {
    id: 'neon-city',
    label: 'サイバーパンク夜景 / neon city',
    group: '雰囲気 / mood',
    prompt:
      'cyberpunk city at night, neon lights, rain, reflections, skyscrapers, holograms, no humans',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'horror',
    label: 'ホラー / horror',
    group: '雰囲気 / mood',
    prompt: 'horror, dark, eerie, fog, abandoned, creepy atmosphere, dim light, unsettling',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'retro-poster',
    label: '昭和レトロポスター / retro poster',
    group: '雰囲気 / mood',
    prompt:
      'showa retro poster, vintage japanese advertisement, faded colors, halftone, bold typography, 1960s',
    negative: '',
    denoise: 0.8,
  },
  {
    id: 'photoreal',
    label: '写真 / photorealistic',
    group: '雰囲気 / mood',
    prompt: 'photorealistic, photograph, realistic lighting, dslr, high detail, natural colors',
    negative: '',
    denoise: 0.8,
    // The one look a 4-step distilled model cannot fake: photographic detail
    // needs the steps.
    profile: 'quality',
  },

  // ---------------------------------------------------------------------- R18
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
    group: 'R18',
    prompt:
      'nsfw, explicit, 1girl, solo, adult woman, mature female, nude, masterpiece, best quality',
    negative: `${MINOR_EXCLUSION_NEGATIVE}, ${DEFAULT_NEGATIVE_PROMPT}`,
  },
];

/** Everything the dice may land on: the whole list except R18. */
export const randomPresets = (): PromptPreset[] => PROMPT_PRESETS.filter((p) => p.group !== 'R18');
