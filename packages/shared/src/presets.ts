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
 *
 * ## A style is not a thing
 *
 * A tag-trained model reads a noun as an object to draw. The first version of
 * these presets said "ink wash painting ... black ink on white paper" and got
 * back a picture *of* a brush and a sheet of paper; "children's book
 * illustration" got a book, "cathedral window" got a window, "clay figures"
 * got figurines on a table. The players' drawing was replaced by a still life
 * of the medium (docs/experiments/2026-09-07-presets/REPORT.md).
 *
 * So, for anything in 画風 / style or 雰囲気 / mood:
 *
 * - name the medium as a *quality* - `X style`, `(medium)`, adjectives - and
 *   never as an object. No brush, paper, canvas, book, pages, window, frame,
 *   easel, tools, hands, typography, annotations, diorama, sprite sheet.
 * - no scene nouns. A style has to combine with whatever the players drew, so
 *   "waves, mount fuji" or "miniature set" is the preset drawing its own
 *   picture over theirs.
 * - no "masterpiece, best quality": on an Illustrious-class model that pair
 *   pulls hard towards polished character art, which is a subject, not a look.
 * - a lower denoise than a subject preset. A style is a way of drawing what is
 *   already there (0.6-0.65); a subject is permission to invent (0.8-0.85).
 * - a negative that names the objects of the medium anyway. It does nothing on
 *   `fast`, where CFG 1.0 never evaluates the negative branch, and it is worth
 *   having on `quality`. Those negatives are written as
 *   `objects, ${DEFAULT_NEGATIVE_PROMPT}` so the ordinary quality tags stay.
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
   * A denoise that suits this look, applied with the prompt. A style wants a
   * light touch (the drawing IS the picture, 0.6-0.65); a subject wants nearly
   * all of it (the drawing is only a hint of where things go, 0.8-0.85).
   * Clamped to whatever the room and the backend allow.
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
      'impressionism, oil painting (medium), visible brushwork, soft natural light, pastel palette, dappled light',
    negative: `easel, paintbrush, palette, canvas, picture frame, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.6,
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
    // "black ink on white paper" used to produce ink and paper, as objects.
    prompt:
      'sumi-e style, ink wash style, monochrome, greyscale, traditional media, ink (medium), rough brushwork, minimalist, negative space, white background',
    negative: `paintbrush, brush, ink bottle, paper, calligraphy, hands, text, ${DEFAULT_NEGATIVE_PROMPT}`,
    // The drawing is most of the picture here, so the model is given least room.
    denoise: 0.6,
  },
  {
    id: 'ukiyo-e',
    label: '浮世絵 / ukiyo-e',
    group: '画風 / style',
    // "waves, mount fuji" drew the famous print instead of the room's drawing.
    prompt:
      'ukiyo-e style, woodblock print style, flat color, bold outlines, limited palette, traditional japanese art style',
    negative: `paper, seal, signature, text, picture frame, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.65,
  },
  {
    id: 'stained-glass',
    label: 'ステンドグラス / stained glass',
    group: '画風 / style',
    // The name is itself an object, and naming it at all - even as "stained
    // glass style" - produced a picture OF a window. So it is described only
    // by what it looks like, and the window is in the negative.
    prompt:
      'thick black outlines, translucent jewel-tone color cells, backlit glow, mosaic of flat color segments, luminous, high contrast',
    negative: `window, cathedral, church, picture frame, lattice, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.65,
  },
  {
    id: 'pixel-art',
    label: 'ドット絵 / pixel art',
    group: '画風 / style',
    prompt:
      'pixel art, 16-bit style, retro game aesthetic, limited palette, dithering, crisp pixels',
    negative: `sprite sheet, grid, user interface, text, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.65,
  },
  {
    id: 'watercolor-book',
    label: '水彩絵本 / watercolor picture book',
    group: '画風 / style',
    // "children's book illustration, storybook" drew books.
    prompt:
      'watercolor (medium), traditional media, storybook illustration style, soft edges, paper texture, pastel colors, whimsical',
    negative: `book, open book, pages, text, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.6,
  },
  {
    id: 'claymation',
    label: '粘土アニメ / claymation',
    group: '画風 / style',
    // "clay figures, miniature set" built a set with figurines on it.
    prompt:
      'claymation style, clay art style, plasticine texture, stop motion aesthetic, soft studio lighting, matte finish',
    negative: `figurine, doll, miniature set, table, hands, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.65,
  },
  {
    id: 'papercraft',
    label: '切り絵 / papercraft',
    group: '画風 / style',
    // "diorama" built a scene in a box.
    prompt:
      'paper cutout style, layered paper art style, kirigami style, soft drop shadows, flat shapes',
    negative: `diorama, scissors, craft table, hands, box, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.65,
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
    // "annotations" is a request for text, and the text a model writes is
    // never text.
    prompt:
      'mecha, blueprint style, technical drawing style, schematic line art, cyan background, mechanical details',
    negative: `text, handwriting, watermark, ${DEFAULT_NEGATIVE_PROMPT}`,
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
    // "bold typography, advertisement" drew a poster, with lettering on it.
    prompt:
      'showa retro style, vintage poster style, faded colors, halftone, muted palette, nostalgic',
    negative: `text, letters, logo, watermark, poster on a wall, ${DEFAULT_NEGATIVE_PROMPT}`,
    denoise: 0.65,
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
