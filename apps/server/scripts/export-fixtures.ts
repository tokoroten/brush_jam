/**
 * Cross-language parity fixtures for the Python port (docs/PYTHON_SERVER_PLAN.md
 * section 4). This runs the *Node* implementation and writes down exactly what
 * it does; `apps/brushjam/tests/test_fixtures.py` replays the same input against
 * the Python one and asserts the same output. It is the parity proof, so it
 * imports the real modules and never re-implements anything.
 *
 *   pnpm --filter @brushjam/server export-fixtures
 *
 * Random ids (users, layers, strokes) are replaced by stable placeholders with
 * the identical rule on both sides, since neither implementation can be asked
 * to produce the other's random bytes.
 */
import { createCanvas } from '@napi-rs/canvas';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fnv1a, noiseRGB, renderStrokes, type ClientMessage } from '@brushjam/shared';
import { validateClientMessage } from '../src/validate.js';
import {
  applyClientMessage,
  createRoom,
  joinMember,
  snapshot,
  type ApplyResult,
  type RoomState,
} from '../src/room.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(here, '..', '..', 'brushjam', 'fixtures');

// ---------------------------------------------------------------- protocol

/** One valid and one invalid sample per client message type. */
const PROTOCOL_SAMPLES: { type: string; valid: unknown[]; invalid: unknown[] }[] = [
  {
    type: 'cursor',
    valid: [{ t: 'cursor', x: 12.5, y: -3 }],
    invalid: [{ t: 'cursor', x: 'a', y: 1 }, { t: 'cursor', x: 1 }],
  },
  {
    type: 'stroke_start',
    valid: [
      { t: 'stroke_start', stroke: { id: 's1', layerId: 'l1', tool: 'pen', color: '#112233', width: 14, points: [{ x: 1, y: 2 }] } },
      { t: 'stroke_start', stroke: { id: 's2', layerId: 'l1', tool: 'noise', color: '#000000', width: 64, alpha: 0.5, points: [{ x: 1, y: 2, p: 0.4 }] } },
      { t: 'stroke_start', stroke: { id: 's3', layerId: 'l1', tool: 'eraser', color: '#000000', width: 32, alpha: 9, points: [] } },
    ],
    invalid: [
      { t: 'stroke_start' },
      { t: 'stroke_start', stroke: { id: '', layerId: 'l1', tool: 'pen', color: '#000000', width: 3, points: [] } },
      { t: 'stroke_start', stroke: { id: 's1', layerId: 'l1', tool: 'brush', color: '#000000', width: 3, points: [] } },
      { t: 'stroke_start', stroke: { id: 's1', layerId: 'l1', tool: 'pen', color: '#000000', width: 3, alpha: 'x', points: [] } },
      { t: 'stroke_start', stroke: { id: 's1', layerId: 'l1', tool: 'pen', color: '#000000', width: 3, points: [{ x: 1 }] } },
    ],
  },
  {
    type: 'stroke_chunk',
    valid: [{ t: 'stroke_chunk', strokeId: 's1', points: [{ x: 1, y: 2 }, { x: 3, y: 4, p: 1 }] }],
    invalid: [{ t: 'stroke_chunk', points: [] }, { t: 'stroke_chunk', strokeId: 's1', points: 'no' }],
  },
  {
    type: 'stroke_end',
    valid: [{ t: 'stroke_end', strokeId: 's1', points: [] }],
    invalid: [{ t: 'stroke_end', strokeId: '', points: [] }],
  },
  { type: 'undo', valid: [{ t: 'undo' }], invalid: [] },
  {
    type: 'clear_layer',
    valid: [{ t: 'clear_layer', layerId: 'l1' }],
    invalid: [{ t: 'clear_layer' }],
  },
  {
    type: 'layer_create',
    valid: [
      { t: 'layer_create', layer: { kind: 'draw' } },
      { t: 'layer_create', layer: { kind: 'reference', imageId: 'img1', x: 5, y: 6, scale: 2, name: 'Ref' } },
    ],
    invalid: [
      { t: 'layer_create', layer: { kind: 'sketch' } },
      { t: 'layer_create', layer: { kind: 'draw', opacity: 'half' } },
      { t: 'layer_create' },
    ],
  },
  {
    type: 'layer_update',
    valid: [
      { t: 'layer_update', id: 'l1', patch: { visible: false, opacity: 0.5, offsetX: 10, unknown: 3 } },
      { t: 'layer_update', id: 'l1', patch: {} },
    ],
    invalid: [
      { t: 'layer_update', id: 'l1', patch: { visible: 'no' } },
      { t: 'layer_update', patch: {} },
    ],
  },
  {
    type: 'layer_delete',
    valid: [{ t: 'layer_delete', id: 'l1' }],
    invalid: [{ t: 'layer_delete', id: 3 }],
  },
  {
    type: 'layer_reorder',
    valid: [{ t: 'layer_reorder', ids: ['a', 'b'] }],
    invalid: [{ t: 'layer_reorder', ids: [] }, { t: 'layer_reorder', ids: ['a', 'a'] }],
  },
  {
    type: 'set_prompt',
    valid: [{ t: 'set_prompt', prompt: 'a town' }],
    invalid: [{ t: 'set_prompt', prompt: 7 }],
  },
  {
    type: 'set_ai_settings',
    valid: [
      { t: 'set_ai_settings', denoise: 0.8 },
      { t: 'set_ai_settings', aiResolution: 768, aiProfile: 'quality', negativePrompt: 'blurry' },
    ],
    invalid: [
      { t: 'set_ai_settings' },
      { t: 'set_ai_settings', denoise: 0.99 },
      { t: 'set_ai_settings', aiResolution: 700 },
      { t: 'set_ai_settings', aiProfile: 'turbo' },
    ],
  },
  {
    type: 'unknown',
    valid: [],
    invalid: [{ t: 'nope' }, { t: 5 }, 'string', [1, 2]],
  },
];

function protocolFixture(): unknown {
  return {
    note: 'Each sample is fed to validateClientMessage; `result` is what the Node validator returned.',
    samples: PROTOCOL_SAMPLES.flatMap((group) =>
      [...group.valid, ...group.invalid].map((input) => {
        const result = validateClientMessage(input);
        return {
          type: group.type,
          input,
          result: result.ok ? { ok: true, msg: result.msg } : { ok: false, error: result.error },
        };
      }),
    ),
  };
}

// ------------------------------------------------------------------- noise

function noiseFixture(): unknown {
  const seeds = ['', 'a', 'stroke-1', 'u9k2:sim1abc', 'ある', '0123456789abcdef'];
  const coords: [number, number][] = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1023, 1023],
    [-5, 7],
    [4096, -4096],
    [123456, 654321],
  ];
  return {
    fnv1a: seeds.map((text) => ({ text, seed: fnv1a(text) })),
    rgb: seeds.flatMap((text) => {
      const seed = fnv1a(text);
      return coords.map(([x, y]) => ({ text, seed, x, y, rgb: noiseRGB(seed, x, y) }));
    }),
  };
}

// ---------------------------------------------------------- noise placement

/**
 * Where a noise stroke's hash is addressed from, taken from the REAL shared
 * renderer rather than transcribed.
 *
 * `renderStrokes` is driven with a canvas stub that reports every pixel as
 * covered, so the noise loop runs over the whole temp raster and the first
 * pixel it writes is `noiseHash(seed, worldX, worldY)` - the origin this
 * fixture exists to pin down. Coverage and antialiasing are not the subject
 * and are deliberately faked; the origin is not.
 *
 * Fractional layer offsets are the interesting case: the renderer keeps the
 * temp origin unrounded and applies `Math.round` to it, so a server that
 * floors first hashes from the previous world pixel.
 */
function noisePlacementFixture(): unknown {
  interface Recorded {
    left: number;
    top: number;
    width: number;
    height: number;
    firstPixel: [number, number, number];
  }

  // The temp canvas the noise pen creates is the one whose pixels matter; the
  // target only reports where it was drawn.
  let tempData: number[] = [];

  const makeCtx = (width: number, height: number, record: null | ((r: Recorded) => void)): any => ({
    save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {}, arc() {},
    stroke() {}, fill() {},
    globalCompositeOperation: 'source-over',
    globalAlpha: 1,
    strokeStyle: '', fillStyle: '', lineWidth: 1, lineCap: '', lineJoin: '',
    canvas: { width, height },
    getImageData(_x: number, _y: number, w: number, h: number) {
      // Everything covered: the noise loop skips fully transparent pixels, and
      // what is being captured is which world pixel it starts from.
      tempData = new Array(w * h * 4).fill(0);
      for (let i = 3; i < tempData.length; i += 4) tempData[i] = 255;
      return { data: tempData };
    },
    putImageData() {},
    drawImage(_image: never, dx: number, dy: number) {
      if (record === null) return;
      record({
        left: dx,
        top: dy,
        width,
        height,
        firstPixel: [tempData[0] ?? 0, tempData[1] ?? 0, tempData[2] ?? 0],
      });
    },
  });

  const cases: unknown[] = [];
  const offsets = [0, 0.5, -0.5, 0.25, -0.25, 1, 2.5, -3.5];
  const strokes = [
    { id: 'noise-a', tool: 'noise' as const, color: '#000000', width: 12, points: [{ x: 40, y: 40 }, { x: 90, y: 70 }] },
    { id: 'noise-b', tool: 'noise' as const, color: '#000000', width: 7, points: [{ x: 41.5, y: 40.5 }, { x: 61.5, y: 80.5 }] },
    { id: 'ある', tool: 'noise' as const, color: '#000000', width: 24, points: [{ x: 10, y: 10 }, { x: 200, y: 190 }] },
    // Starts off the top-left corner, so the temp box is clamped to -pad and
    // the origin becomes fractional under a fractional offset. That is the
    // only way to reach an exact .5, where Math.round and Python's
    // ties-to-even round() disagree.
    { id: 'noise-clipped', tool: 'noise' as const, color: '#000000', width: 7, points: [{ x: -30, y: -30 }, { x: 20, y: 25 }] },
  ];
  const bounds = { width: 256, height: 256 };

  for (const stroke of strokes) {
    for (const offsetX of offsets) {
      for (const offsetY of offsets) {
        let recorded: Recorded | null = null;
        const target = makeCtx(bounds.width, bounds.height, (r) => (recorded = r));
        renderStrokes(target as never, [stroke], {
          offsetX,
          offsetY,
          bounds,
          createCanvas: (w: number, h: number) => ({
            width: w,
            height: h,
            getContext: () => makeCtx(w, h, null),
          } as never),
        });
        if (recorded === null) continue;
        const r = recorded as Recorded;
        cases.push({
          strokeId: stroke.id,
          seed: fnv1a(stroke.id),
          width: stroke.width,
          points: stroke.points,
          offsetX,
          offsetY,
          bounds,
          // What the renderer actually used, unrounded.
          logicalLeft: r.left,
          logicalTop: r.top,
          tempWidth: r.width,
          tempHeight: r.height,
          // noiseHash at the temp origin, i.e. the world pixel the stroke's
          // noise starts from. This is the value a port must reproduce.
          originRGB: r.firstPixel,
        });
      }
    }
  }
  return { cases, pixels: noisePixelCases() };
}

// ----------------------------------------------------------------- reducer

/**
 * Random ids cannot match across implementations, so both sides rewrite them
 * with the same rule: members in join order, layers in creation order, and a
 * stroke id is `<userId>:<raw>` so it follows its author.
 */
function idMap(room: RoomState, users: string[]): Map<string, string> {
  const map = new Map<string, string>();
  users.forEach((userId, i) => map.set(userId, `U${i}`));
  [...room.layers]
    .sort((a, b) => a.order - b.order)
    .forEach((layer, i) => {
      if (!map.has(layer.id)) map.set(layer.id, `L${i}`);
    });
  return map;
}

function normalise(value: unknown, map: Map<string, string>): unknown {
  if (typeof value === 'string') {
    let out = value;
    for (const [id, placeholder] of map) out = out.split(id).join(placeholder);
    return out;
  }
  if (Array.isArray(value)) return value.map((v) => normalise(v, map));
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = normalise(v, map);
    return out;
  }
  return value;
}

interface Step {
  /** Index into the user list. */
  by: number;
  msg: unknown;
}

const STEPS: Step[] = [
  { by: 0, msg: { t: 'cursor', x: 10, y: 20 } },
  { by: 0, msg: { t: 'stroke_start', stroke: { id: 'a1', layerId: '@layer0', tool: 'pen', color: '#ff0000', width: 12, points: [{ x: 10, y: 10 }] } } },
  { by: 0, msg: { t: 'stroke_chunk', strokeId: 'a1', points: [{ x: 20, y: 20 }, { x: 30, y: 25, p: 0.5 }] } },
  { by: 0, msg: { t: 'stroke_end', strokeId: 'a1', points: [{ x: 40, y: 30 }] } },
  { by: 1, msg: { t: 'stroke_start', stroke: { id: 'b1', layerId: '@layer0', tool: 'noise', color: '#00ff00', width: 64, alpha: 0.4, points: [{ x: 100, y: 100 }] } } },
  { by: 1, msg: { t: 'stroke_end', strokeId: 'b1', points: [{ x: 140, y: 160 }] } },
  { by: 0, msg: { t: 'stroke_start', stroke: { id: 'a2', layerId: '@layer0', tool: 'pen', color: 'red', width: 500, points: [{ x: 200, y: 200 }] } } },
  { by: 0, msg: { t: 'stroke_end', strokeId: 'a2', points: [{ x: 210, y: 220 }] } },
  // per-user undo: Alice's undo takes her own latest stroke, not Bob's
  { by: 0, msg: { t: 'undo' } },
  { by: 1, msg: { t: 'undo' } },
  { by: 1, msg: { t: 'undo' } },
  { by: 0, msg: { t: 'layer_create', layer: { kind: 'draw', name: 'Second' } } },
  { by: 0, msg: { t: 'layer_update', id: '@layer1', patch: { opacity: 0.25, visible: false } } },
  { by: 0, msg: { t: 'layer_update', id: '@layer0', patch: { offsetX: 40, offsetY: -10 } } },
  { by: 0, msg: { t: 'layer_update', id: '@layer1', patch: { locked: true } } },
  { by: 0, msg: { t: 'layer_update', id: '@layer1', patch: { offsetX: 5 } } },
  { by: 0, msg: { t: 'layer_reorder', ids: ['@layer1', '@layer0'] } },
  { by: 0, msg: { t: 'layer_delete', id: '@layer1' } },
  { by: 0, msg: { t: 'layer_delete', id: '@layer0' } },
  { by: 1, msg: { t: 'clear_layer', layerId: '@layer0' } },
  { by: 1, msg: { t: 'clear_layer', layerId: 'nope42' } },
  { by: 0, msg: { t: 'set_prompt', prompt: 'a quiet harbour' } },
  { by: 0, msg: { t: 'set_prompt', prompt: 'a quiet harbour' } },
  { by: 1, msg: { t: 'set_ai_settings', denoise: 0.85 } },
  { by: 1, msg: { t: 'set_ai_settings', aiProfile: 'quality' } },
  { by: 1, msg: { t: 'set_ai_settings', aiResolution: 512, negativePrompt: 'text, watermark' } },
  { by: 1, msg: { t: 'set_ai_settings', aiResolution: 2048 } },
];

function reducerFixture(): unknown {
  const room = createRoom('trace01', 0.7, 1024, 768, true, 'fast', {
    profiles: ['fast', 'quality'],
    maxDenoise: 0.95,
    maxResolution: 1024,
    negativePromptActive: { fast: true, quality: true },
  });
  const alice = joinMember(room, 'Alice', 'tok-alice-0001');
  const bob = joinMember(room, 'Bob', 'tok-bob-0001');
  const users = [alice.userId, bob.userId];
  // Placeholders in the script are resolved against the layers as they exist
  // when the step runs, in creation order.
  const layerIds = (): string[] => [...room.layers].sort((a, b) => a.order - b.order).map((l) => l.id);
  const created: string[] = [...layerIds()];

  const steps = STEPS.map((step) => {
    for (const id of layerIds()) if (!created.includes(id)) created.push(id);
    const resolved = JSON.parse(
      JSON.stringify(step.msg).replace(/"@layer(\d+)"/g, (_m, n: string) => JSON.stringify(created[Number(n)] ?? 'missing')),
    ) as unknown;
    const validated = validateClientMessage(resolved);
    const map = idMap(room, users);
    for (const [i, id] of created.entries()) map.set(id, `L${i}`);
    if (!validated.ok) {
      return { by: step.by, msg: normalise(step.msg, map), rejected: validated.error };
    }
    const result: ApplyResult = applyClientMessage(room, users[step.by]!, validated.msg as ClientMessage);
    const after = snapshot(room, users[step.by]!, 'idle', { window: 768, apply: 1024, canvasSize: 1024 });
    const map2 = idMap(room, users);
    for (const id of layerIds()) if (!created.includes(id)) created.push(id);
    for (const [i, id] of created.entries()) map2.set(id, `L${i}`);
    return {
      by: step.by,
      msg: normalise(step.msg, map2),
      result: normalise(
        {
          broadcast: result.broadcast,
          relay: result.relay,
          toSender: result.toSender ?? [],
          dirty: result.dirty,
          promptChanged: result.promptChanged ?? false,
        },
        map2,
      ),
      snapshot: normalise(after, map2),
    };
  });

  return {
    note: 'createRoom("trace01", 0.7, 1024, 768, true, "fast", {fast+quality, 0.95, 1024}); Alice then Bob join.',
    room: { id: 'trace01', denoise: 0.7, canvasSize: 1024, resolution: 768, adjustable: true, profile: 'fast' },
    users: ['Alice', 'Bob'],
    steps,
  };
}

mkdirSync(OUT_DIR, { recursive: true });

/**
 * Actual pixels, from a real canvas.
 *
 * The stub above records where the noise pen starts hashing; it cannot say
 * what the target ends up holding, and that is where the second half of the
 * bug lived: the shared renderer draws its temp canvas at a *fractional*
 * origin, so the canvas resamples it and every interior pixel of a noise
 * stroke becomes a blend of neighbouring random values. A port that composites
 * at the floored origin gets a completely different colour in each one.
 */
function noisePixelCases(): unknown[] {
  const bounds = { width: 256, height: 256 };
  const specs = [
    { id: 'u:s1', tool: 'noise' as const, width: 40, alpha: 1, points: [{ x: 100, y: 100 }, { x: 160, y: 100 }], offsetX: -0.5, offsetY: 0 },
    { id: 'u:s2', tool: 'noise' as const, width: 30, alpha: 1, points: [{ x: 60, y: 60 }, { x: 180, y: 150 }], offsetX: 0.25, offsetY: -0.75 },
    { id: 'u:s3', tool: 'noise' as const, width: 24, alpha: 1, points: [{ x: 40, y: 200 }, { x: 200, y: 190 }], offsetX: 0, offsetY: 0 },
    { id: 'u:s4', tool: 'pen' as const, color: '#204080', width: 36, alpha: 0.5, points: [{ x: 50, y: 50 }, { x: 200, y: 120 }], offsetX: -0.5, offsetY: 0.5 },
    { id: 'u:s5', tool: 'pen' as const, color: '#d02010', width: 36, alpha: 1, points: [{ x: 40, y: 120 }, { x: 210, y: 130 }], offsetX: 0.3, offsetY: -0.2 },
  ];

  const out: unknown[] = [];
  for (const spec of specs) {
    const canvas = createCanvas(bounds.width, bounds.height);
    const ctx = canvas.getContext('2d');
    const stroke = {
      id: spec.id,
      tool: spec.tool,
      color: (spec as { color?: string }).color ?? '#000000',
      width: spec.width,
      alpha: spec.alpha,
      points: spec.points,
    };
    renderStrokes(ctx as never, [stroke as never], {
      offsetX: spec.offsetX,
      offsetY: spec.offsetY,
      bounds,
      createCanvas: (w: number, h: number) => createCanvas(w, h) as never,
    });
    const data = ctx.getImageData(0, 0, bounds.width, bounds.height).data;
    // Only pixels the stroke covers completely: an antialiased edge is a
    // coverage difference between two rasterisers, which parity never claimed.
    // The alpha a fully covered pixel ends up with: a translucent stroke is
    // composited once, at its own opacity.
    const solid = Math.round(spec.alpha * 255);
    const samples: unknown[] = [];
    for (let y = 0; y < bounds.height; y++) {
      for (let x = 0; x < bounds.width; x++) {
        const i = (y * bounds.width + x) * 4;
        if (data[i + 3] !== solid) continue;
        if ((y * bounds.width + x) % 97 !== 0) continue;
        samples.push({ x, y, rgb: [data[i], data[i + 1], data[i + 2]] });
      }
    }
    out.push({ stroke, offsetX: spec.offsetX, offsetY: spec.offsetY, bounds, alpha8: solid, samples });
  }
  return out;
}

const files: [string, unknown][] = [
  ['protocol-samples.json', protocolFixture()],
  ['noise-samples.json', noiseFixture()],
  ['noise-placement.json', noisePlacementFixture()],
  ['reducer-trace.json', reducerFixture()],
];
for (const [name, body] of files) {
  const file = path.join(OUT_DIR, name);
  writeFileSync(file, `${JSON.stringify(body, null, 2)}\n`);
  console.log(`[fixtures] wrote ${file}`);
}
