import { createCanvas, loadImage } from '@napi-rs/canvas';
import { describe, expect, it } from 'vitest';
import { renderStrokes, type Layer, type Stroke } from '@brushjam/shared';
import { MockBackend } from '../src/ai/backends/index.js';
import { addMember, applyClientMessage, captureRenderSnapshot, createRoom, type RoomState } from '../src/room.js';
import { AICanvas, buildMask, renderCropInput } from '../src/raster.js';

const SIZE = 256;
const crop = { x: 0, y: 0, width: 512, height: 512 };
const apply = { x: 64, y: 64, width: 384, height: 384 };

function roomWithStroke(): RoomState {
  const state = createRoom('rasterroom');
  const userId = addMember(state, 'Alice').userId;
  const layerId = state.layers[0]!.id;
  applyClientMessage(state, userId, {
    t: 'stroke_start',
    stroke: { id: 's1', layerId, tool: 'pen', color: '#ff0000', width: 40, points: [{ x: 100, y: 100 }] },
  });
  applyClientMessage(state, userId, { t: 'stroke_end', strokeId: 's1', points: [{ x: 300, y: 300 }] });
  return state;
}

const snapshotOf = (state: RoomState): ReturnType<typeof captureRenderSnapshot> => captureRenderSnapshot(state);

async function nonTransparentPixels(png: Buffer, size: number): Promise<number> {
  const img = await loadImage(png);
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  let count = 0;
  for (let i = 3; i < data.length; i += 4) if (data[i]! > 0) count += 1;
  return count;
}

async function pixelAt(png: Buffer, size: number, x: number, y: number): Promise<number[]> {
  const img = await loadImage(png);
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, size, size);
  return [...ctx.getImageData(x, y, 1, 1).data];
}

describe('renderCropInput', () => {
  it('renders an opaque white background with the stroke on top', async () => {
    const png = await renderCropInput(snapshotOf(roomWithStroke()), crop, SIZE);
    expect(await nonTransparentPixels(png, SIZE)).toBe(SIZE * SIZE);
  });

  it('skips layers excluded from AI input', async () => {
    const state = roomWithStroke();
    state.layers[0]!.includeInAI = false;
    const withLayer = await renderCropInput(snapshotOf(roomWithStroke()), crop, SIZE);
    const without = await renderCropInput(snapshotOf(state), crop, SIZE);
    expect(withLayer.equals(without)).toBe(false);
  });

  it('is deterministic', async () => {
    const snap = snapshotOf(roomWithStroke());
    expect((await renderCropInput(snap, crop, SIZE)).equals(await renderCropInput(snap, crop, SIZE))).toBe(true);
  });
});

/**
 * Finding 5: the server used to draw every layer straight onto the shared white
 * crop, so an eraser cut through the background and lower layers, and repeated
 * strokes on a translucent layer accumulated opacity. The reference below is the
 * browser's algorithm (one transparent canvas per layer, composited once).
 */
describe('layer compositing matches the browser (finding 5)', () => {
  function twoLayerRoom(): RoomState {
    const state = createRoom('composite');
    const userId = addMember(state, 'Alice').userId;
    const lower = state.layers[0]!.id;
    const created = applyClientMessage(state, userId, { t: 'layer_create', layer: { kind: 'draw' } });
    const upper = (created.broadcast[0] as { layer: Layer }).layer.id;
    applyClientMessage(state, userId, { t: 'layer_update', id: upper, patch: { opacity: 0.5 } });

    const stroke = (layerId: string, id: string, tool: 'pen' | 'eraser', color: string, pts: Array<[number, number]>): void => {
      applyClientMessage(state, userId, {
        t: 'stroke_start',
        stroke: { id, layerId, tool, color, width: 60, points: [{ x: pts[0]![0], y: pts[0]![1] }] },
      });
      applyClientMessage(state, userId, { t: 'stroke_end', strokeId: id, points: pts.slice(1).map(([x, y]) => ({ x, y })) });
    };

    stroke(lower, 'l1', 'pen', '#0000ff', [[50, 250], [450, 250]]);
    // two overlapping strokes on the 50% layer, plus an eraser over the lower one
    stroke(upper, 'u1', 'pen', '#ff0000', [[100, 150], [400, 150]]);
    stroke(upper, 'u2', 'pen', '#ff0000', [[100, 170], [400, 170]]);
    stroke(upper, 'u3', 'eraser', '#000000', [[200, 250], [300, 250]]);
    return state;
  }

  /** The browser's compositing path, reimplemented here as the expected result. */
  async function browserReference(state: RoomState, size: number): Promise<Buffer> {
    const out = createCanvas(size, size);
    const octx = out.getContext('2d');
    octx.fillStyle = '#ffffff';
    octx.fillRect(0, 0, size, size);
    const scale = size / crop.width;
    for (const layer of [...state.layers].sort((a, b) => a.order - b.order)) {
      if (!layer.visible || !layer.includeInAI) continue;
      const layerCanvas = createCanvas(size, size);
      const lctx = layerCanvas.getContext('2d');
      lctx.scale(scale, scale);
      const strokes: Stroke[] = state.strokes.filter((s) => s.layerId === layer.id);
      renderStrokes(lctx as unknown as never, strokes, { undone: state.undone, offsetX: crop.x, offsetY: crop.y });
      octx.save();
      octx.globalAlpha = layer.opacity;
      octx.drawImage(layerCanvas, 0, 0);
      octx.restore();
    }
    return out.toBuffer('image/png');
  }

  it('produces the same pixels as the browser algorithm', async () => {
    const state = twoLayerRoom();
    const server = await renderCropInput(snapshotOf(state), crop, SIZE);
    const expected = await browserReference(state, SIZE);
    expect(server.equals(expected)).toBe(true);
  });

  it('keeps the eraser inside its own layer instead of cutting to the background', async () => {
    const state = twoLayerRoom();
    const png = await renderCropInput(snapshotOf(state), crop, SIZE);
    // (125, 125) in output space is (250, 250) in world space: on the lower blue
    // stroke, under the upper layer's eraser. The blue must survive.
    const [r, g, b, a] = await pixelAt(png, SIZE, 125, 125);
    expect(a).toBe(255);
    expect(b!).toBeGreaterThan(200);
    expect(r!).toBeLessThan(80);
    expect(g!).toBeLessThan(80);
  });

  it('applies layer opacity once rather than per overlapping stroke', async () => {
    const state = twoLayerRoom();
    const png = await renderCropInput(snapshotOf(state), crop, SIZE);
    // world (250, 160) sits where the two 50%-layer strokes overlap; a single
    // 50% red over white is ~(255,128,128) and must not darken to ~(255,64,64).
    const [r, g] = await pixelAt(png, SIZE, 125, 80);
    expect(r!).toBeGreaterThan(240);
    expect(g!).toBeGreaterThan(100);
    expect(g!).toBeLessThan(160);
  });

  it('renders nothing for a fully transparent layer', async () => {
    const state = twoLayerRoom();
    for (const layer of state.layers) layer.opacity = 0;
    const png = await renderCropInput(snapshotOf(state), crop, SIZE);
    const [r, g, b] = await pixelAt(png, SIZE, 125, 125);
    expect([r, g, b]).toEqual([255, 255, 255]);
  });
});

describe('buildMask', () => {
  it('is white inside the dirty area and black far away', () => {
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, apply);
    const ctx = built.alpha.getContext('2d');
    expect(ctx.getImageData(115, 115, 1, 1).data[3]!).toBeGreaterThan(200);
    expect(ctx.getImageData(2, 2, 1, 1).data[3]!).toBe(0);
    expect(built.plan.empty).toBe(false);
  });

  it('keeps everything outside the apply area unmasked', () => {
    const small = { x: 192, y: 192, width: 128, height: 128 };
    const built = buildMask([{ x: 0, y: 0, width: 512, height: 512 }], crop, SIZE, small);
    const ctx = built.alpha.getContext('2d');
    expect(ctx.getImageData(5, 5, 1, 1).data[3]!).toBe(0);
    expect(ctx.getImageData(128, 128, 1, 1).data[3]!).toBeGreaterThan(100);
  });

  it('follows an apply rect shifted to a canvas edge', () => {
    const edgeApply = { x: 0, y: 0, width: 384, height: 384 };
    const built = buildMask([{ x: 10, y: 10, width: 40, height: 40 }], crop, SIZE, edgeApply);
    const ctx = built.alpha.getContext('2d');
    expect(built.plan.empty).toBe(false);
    expect(ctx.getImageData(20, 20, 1, 1).data[3]!).toBeGreaterThan(0);
    expect(ctx.getImageData(250, 250, 1, 1).data[3]!).toBe(0);
  });

  it('reports an empty plan when nothing is inside the crop', () => {
    expect(buildMask([{ x: 3000, y: 3000, width: 10, height: 10 }], crop, SIZE, apply).plan.empty).toBe(true);
  });
});

describe('AICanvas', () => {
  it('composites a patch through the mask and returns the crop', async () => {
    const ai = new AICanvas(1024);
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, apply);
    const patch = createCanvas(SIZE, SIZE);
    const pctx = patch.getContext('2d');
    pctx.fillStyle = '#00ff00';
    pctx.fillRect(0, 0, SIZE, SIZE);

    const out = await ai.composite(patch.toBuffer('image/png'), crop, built.alpha);
    const painted = await nonTransparentPixels(out, SIZE);
    expect(painted).toBeGreaterThan(0);
    expect(painted).toBeLessThan(SIZE * SIZE);
  });
});

describe('MockBackend', () => {
  it('returns a same-size PNG and changes pixels only inside the mask', async () => {
    const imagePng = await renderCropInput(snapshotOf(roomWithStroke()), crop, SIZE);
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, apply);
    const backend = new MockBackend(0);
    const req = {
      prompt: 'a town',
      negativePrompt: '',
      imagePng,
      maskPng: built.png,
      size: SIZE,
      denoise: 0.5,
      steps: 8,
      seed: 1,
      tag: 't',
    };
    const out = await backend.generate(req, new AbortController().signal);
    expect(out.equals(imagePng)).toBe(false);
    expect(await nonTransparentPixels(out, SIZE)).toBe(SIZE * SIZE);
    expect((await backend.generate(req, new AbortController().signal)).equals(out)).toBe(true);
  });

  it('honours the abort signal', async () => {
    const backend = new MockBackend(500);
    const controller = new AbortController();
    const imagePng = await renderCropInput(snapshotOf(roomWithStroke()), crop, SIZE);
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, apply);
    const promise = backend.generate(
      { prompt: 'p', negativePrompt: '', imagePng, maskPng: built.png, size: SIZE, denoise: 0.5, steps: 8, seed: 1, tag: 't' },
      controller.signal,
    );
    controller.abort();
    await expect(promise).rejects.toThrow(/aborted/);
  });
});
