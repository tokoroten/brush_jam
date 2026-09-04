import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it } from 'vitest';
import { MockBackend } from '../src/ai/backends/index.js';
import { addMember, applyClientMessage, createRoom } from '../src/room.js';
import { AICanvas, buildMask, renderCropInput } from '../src/raster.js';

const SIZE = 256;
const crop = { x: 0, y: 0, width: 512, height: 512 };

function roomWithStroke(): ReturnType<typeof createRoom> {
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

function nonTransparentPixels(png: Buffer, size: number): Promise<number> {
  return import('@napi-rs/canvas').then(async ({ loadImage }) => {
    const img = await loadImage(png);
    const canvas = createCanvas(size, size);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;
    let count = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i]! > 0) count += 1;
    return count;
  });
}

describe('renderCropInput', () => {
  it('renders an opaque white background with the stroke on top', async () => {
    const png = await renderCropInput(roomWithStroke(), crop, SIZE);
    expect(await nonTransparentPixels(png, SIZE)).toBe(SIZE * SIZE);
  });

  it('skips layers excluded from AI input', async () => {
    const state = roomWithStroke();
    state.layers[0]!.includeInAI = false;
    const withLayer = await renderCropInput(roomWithStroke(), crop, SIZE);
    const without = await renderCropInput(state, crop, SIZE);
    expect(withLayer.equals(without)).toBe(false);
  });

  it('is deterministic', async () => {
    const state = roomWithStroke();
    expect((await renderCropInput(state, crop, SIZE)).equals(await renderCropInput(state, crop, SIZE))).toBe(true);
  });
});

describe('buildMask', () => {
  it('is white inside the dirty area and black far away', () => {
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, 384);
    const ctx = built.alpha.getContext('2d');
    const center = ctx.getImageData(115, 115, 1, 1).data;
    const corner = ctx.getImageData(2, 2, 1, 1).data;
    expect(center[3]!).toBeGreaterThan(200);
    expect(corner[3]!).toBe(0);
    expect(built.plan.empty).toBe(false);
  });

  it('keeps everything outside the apply area unmasked', () => {
    const built = buildMask([{ x: 0, y: 0, width: 512, height: 512 }], crop, SIZE, 128);
    const ctx = built.alpha.getContext('2d');
    expect(ctx.getImageData(5, 5, 1, 1).data[3]!).toBe(0);
    expect(ctx.getImageData(128, 128, 1, 1).data[3]!).toBeGreaterThan(100);
  });

  it('reports an empty plan when nothing is inside the crop', () => {
    expect(buildMask([{ x: 3000, y: 3000, width: 10, height: 10 }], crop, SIZE, 384).plan.empty).toBe(true);
  });
});

describe('AICanvas', () => {
  it('composites a patch through the mask and returns the crop', async () => {
    const ai = new AICanvas(1024);
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, 384);
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
    const state = roomWithStroke();
    const imagePng = await renderCropInput(state, crop, SIZE);
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, 384);
    const backend = new MockBackend(0);
    const out = await backend.generate(
      { prompt: 'a town', negativePrompt: '', imagePng, maskPng: built.png, size: SIZE, denoise: 0.5, steps: 8, seed: 1, tag: 't' },
      new AbortController().signal,
    );
    expect(out.equals(imagePng)).toBe(false);
    expect(await nonTransparentPixels(out, SIZE)).toBe(SIZE * SIZE);

    const again = await backend.generate(
      { prompt: 'a town', negativePrompt: '', imagePng, maskPng: built.png, size: SIZE, denoise: 0.5, steps: 8, seed: 1, tag: 't' },
      new AbortController().signal,
    );
    expect(again.equals(out)).toBe(true);
  });

  it('honours the abort signal', async () => {
    const backend = new MockBackend(500);
    const controller = new AbortController();
    const state = roomWithStroke();
    const imagePng = await renderCropInput(state, crop, SIZE);
    const built = buildMask([{ x: 200, y: 200, width: 60, height: 60 }], crop, SIZE, 384);
    const promise = backend.generate(
      { prompt: 'p', negativePrompt: '', imagePng, maskPng: built.png, size: SIZE, denoise: 0.5, steps: 8, seed: 1, tag: 't' },
      controller.signal,
    );
    controller.abort();
    await expect(promise).rejects.toThrow(/aborted/);
  });
});
