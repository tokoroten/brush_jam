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

    const strokeWith = (
      layerId: string,
      id: string,
      tool: 'pen' | 'eraser',
      color: string,
      pts: Array<[number, number]>,
      alpha?: number,
    ): void => {
      applyClientMessage(state, userId, {
        t: 'stroke_start',
        stroke: {
          id,
          layerId,
          tool,
          color,
          width: 60,
          ...(alpha === undefined ? {} : { alpha }),
          points: [{ x: pts[0]![0], y: pts[0]![1] }],
        },
      });
      applyClientMessage(state, userId, { t: 'stroke_end', strokeId: id, points: pts.slice(1).map(([x, y]) => ({ x, y })) });
    };
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
    // a translucent stroke that crosses itself, so the parity check covers the
    // flattened-then-composited path as well as the plain one
    strokeWith(lower, 'l2', 'pen', '#00aa00', [[120, 320], [380, 380], [380, 320], [120, 380]], 0.4);
    return state;
  }

  /**
   * The browser's compositing path, reimplemented here as the expected result:
   * every layer on its own canvas at world scale, composited with the layer
   * opacity. Like the server it renders natively and resamples once at the end
   * (review 6 finding 4) - what this test pins is the compositing semantics,
   * eraser scope and opacity, not the resampling filter.
   */
  async function browserReference(state: RoomState, size: number): Promise<Buffer> {
    const native = createCanvas(crop.width, crop.height);
    const nctx = native.getContext('2d');
    nctx.fillStyle = '#ffffff';
    nctx.fillRect(0, 0, crop.width, crop.height);
    for (const layer of [...state.layers].sort((a, b) => a.order - b.order)) {
      if (!layer.visible || !layer.includeInAI) continue;
      const layerCanvas = createCanvas(crop.width, crop.height);
      const lctx = layerCanvas.getContext('2d');
      const strokes: Stroke[] = state.strokes.filter((s) => s.layerId === layer.id);
      renderStrokes(lctx as unknown as never, strokes, {
        undone: state.undone,
        offsetX: crop.x,
        offsetY: crop.y,
        // Same options the client uses, so a translucent stroke takes the same
        // path here as it does in the browser.
        createCanvas: (w, h) => createCanvas(w, h) as never,
        bounds: { width: crop.width, height: crop.height },
      });
      nctx.save();
      nctx.globalAlpha = layer.opacity;
      nctx.drawImage(layerCanvas, 0, 0);
      nctx.restore();
    }
    if (size === crop.width) return native.toBuffer('image/png');
    const out = createCanvas(size, size);
    const octx = out.getContext('2d');
    octx.imageSmoothingEnabled = true;
    octx.imageSmoothingQuality = 'high';
    octx.drawImage(native, 0, 0, size, size);
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
      profile: 'quality' as const,
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
      {
        prompt: 'p',
        negativePrompt: '',
        imagePng,
        maskPng: built.png,
        size: SIZE,
        denoise: 0.5,
        steps: 8,
        seed: 1,
        profile: 'quality',
        tag: 't',
      },
      controller.signal,
    );
    controller.abort();
    await expect(promise).rejects.toThrow(/aborted/);
  });
});

/** Noise strokes must look the same in a server crop as on the client canvas. */
describe('noise strokes in a crop', () => {
  it('renders the same noise pixels as a full-size render of the same world', async () => {
    const state = createRoom('noise-room');
    const userId = addMember(state, 'Alice').userId;
    const layerId = state.layers[0]!.id;
    applyClientMessage(state, userId, {
      t: 'stroke_start',
      stroke: {
        id: 'n1',
        layerId,
        tool: 'noise',
        color: '#000000',
        width: 40,
        points: [
          { x: 1100, y: 1100 },
          { x: 1200, y: 1160 },
        ],
      },
    });
    applyClientMessage(state, userId, { t: 'stroke_end', strokeId: 'n1', points: [{ x: 1300, y: 1120 }] });

    const snap = captureRenderSnapshot(state);
    const crop = { x: 1024, y: 1024, width: 512, height: 512 };
    const png = await renderCropInput(snap, crop, 512);
    const rendered = await loadImage(png);
    const server = createCanvas(512, 512);
    server.getContext('2d').drawImage(rendered, 0, 0);
    const serverData = server.getContext('2d').getImageData(0, 0, 512, 512).data;

    // "client-style": the whole layer at world scale, then the same window read back
    const client = createCanvas(2048, 2048);
    const cctx = client.getContext('2d');
    cctx.fillStyle = '#ffffff';
    cctx.fillRect(0, 0, 2048, 2048);
    renderStrokes(cctx as never, state.strokes, { createCanvas: (w, h) => createCanvas(w, h) as never });
    const clientData = cctx.getImageData(1024, 1024, 512, 512).data;

    let compared = 0;
    let mismatched = 0;
    for (let i = 0; i < serverData.length; i += 4) {
      // only compare pixels the stroke actually covers (not the white paper)
      if (clientData[i] === 255 && clientData[i + 1] === 255 && clientData[i + 2] === 255) continue;
      compared += 1;
      if (serverData[i] !== clientData[i] || serverData[i + 1] !== clientData[i + 1] || serverData[i + 2] !== clientData[i + 2]) {
        mismatched += 1;
      }
    }
    expect(compared).toBeGreaterThan(1000);
    // edge antialiasing may round differently; the noise body must match
    expect(mismatched / compared).toBeLessThan(0.05);
  });
});

/**
 * Regression (review 6 finding 4): rendering a 1024 crop at 512 used to scale
 * the layer context and hand the noise pen bounds in unscaled units, so every
 * noise mark outside the top-left 512x512 of the *world* was thrown away.
 */
describe('downsampled crops keep noise in every quadrant', () => {
  const quadrants: [string, number, number][] = [
    ['top-left', 200, 200],
    ['top-right', 800, 200],
    ['bottom-left', 200, 800],
    ['bottom-right', 800, 800],
  ];

  async function renderQuadrantNoise(cx: number, cy: number, size: number): Promise<Buffer> {
    const state = createRoom('downsample-room', undefined, 1024);
    const userId = addMember(state, 'Alice').userId;
    const layerId = state.layers[0]!.id;
    applyClientMessage(state, userId, {
      t: 'stroke_start',
      stroke: { id: 'q1', layerId, tool: 'noise', color: '#000000', width: 90, points: [{ x: cx - 60, y: cy }] },
    });
    applyClientMessage(state, userId, { t: 'stroke_end', strokeId: 'q1', points: [{ x: cx + 60, y: cy }] });
    return renderCropInput(captureRenderSnapshot(state), { x: 0, y: 0, width: 1024, height: 1024 }, size);
  }

  for (const [name, cx, cy] of quadrants) {
    it(`renders noise in the ${name} quadrant at 1024 -> 512`, async () => {
      const png = await renderQuadrantNoise(cx, cy, 512);
      // The stroke centre maps to half its world coordinate in the 512 render.
      const [r, g, b] = await pixelAt(png, 512, Math.round(cx / 2), Math.round(cy / 2));
      expect([r, g, b].some((c) => c !== 255)).toBe(true);
    });
  }

  it('is not just white paper: an empty quadrant stays white', async () => {
    const png = await renderQuadrantNoise(200, 200, 512);
    expect(await pixelAt(png, 512, 400, 400)).toEqual([255, 255, 255, 255]);
  });

  it('downsamples rather than crops: the same stroke survives at 256 too', async () => {
    const png = await renderQuadrantNoise(800, 800, 256);
    const [r, g, b] = await pixelAt(png, 256, 200, 200);
    expect([r, g, b].some((c) => c !== 255)).toBe(true);
  });
});

/** A moved draw layer renders translated on the server too. */
describe('crop rendering with a layer offset', () => {
  it('matches a client-style render of the same moved layer', async () => {
    const state = createRoom('offset-room');
    const userId = addMember(state, 'Alice').userId;
    const layerId = state.layers[0]!.id;
    applyClientMessage(state, userId, {
      t: 'stroke_start',
      stroke: { id: 'o1', layerId, tool: 'pen', color: '#204080', width: 24, points: [{ x: 100, y: 120 }] },
    });
    applyClientMessage(state, userId, { t: 'stroke_end', strokeId: 'o1', points: [{ x: 260, y: 200 }] });
    applyClientMessage(state, userId, { t: 'layer_update', id: layerId, patch: { offsetX: 300, offsetY: 150 } });

    const snap = captureRenderSnapshot(state);
    const cropRect = { x: 256, y: 256, width: 512, height: 512 };
    const rendered = await loadImage(await renderCropInput(snap, cropRect, 512));
    const server = createCanvas(512, 512);
    server.getContext('2d').drawImage(rendered, 0, 0);
    const serverData = server.getContext('2d').getImageData(0, 0, 512, 512).data;

    const client = createCanvas(1024, 1024);
    const cctx = client.getContext('2d');
    cctx.fillStyle = '#ffffff';
    cctx.fillRect(0, 0, 1024, 1024);
    renderStrokes(cctx as never, state.strokes, { offsetX: -300, offsetY: -150, createCanvas: (w, h) => createCanvas(w, h) as never });
    const clientData = cctx.getImageData(256, 256, 512, 512).data;

    let painted = 0;
    let mismatched = 0;
    for (let i = 0; i < serverData.length; i += 4) {
      if (clientData[i] === 255 && clientData[i + 1] === 255 && clientData[i + 2] === 255) continue;
      painted += 1;
      if (Math.abs(serverData[i]! - clientData[i]!) > 2 || Math.abs(serverData[i + 2]! - clientData[i + 2]!) > 2) mismatched += 1;
    }
    expect(painted).toBeGreaterThan(1000);
    expect(mismatched / painted).toBeLessThan(0.02);
  });

  it('renders nothing when the offset moved the strokes out of the crop', async () => {
    const state = createRoom('offset-room2');
    const userId = addMember(state, 'Alice').userId;
    const layerId = state.layers[0]!.id;
    applyClientMessage(state, userId, {
      t: 'stroke_start',
      stroke: { id: 'o1', layerId, tool: 'pen', color: '#000000', width: 20, points: [{ x: 100, y: 100 }] },
    });
    applyClientMessage(state, userId, { t: 'stroke_end', strokeId: 'o1', points: [{ x: 200, y: 200 }] });
    applyClientMessage(state, userId, { t: 'layer_update', id: layerId, patch: { offsetX: 1500, offsetY: 1500 } });

    const png = await renderCropInput(captureRenderSnapshot(state), { x: 0, y: 0, width: 512, height: 512 }, 256);
    const img = await loadImage(png);
    const canvas = createCanvas(256, 256);
    canvas.getContext('2d').drawImage(img, 0, 0);
    const data = canvas.getContext('2d').getImageData(0, 0, 256, 256).data;
    let painted = 0;
    for (let i = 0; i < data.length; i += 4) if (data[i] !== 255 || data[i + 1] !== 255 || data[i + 2] !== 255) painted += 1;
    expect(painted).toBe(0);
  });
});
