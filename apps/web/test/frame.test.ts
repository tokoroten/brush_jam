import { createCanvas } from '@napi-rs/canvas';
import { beforeAll, describe, expect, it } from 'vitest';
import type { Layer, Point, RenderableStroke } from '@brushjam/shared';
import { drawHumanFrame, setScratchCanvasFactory, type HumanFrameModel } from '../src/raster.js';

/**
 * Review 10 finding 3: live strokes used to be drawn on top of the whole
 * stack, so a live eraser punched through every layer to the page background
 * and a live stroke on a lower layer floated above the layers covering it -
 * both snapping into place the instant the stroke committed.
 */
const SIZE = 64;

beforeAll(() => {
  setScratchCanvasFactory((w, h) => createCanvas(w, h) as unknown as HTMLCanvasElement);
});

const layer = (id: string, order: number, extra: Partial<Layer> = {}): Layer => ({
  id,
  name: id,
  kind: 'draw',
  visible: true,
  locked: false,
  opacity: 1,
  order,
  includeInAI: true,
  ...extra,
});

const line = (tool: 'pen' | 'eraser', color: string, layerId: string): { init: RenderableStroke & { layerId: string }; points: Point[] } => ({
  init: { id: `s-${tool}-${layerId}`, tool, color, width: 40, points: [], layerId },
  points: [
    { x: 4, y: 32 },
    { x: 60, y: 32 },
  ],
});

/** A filled layer raster, so "did the live stroke stay inside it" is visible. */
function filled(color: string): HTMLCanvasElement {
  const c = createCanvas(SIZE, SIZE);
  const ctx = c.getContext('2d');
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, SIZE, SIZE);
  return c as unknown as HTMLCanvasElement;
}

function frame(model: Partial<HumanFrameModel> & Pick<HumanFrameModel, 'orderedLayers' | 'layerCanvases' | 'live'>): {
  at: (x: number, y: number) => [number, number, number, number];
} {
  const target = createCanvas(SIZE, SIZE);
  const ctx = target.getContext('2d');
  drawHumanFrame(ctx as unknown as CanvasRenderingContext2D, {
    canvasSize: SIZE,
    previewRaster: () => null,
    ...model,
  });
  const data = ctx.getImageData(0, 0, SIZE, SIZE).data;
  return {
    at: (x, y) => {
      const i = (y * SIZE + x) * 4;
      return [data[i]!, data[i + 1]!, data[i + 2]!, data[i + 3]!];
    },
  };
}

describe('drawHumanFrame', () => {
  it('keeps a live eraser inside its own layer, leaving the white background', () => {
    const a = layer('a', 0);
    const { at } = frame({
      orderedLayers: [a],
      layerCanvases: new Map([['a', filled('#3366aa')]]),
      live: new Map([['e1', line('eraser', '#000000', 'a')]]),
    });
    // Where the eraser ran, the layer's blue is gone but the canvas is still
    // the white background - not a transparent hole showing the page behind.
    const [r, g, b, alpha] = at(32, 32);
    expect(alpha).toBe(255);
    expect([r, g, b]).toEqual([255, 255, 255]);
  });

  it('does not let a live eraser remove a layer underneath it', () => {
    const under = layer('under', 0);
    const over = layer('over', 1);
    const { at } = frame({
      orderedLayers: [under, over],
      layerCanvases: new Map([
        ['under', filled('#22aa44')],
        ['over', filled('#3366aa')],
      ]),
      live: new Map([['e1', line('eraser', '#000000', 'over')]]),
    });
    // erasing the top layer reveals the one below, not the page
    expect(at(32, 32)).toEqual([34, 170, 68, 255]);
  });

  it('draws a live stroke under the layers that cover it', () => {
    const under = layer('under', 0);
    const over = layer('over', 1);
    const { at } = frame({
      orderedLayers: [under, over],
      layerCanvases: new Map([
        ['under', filled('#ffffff')],
        ['over', filled('#3366aa')],
      ]),
      live: new Map([['p1', line('pen', '#ff0000', 'under')]]),
    });
    // the red live stroke is on the lower layer, so the blue layer hides it
    expect(at(32, 32)).toEqual([51, 102, 170, 255]);
  });

  it('shows a live stroke on the top layer', () => {
    const under = layer('under', 0);
    const over = layer('over', 1);
    const { at } = frame({
      orderedLayers: [under, over],
      layerCanvases: new Map([
        ['under', filled('#ffffff')],
        ['over', filled('#3366aa')],
      ]),
      live: new Map([['p1', line('pen', '#ff0000', 'over')]]),
    });
    expect(at(32, 32)).toEqual([255, 0, 0, 255]);
  });

  it('respects layer visibility for live strokes too', () => {
    const hidden = layer('hidden', 0, { visible: false });
    const { at } = frame({
      orderedLayers: [hidden],
      layerCanvases: new Map([['hidden', filled('#3366aa')]]),
      live: new Map([['p1', line('pen', '#ff0000', 'hidden')]]),
    });
    expect(at(32, 32)).toEqual([255, 255, 255, 255]);
  });

  it('applies the layer opacity to the live stroke as well', () => {
    const faint = layer('faint', 0, { opacity: 0.5 });
    const { at } = frame({
      orderedLayers: [faint],
      layerCanvases: new Map([['faint', createCanvas(SIZE, SIZE) as unknown as HTMLCanvasElement]]),
      live: new Map([['p1', line('pen', '#ff0000', 'faint')]]),
    });
    const [r, g, b] = at(32, 32);
    // half-strength red over white, not full red
    expect(r).toBe(255);
    expect(g).toBeGreaterThan(100);
    expect(g).toBeLessThan(160);
    expect(b).toBe(g);
  });

  it('ignores a live stroke whose layer no longer exists', () => {
    const { at } = frame({
      orderedLayers: [layer('a', 0)],
      layerCanvases: new Map([['a', filled('#ffffff')]]),
      live: new Map([['p1', line('pen', '#ff0000', 'gone')]]),
    });
    expect(at(32, 32)).toEqual([255, 255, 255, 255]);
  });
});
