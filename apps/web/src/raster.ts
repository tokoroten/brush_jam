import { CANVAS_SIZE, alphaOf, renderStrokes, type Layer, type Point, type RenderableStroke, type Stroke } from '@brushjam/shared';

/** Temp canvases for the noise pen; kept here so both renderers share it. */
let scratchFactory = (width: number, height: number): HTMLCanvasElement => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
};

export const scratchCanvas = (width: number, height: number): HTMLCanvasElement => scratchFactory(width, height);

/** Test seam: the browser has `document`, a Node test does not. */
export function setScratchCanvasFactory(factory: (width: number, height: number) => HTMLCanvasElement): void {
  scratchFactory = factory;
}

export function createRaster(size = CANVAS_SIZE): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  return canvas;
}

export function ctxOf(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2d canvas context unavailable');
  return ctx;
}

/** Repaint one layer from the stroke log (used after undo/clear/reorder). */
export function redrawLayer(
  canvas: HTMLCanvasElement,
  layer: Layer,
  strokes: readonly Stroke[],
  undone: ReadonlySet<string>,
  images: ReadonlyMap<string, HTMLImageElement>,
): void {
  const ctx = ctxOf(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (layer.kind === 'reference') {
    const img = layer.imageId ? images.get(layer.imageId) : undefined;
    if (img) {
      const scale = layer.scale ?? 1;
      ctx.drawImage(img, layer.x ?? 0, layer.y ?? 0, img.naturalWidth * scale, img.naturalHeight * scale);
    }
    return;
  }
  // A moved draw layer is translated at render time; the log is untouched.
  renderStrokes(
    ctx as unknown as never,
    strokes.filter((s) => s.layerId === layer.id),
    {
      undone,
      offsetX: -(layer.offsetX ?? 0),
      offsetY: -(layer.offsetY ?? 0),
      bounds: { width: canvas.width, height: canvas.height },
      createCanvas: (w, h) => scratchCanvas(w, h) as never,
    },
  );
}

/** Draw a single committed stroke incrementally (the common case). */
export function drawStroke(canvas: HTMLCanvasElement, stroke: Stroke, layer?: Layer): void {
  renderStrokes(ctxOf(canvas) as unknown as never, [stroke], {
    offsetX: -(layer?.offsetX ?? 0),
    offsetY: -(layer?.offsetY ?? 0),
    bounds: { width: canvas.width, height: canvas.height },
    createCanvas: (w, h) => scratchCanvas(w, h) as never,
  });
}

/**
 * Live preview of an in-progress stroke, drawn incrementally: only the newly
 * arrived segment is rendered each frame. Re-rendering a whole noise stroke
 * every frame meant hashing its entire bounding box 60 times a second.
 */
export function drawStrokeSegment(canvas: HTMLCanvasElement, stroke: RenderableStroke): void {
  // Built at full strength and composited at the stroke's alpha when the frame
  // is drawn. Applying alpha per segment would darken the overlaps between
  // consecutive chunks, so the preview would not match the committed stroke.
  renderStrokes(ctxOf(canvas) as unknown as never, [{ ...stroke, alpha: 1 } as never], {
    bounds: { width: canvas.width, height: canvas.height },
    createCanvas: (w, h) => scratchCanvas(w, h) as never,
  });
}

/**
 * One canvas reused for compositing a layer plus its in-progress strokes. A
 * fresh 4096-square canvas per layer per frame is not affordable, and this is
 * only ever used synchronously inside one drawHumanFrame() call.
 */
let frameScratch: HTMLCanvasElement | null = null;
function layerScratch(size: number): HTMLCanvasElement {
  if (!frameScratch || frameScratch.width !== size || frameScratch.height !== size) {
    frameScratch = scratchCanvas(size, size);
  }
  const ctx = ctxOf(frameScratch);
  ctx.clearRect(0, 0, size, size);
  return frameScratch;
}

/** What drawHumanFrame needs from the room; a subset of RoomClient. */
export interface HumanFrameModel {
  canvasSize: number;
  orderedLayers: readonly Layer[];
  layerCanvases: ReadonlyMap<string, HTMLCanvasElement>;
  live: ReadonlyMap<string, { init: RenderableStroke & { layerId: string }; points: Point[] }>;
  previewRaster(id: string): HTMLCanvasElement | null;
}

/**
 * The human canvas: every visible layer in order, with each drawer's
 * in-progress stroke composited INTO its own layer rather than on top of the
 * stack. Drawing live strokes last was wrong in two visible ways: a live
 * eraser punched through every layer to the page background instead of erasing
 * its own layer, and a stroke on a lower layer appeared above the layers that
 * should cover it. Both resolve the moment the stroke is committed, which made
 * the canvas appear to jump.
 */
export function drawHumanFrame(ctx: CanvasRenderingContext2D, model: HumanFrameModel): void {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, model.canvasSize, model.canvasSize);

  type LiveEntry = [string, { init: RenderableStroke & { layerId: string }; points: Point[] }];
  const liveByLayer = new Map<string, LiveEntry[]>();
  for (const entry of model.live) {
    const layerId = entry[1].init.layerId;
    const list = liveByLayer.get(layerId);
    if (list) list.push(entry);
    else liveByLayer.set(layerId, [entry]);
  }

  for (const layer of model.orderedLayers) {
    if (!layer.visible) continue;
    const raster = model.layerCanvases.get(layer.id);
    const lives = liveByLayer.get(layer.id) ?? [];
    if (lives.length === 0) {
      if (!raster) continue;
      ctx.globalAlpha = layer.opacity;
      ctx.drawImage(raster, 0, 0);
      continue;
    }

    const scratch = layerScratch(model.canvasSize);
    const sctx = ctxOf(scratch);
    if (raster) sctx.drawImage(raster, 0, 0);
    const dx = layer.offsetX ?? 0;
    const dy = layer.offsetY ?? 0;
    for (const [id, live] of lives) {
      if (live.init.tool === 'noise') {
        const preview = model.previewRaster(id);
        if (preview) {
          // The preview raster holds the stroke at full strength; its opacity
          // is applied here, once, exactly as the committed stroke will be.
          sctx.globalAlpha = alphaOf(live.init);
          sctx.drawImage(preview, dx, dy);
          sctx.globalAlpha = 1;
        }
        continue;
      }
      renderStrokes(sctx as unknown as never, [{ ...live.init, points: live.points } as never], {
        offsetX: -dx,
        offsetY: -dy,
        bounds: { width: scratch.width, height: scratch.height },
        createCanvas: (w, h) => scratchCanvas(w, h) as never,
      });
    }
    ctx.globalAlpha = layer.opacity;
    ctx.drawImage(scratch, 0, 0);
  }
  ctx.globalAlpha = 1;
}

export const ASSET_TIMEOUT_MS = 10_000;

/**
 * Load an image with a deadline and cancellation. Without both, a single
 * stalled HTTP request would sit in the client's ordered message queue forever
 * and freeze presence, strokes and every later AI result.
 */
export function loadImageElement(
  src: string,
  options: { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<HTMLImageElement> {
  const { timeoutMs = ASSET_TIMEOUT_MS, signal } = options;
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error('cancelled'));
      return;
    }
    const img = new Image();
    let settled = false;
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      fn();
    };
    const cancel = (message: string): void => {
      img.src = ''; // tells the browser to drop the pending request
      finish(() => reject(new Error(message)));
    };
    const onAbort = (): void => cancel('cancelled');
    const timer = setTimeout(() => cancel(`timed out loading ${src}`), timeoutMs);

    img.crossOrigin = 'anonymous';
    img.onload = () => finish(() => resolve(img));
    img.onerror = () => finish(() => reject(new Error(`failed to load ${src}`)));
    signal?.addEventListener('abort', onAbort, { once: true });
    img.src = src;
  });
}
