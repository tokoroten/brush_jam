import { CANVAS_SIZE, renderStrokes, type Layer, type Stroke } from '@brushjam/shared';

/** Temp canvases for the noise pen; kept here so both renderers share it. */
export const scratchCanvas = (width: number, height: number): HTMLCanvasElement => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
};

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
  renderStrokes(
    ctx as unknown as never,
    strokes.filter((s) => s.layerId === layer.id),
    { undone, createCanvas: (w, h) => scratchCanvas(w, h) as never },
  );
}

/** Draw a single committed stroke incrementally (the common case). */
export function drawStroke(canvas: HTMLCanvasElement, stroke: Stroke): void {
  renderStrokes(ctxOf(canvas) as unknown as never, [stroke], { createCanvas: (w, h) => scratchCanvas(w, h) as never });
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
