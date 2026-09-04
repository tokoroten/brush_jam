import { CANVAS_SIZE, renderStrokes, type Layer, type Stroke } from '@brushjam/shared';

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
    { undone },
  );
}

/** Draw a single committed stroke incrementally (the common case). */
export function drawStroke(canvas: HTMLCanvasElement, stroke: Stroke): void {
  renderStrokes(ctxOf(canvas) as unknown as never, [stroke]);
}

export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}
