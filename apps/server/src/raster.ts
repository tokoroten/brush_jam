import { createCanvas, loadImage, type Canvas, type Image, type SKRSContext2D } from '@napi-rs/canvas';
import { CANVAS_SIZE, planMask, renderStrokes, type MaskPlan, type Rect } from '@brushjam/shared';
import { strokesForCrop, type RenderSnapshot } from './room.js';

const imageCache = new Map<string, Image>();

/** Drop decoded images for a room that went away, so the cache cannot grow forever. */
export function forgetImages(ids: Iterable<string>): void {
  for (const id of ids) imageCache.delete(id);
}

async function decode(id: string, bytes: Buffer): Promise<Image> {
  const hit = imageCache.get(id);
  if (hit) return hit;
  const img = await loadImage(bytes);
  imageCache.set(id, img);
  return img;
}

/**
 * Render the AI input for a crop: white background (SDXL needs an opaque input),
 * then every visible AI-input layer in order. Each draw layer is rasterised into
 * its own transparent canvas first and composited with the layer opacity, which
 * is exactly what the browser does - otherwise an eraser would cut through the
 * white background and lower layers, and overlapping strokes on a translucent
 * layer would accumulate opacity instead of the layer being faded once.
 */
export async function renderCropInput(snapshot: RenderSnapshot, crop: Rect, size: number): Promise<Buffer> {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);
  const scale = size / crop.width;

  for (const layer of snapshot.layers) {
    if (!layer.visible || layer.opacity <= 0 || !layer.includeInAI) continue;

    const layerCanvas = createCanvas(size, size);
    const lctx = layerCanvas.getContext('2d');
    lctx.scale(scale, scale);
    if (layer.kind === 'reference' && layer.imageId) {
      const stored = snapshot.images.get(layer.imageId);
      if (!stored) continue;
      const img = await decode(stored.id, stored.bytes);
      const s = layer.scale ?? 1;
      lctx.drawImage(img, (layer.x ?? 0) - crop.x, (layer.y ?? 0) - crop.y, stored.width * s, stored.height * s);
    } else {
      renderStrokes(lctx as unknown as never, strokesForCrop(snapshot, crop, layer.id), {
        undone: snapshot.undone,
        offsetX: crop.x,
        offsetY: crop.y,
      });
    }

    ctx.save();
    ctx.globalAlpha = layer.opacity;
    ctx.drawImage(layerCanvas, 0, 0);
    ctx.restore();
  }
  return canvas.toBuffer('image/png');
}

export interface BuiltMask {
  /** Opaque black/white PNG, white = regenerate. What ComfyUI receives. */
  png: Buffer;
  /** Same shape as an alpha channel, used locally for soft compositing. */
  alpha: Canvas;
  plan: MaskPlan;
}

function paintSoftShapes(ctx: SKRSContext2D, plan: MaskPlan, scale: number, size: number): void {
  ctx.clearRect(0, 0, size, size);
  ctx.save();
  try {
    ctx.filter = `blur(${Math.max(1, plan.feather * scale)}px)`;
  } catch {
    /* blur unsupported: hard edges are acceptable */
  }
  ctx.fillStyle = '#ffffff';
  for (const s of plan.shapes) ctx.fillRect(s.x * scale, s.y * scale, s.width * scale, s.height * scale);
  ctx.restore();
}

/**
 * Dirty stroke bboxes -> dilated, feathered soft mask, multiplied by a softened
 * rectangle limited to the central apply area (context window > applied region).
 */
export function buildMask(dirty: readonly Rect[], crop: Rect, size: number, apply: Rect): BuiltMask {
  const plan = planMask(dirty, crop, apply);
  const scale = size / crop.width;
  const alpha = createCanvas(size, size);
  const actx = alpha.getContext('2d');
  paintSoftShapes(actx, plan, scale, size);

  // limit to the central apply area, with a soft edge of its own
  const limit = createCanvas(size, size);
  const lctx = limit.getContext('2d');
  lctx.save();
  try {
    lctx.filter = `blur(${Math.max(1, plan.feather * scale)}px)`;
  } catch {
    /* ignore */
  }
  lctx.fillStyle = '#ffffff';
  const inset = plan.feather * scale;
  lctx.fillRect(
    plan.apply.x * scale + inset,
    plan.apply.y * scale + inset,
    Math.max(1, plan.apply.width * scale - inset * 2),
    Math.max(1, plan.apply.height * scale - inset * 2),
  );
  lctx.restore();
  actx.globalCompositeOperation = 'destination-in';
  actx.drawImage(limit, 0, 0);
  actx.globalCompositeOperation = 'source-over';

  // flatten onto black for ComfyUI (LoadImage -> ImageToMask reads the red channel)
  const flat = createCanvas(size, size);
  const fctx = flat.getContext('2d');
  fctx.fillStyle = '#000000';
  fctx.fillRect(0, 0, size, size);
  fctx.drawImage(alpha, 0, 0);
  return { png: flat.toBuffer('image/png'), alpha, plan };
}

/** The room's persistent AI canvas: a full-size, initially transparent raster. */
export class AICanvas {
  readonly canvas: Canvas;
  private readonly ctx: SKRSContext2D;

  constructor(readonly size: number = CANVAS_SIZE) {
    this.canvas = createCanvas(size, size);
    this.ctx = this.canvas.getContext('2d');
  }

  /** Composite an AI patch through the soft mask; returns the crop as a PNG. */
  async composite(patchPng: Buffer, crop: Rect, mask: Canvas): Promise<Buffer> {
    const patch = await loadImage(patchPng);
    const work = createCanvas(mask.width, mask.height);
    const wctx = work.getContext('2d');
    wctx.drawImage(patch, 0, 0, mask.width, mask.height);
    wctx.globalCompositeOperation = 'destination-in';
    wctx.drawImage(mask, 0, 0);

    this.ctx.drawImage(work, crop.x, crop.y, crop.width, crop.height);
    const out = createCanvas(crop.width, crop.height);
    out.getContext('2d').drawImage(this.canvas, -crop.x, -crop.y);
    return out.toBuffer('image/png');
  }

  toPng(): Buffer {
    return this.canvas.toBuffer('image/png');
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.size, this.size);
  }
}

export async function imageSize(bytes: Buffer): Promise<{ width: number; height: number }> {
  const img = await loadImage(bytes);
  return { width: img.width, height: img.height };
}
