import { createCanvas, loadImage } from '@napi-rs/canvas';
import { delay, type AIBackend, type GenerateRequest } from './types.js';

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const posterize = (v: number, levels: number): number => Math.round((Math.round((v / 255) * (levels - 1)) / (levels - 1)) * 255);

/**
 * GPU-free stand-in: a deterministic stylisation (posterise + hue shift derived
 * from the prompt + edge darkening) applied inside the mask. Same contract as
 * the real backends, so the whole scheduler path is exercised without ComfyUI.
 */
export class MockBackend implements AIBackend {
  readonly name = 'mock';

  constructor(private readonly latencyMs = 800) {}

  async generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer> {
    await delay(this.latencyMs, signal);
    const size = req.size;
    const image = await loadImage(req.imagePng);
    const mask = await loadImage(req.maskPng);

    const src = createCanvas(size, size);
    const sctx = src.getContext('2d');
    sctx.drawImage(image, 0, 0, size, size);
    const pixels = sctx.getImageData(0, 0, size, size);

    const mcanvas = createCanvas(size, size);
    const mctx = mcanvas.getContext('2d');
    mctx.drawImage(mask, 0, 0, size, size);
    const maskData = mctx.getImageData(0, 0, size, size).data;

    const h = hash(req.prompt + req.seed);
    const shift = [h % 96, (h >> 8) % 96, (h >> 16) % 96];
    const levels = 3 + (h % 3);
    const data = pixels.data;
    const original = Uint8ClampedArray.from(data);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 4;
        const m = maskData[i]! / 255;
        if (m <= 0) continue;
        const left = x > 0 ? original[i - 4]! : original[i]!;
        const up = y > 0 ? original[i - size * 4]! : original[i]!;
        const edge = Math.min(60, Math.abs(original[i]! - left) + Math.abs(original[i]! - up));
        for (let c = 0; c < 3; c++) {
          const base = original[i + c]!;
          const styled = Math.max(0, Math.min(255, posterize(base, levels) + shift[c]! - 48 - edge));
          data[i + c] = base + (styled - base) * m;
        }
        data[i + 3] = 255;
      }
    }
    sctx.putImageData(pixels, 0, 0);
    return src.toBuffer('image/png');
  }
}
