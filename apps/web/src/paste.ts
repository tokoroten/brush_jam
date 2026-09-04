import { MAX_PASTE_SIZE } from '@brushjam/shared';

/** Scale factor that fits an image inside `max` on its long side (never upscales). */
export function fitScale(width: number, height: number, max = MAX_PASTE_SIZE): number {
  const longest = Math.max(width, height);
  return longest <= max ? 1 : max / longest;
}

/** Target pixel size for a pasted image after auto-fit. */
export function downscaleSize(width: number, height: number, max = MAX_PASTE_SIZE): { width: number; height: number } {
  const scale = fitScale(width, height, max);
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

/**
 * Where a pasted image should land: centred on the viewport, and kept inside a
 * canvas that may be much smaller than MAX_PASTE_SIZE (1024 by default).
 */
export function pastePlacement(
  center: { x: number; y: number },
  size: { width: number; height: number },
  canvasSize?: number,
): { x: number; y: number } {
  let x = Math.round(center.x - size.width / 2);
  let y = Math.round(center.y - size.height / 2);
  if (canvasSize !== undefined) {
    x = Math.round(Math.min(Math.max(x, 0), Math.max(0, canvasSize - size.width)));
    y = Math.round(Math.min(Math.max(y, 0), Math.max(0, canvasSize - size.height)));
  }
  return { x, y };
}

/** Longest side a pasted image may have on this canvas. */
export const pasteLimit = (canvasSize: number): number => Math.max(64, Math.min(MAX_PASTE_SIZE, canvasSize));

export const ACCEPTED_PASTE_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

export async function downscaleBlob(blob: Blob, max = MAX_PASTE_SIZE): Promise<{ blob: Blob; width: number; height: number }> {
  const bitmap = await createImageBitmap(blob);
  const size = downscaleSize(bitmap.width, bitmap.height, max);
  if (size.width === bitmap.width && size.height === bitmap.height && blob.type === 'image/png') {
    bitmap.close();
    return { blob, width: size.width, height: size.height };
  }
  const canvas = document.createElement('canvas');
  canvas.width = size.width;
  canvas.height = size.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0, size.width, size.height);
  bitmap.close();
  const out = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!out) throw new Error('could not encode pasted image');
  return { blob: out, width: size.width, height: size.height };
}
