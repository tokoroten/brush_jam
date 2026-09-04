import { CANVAS_SIZE } from '@brushjam/shared';

export interface ImageInfo {
  mime: 'image/png' | 'image/jpeg' | 'image/webp';
  width: number;
  height: number;
}

/** Uploads may not exceed the world size on either side, nor this pixel count. */
export const MAX_IMAGE_SIDE = CANVAS_SIZE;
export const MAX_IMAGE_PIXELS = 4096 * 4096;

/**
 * Read the declared dimensions straight from the file header. A tiny, highly
 * compressed PNG can claim 50000x50000 and make a native decode allocate
 * gigabytes, so nothing reaches the decoder until the header looks sane.
 */
export function probeImage(bytes: Buffer): ImageInfo | null {
  return probePng(bytes) ?? probeJpeg(bytes) ?? probeWebp(bytes);
}

function probePng(b: Buffer): ImageInfo | null {
  if (b.length < 24) return null;
  if (b.readUInt32BE(0) !== 0x89504e47 || b.readUInt32BE(4) !== 0x0d0a1a0a) return null;
  if (b.toString('ascii', 12, 16) !== 'IHDR') return null;
  return { mime: 'image/png', width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}

function probeJpeg(b: Buffer): ImageInfo | null {
  if (b.length < 4 || b[0] !== 0xff || b[1] !== 0xd8) return null;
  let offset = 2;
  while (offset + 9 < b.length) {
    if (b[offset] !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = b[offset + 1]!;
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      offset += 2;
      continue;
    }
    const length = b.readUInt16BE(offset + 2);
    // SOF0..SOF15, excluding the DHT/JPG/DAC markers that share the range
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      return { mime: 'image/jpeg', height: b.readUInt16BE(offset + 5), width: b.readUInt16BE(offset + 7) };
    }
    if (length < 2) return null;
    offset += 2 + length;
  }
  return null;
}

function probeWebp(b: Buffer): ImageInfo | null {
  if (b.length < 30) return null;
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WEBP') return null;
  const format = b.toString('ascii', 12, 16);
  if (format === 'VP8 ') {
    return { mime: 'image/webp', width: b.readUInt16LE(26) & 0x3fff, height: b.readUInt16LE(28) & 0x3fff };
  }
  if (format === 'VP8L') {
    const bits = b.readUInt32LE(21);
    return { mime: 'image/webp', width: (bits & 0x3fff) + 1, height: ((bits >> 14) & 0x3fff) + 1 };
  }
  if (format === 'VP8X') {
    const w = 1 + (b[24]! | (b[25]! << 8) | (b[26]! << 16));
    const h = 1 + (b[27]! | (b[28]! << 8) | (b[29]! << 16));
    return { mime: 'image/webp', width: w, height: h };
  }
  return null;
}

export type ImageCheck = { ok: true; info: ImageInfo } | { ok: false; error: string };

export function checkImage(bytes: Buffer, declaredMime: string): ImageCheck {
  const info = probeImage(bytes);
  if (!info) return { ok: false, error: 'unrecognised image: expected PNG, JPEG or WebP' };
  if (info.mime !== declaredMime) return { ok: false, error: `content-type ${declaredMime} does not match the actual ${info.mime}` };
  if (info.width < 1 || info.height < 1) return { ok: false, error: 'image has no pixels' };
  if (info.width > MAX_IMAGE_SIDE || info.height > MAX_IMAGE_SIDE) {
    return { ok: false, error: `image is larger than ${MAX_IMAGE_SIDE}px on a side (${info.width}x${info.height})` };
  }
  if (info.width * info.height > MAX_IMAGE_PIXELS) return { ok: false, error: 'image has too many pixels' };
  return { ok: true, info };
}
