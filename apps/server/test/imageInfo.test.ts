import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it } from 'vitest';
import { checkImage, probeImage, validateStructure } from '../src/imageInfo.js';

function realPng(width: number, height: number): Buffer {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#123456';
  ctx.fillRect(0, 0, width, height);
  return canvas.toBuffer('image/png');
}

function realJpeg(width: number, height: number): Buffer {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#abcdef';
  ctx.fillRect(0, 0, width, height);
  return canvas.toBuffer('image/jpeg');
}

/** Finding 3: dimensions come from the header, before any decode. */
describe('probeImage', () => {
  it('reads PNG dimensions', () => {
    expect(probeImage(realPng(64, 32))).toEqual({ mime: 'image/png', width: 64, height: 32 });
  });

  it('reads JPEG dimensions', () => {
    expect(probeImage(realJpeg(40, 24))).toEqual({ mime: 'image/jpeg', width: 40, height: 24 });
  });

  it('returns null for junk', () => {
    expect(probeImage(Buffer.from('not an image at all, really not'))).toBeNull();
    expect(probeImage(Buffer.alloc(0))).toBeNull();
  });
});

describe('checkImage', () => {
  it('accepts a normal upload', () => {
    const result = checkImage(realPng(100, 50), 'image/png');
    expect(result).toEqual({ ok: true, info: { mime: 'image/png', width: 100, height: 50 } });
  });

  it('rejects a decompression bomb declared in the header', () => {
    // A valid 1x1 PNG whose IHDR claims 50000x50000: only 70-odd bytes on the
    // wire, but multiple gigabytes if it ever reached the decoder.
    const bomb = realPng(1, 1);
    bomb.writeUInt32BE(50_000, 16);
    bomb.writeUInt32BE(50_000, 20);
    const result = checkImage(bomb, 'image/png');
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toMatch(/larger than/);
  });

  it('rejects a content-type that lies about the format', () => {
    const result = checkImage(realPng(10, 10), 'image/jpeg');
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toMatch(/does not match/);
  });

  it('rejects unrecognised bytes', () => {
    expect(checkImage(Buffer.from('<html>hello</html>'), 'image/png').ok).toBe(false);
  });

  it('accepts an image exactly at the size limit', () => {
    const edge = realPng(1, 1);
    edge.writeUInt32BE(4096, 16);
    edge.writeUInt32BE(4096, 20);
    expect(checkImage(edge, 'image/png').ok).toBe(true);
  });
});

/**
 * `loadImage` segfaults (exit 139, verified) on a PNG with a valid header and no
 * image data, so these must be rejected in pure JS before anything decodes.
 */
describe('validateStructure', () => {
  const headerOnlyPng = (): Buffer => {
    const b = Buffer.alloc(24);
    b.writeUInt32BE(0x89504e47, 0);
    b.writeUInt32BE(0x0d0a1a0a, 4);
    b.write('IHDR', 12, 'ascii');
    b.writeUInt32BE(8, 16);
    b.writeUInt32BE(8, 20);
    return b;
  };

  it('accepts a complete PNG', () => {
    expect(validateStructure(realPng(32, 32), 'image/png')).toBeNull();
  });

  it('rejects a PNG header with no chunks (the crashing case)', () => {
    expect(validateStructure(headerOnlyPng(), 'image/png')).toMatch(/IEND|image data/);
    expect(checkImage(headerOnlyPng(), 'image/png').ok).toBe(false);
  });

  it('rejects a PNG whose IEND chunk is missing', () => {
    const full = realPng(32, 32);
    expect(validateStructure(full.subarray(0, full.length - 12), 'image/png')).toMatch(/IEND/);
  });

  it('rejects a PNG chunk claiming an impossible length', () => {
    const b = realPng(32, 32);
    b.writeUInt32BE(0xfffffff0, 8);
    expect(validateStructure(b, 'image/png')).toBeTruthy();
  });

  it('accepts a complete JPEG and rejects a truncated one', () => {
    const jpeg = realJpeg(32, 32);
    expect(validateStructure(jpeg, 'image/jpeg')).toBeNull();
    expect(validateStructure(jpeg.subarray(0, jpeg.length - 2), 'image/jpeg')).toMatch(/end-of-image/);
  });

  it('rejects a WebP whose RIFF payload is truncated', () => {
    const b = Buffer.alloc(64);
    b.write('RIFF', 0, 'ascii');
    b.writeUInt32LE(10_000, 4);
    b.write('WEBP', 8, 'ascii');
    expect(validateStructure(b, 'image/webp')).toMatch(/truncated/);
  });
});
