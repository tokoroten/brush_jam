import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it } from 'vitest';
import { checkImage, probeImage } from '../src/imageInfo.js';

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
