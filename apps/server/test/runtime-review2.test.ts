import { createCanvas } from '@napi-rs/canvas';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MockBackend } from '../src/ai/backends/index.js';
import { loadConfig } from '../src/config.js';
import { RoomRegistry, RoomRuntime, MAX_ROOMS } from '../src/runtime.js';
import { decodedCacheStats, forgetImages, renderCropInput } from '../src/raster.js';
import { applyClientMessage, addMember, captureRenderSnapshot } from '../src/room.js';

const config = loadConfig({ AI_BACKEND: 'mock', AI_DEBOUNCE_MS: '100000', ROOM_IDLE_MS: '10000' } as NodeJS.ProcessEnv);
const backend = new MockBackend(0);

const png = (w: number, h: number, color = '#3355ff'): Buffer => {
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, w, h);
  return canvas.toBuffer('image/png');
};

let registry: RoomRegistry;

beforeEach(() => {
  registry = new RoomRegistry(backend, config);
});

afterEach(() => {
  registry.dispose();
});

/** Finding B4: header validation is a preflight; the file must really decode. */
describe('upload decoding', () => {
  it('accepts a real image', async () => {
    const room = new RoomRuntime('decode1', backend, config);
    const stored = await room.addImage(png(40, 20), 'image/png');
    expect(stored).toMatchObject({ width: 40, height: 20 });
    room.dispose();
  });

  it('rejects a file with a valid header and no pixel data', async () => {
    const room = new RoomRuntime('decode2', backend, config);
    // 24 bytes: PNG signature + IHDR claiming 8x8, then nothing at all
    const fake = Buffer.alloc(24);
    fake.writeUInt32BE(0x89504e47, 0);
    fake.writeUInt32BE(0x0d0a1a0a, 4);
    fake.write('IHDR', 12, 'ascii');
    fake.writeUInt32BE(8, 16);
    fake.writeUInt32BE(8, 20);

    // Rejected structurally, *before* any decode: loadImage segfaults on this
    // input, so a try/catch around the decoder would not have saved the process.
    const result = await room.addImage(fake, 'image/png');
    expect(result).toMatchObject({ error: expect.stringContaining('PNG') });
    expect(room.state.images.size).toBe(0);
    room.dispose();
  });

  it('rejects a truncated real image', async () => {
    const room = new RoomRuntime('decode3', backend, config);
    expect(await room.addImage(png(64, 64).subarray(0, 40), 'image/png')).toHaveProperty('error');
    const full = png(64, 64);
    // the IEND chunk chopped off: incomplete, so refused
    expect(await room.addImage(full.subarray(0, full.length - 12), 'image/png')).toHaveProperty('error');
    room.dispose();
  });
});

/** Finding B1: decoded pixels are bounded and unreferenced images are dropped. */
describe('image lifecycle', () => {
  async function roomWithReference(): Promise<{ room: RoomRuntime; userId: string; imageId: string; layerId: string }> {
    const room = new RoomRuntime('images1', backend, config);
    const userId = addMember(room.state, 'Alice').userId;
    const stored = await room.addImage(png(64, 64), 'image/png');
    if ('error' in stored) throw new Error(stored.error);
    const created = applyClientMessage(room.state, userId, {
      t: 'layer_create',
      layer: { kind: 'reference', imageId: stored.imageId, x: 0, y: 0 },
    });
    const layerId = (created.broadcast[0] as { layer: { id: string } }).layer.id;
    return { room, userId, imageId: stored.imageId, layerId };
  }

  it('keeps an image while a layer references it', async () => {
    const { room, imageId } = await roomWithReference();
    expect(room.pruneImages(Date.now() + 60 * 60_000)).toBe(0);
    expect(room.state.images.has(imageId)).toBe(true);
    room.dispose();
  });

  it('drops the image and its decoded pixels once no layer references it', async () => {
    const { room, userId, imageId, layerId } = await roomWithReference();
    applyClientMessage(room.state, userId, { t: 'layer_update', id: layerId, patch: { includeInAI: true } });
    await renderCropInput(captureRenderSnapshot(room.state), { x: 0, y: 0, width: 128, height: 128 }, 64);
    expect(decodedCacheStats().entries).toBeGreaterThan(0);

    applyClientMessage(room.state, userId, { t: 'layer_delete', id: layerId });
    expect(room.pruneImages(Date.now() + 60 * 60_000)).toBe(1);
    expect(room.state.images.has(imageId)).toBe(false);
    room.dispose();
  });

  it('does not sweep a freshly uploaded image before it becomes a layer', async () => {
    const room = new RoomRuntime('images2', backend, config);
    const stored = await room.addImage(png(16, 16), 'image/png');
    expect(room.pruneImages(Date.now())).toBe(0);
    expect(room.state.images.size).toBe(1);
    // ...but it is swept once the grace period passes with nothing using it
    expect(room.pruneImages(Date.now() + 10 * 60_000)).toBe(1);
    expect('imageId' in stored).toBe(true);
    room.dispose();
  });

  it('frees storage accounting so more uploads fit afterwards', async () => {
    const room = new RoomRuntime('images3', backend, config);
    for (let i = 0; i < 32; i++) await room.addImage(png(8, 8, `#0000${(i % 90) + 10}`), 'image/png');
    expect(await room.addImage(png(8, 8), 'image/png')).toHaveProperty('error');

    room.pruneImages(Date.now() + 10 * 60_000);
    expect(await room.addImage(png(8, 8), 'image/png')).toHaveProperty('imageId');
    room.dispose();
  });

  it('disposing a room forgets its decoded images', async () => {
    const { room, imageId } = await roomWithReference();
    room.dispose();
    forgetImages([imageId]);
    expect(room.state.images.size).toBe(0);
  });
});

/** Finding 2 (completing): the number of live rooms is capped. */
describe('room capacity', () => {
  it('creates rooms up to the cap and then refuses', () => {
    for (let i = 0; i < MAX_ROOMS; i++) expect(registry.create()).not.toBeNull();
    expect(registry.size).toBe(MAX_ROOMS);
    expect(registry.create()).toBeNull();
    expect(registry.ensure('brandnewid')).toBeNull();
  });

  it('reuses an existing room even at capacity', () => {
    const first = registry.create()!;
    for (let i = 1; i < MAX_ROOMS; i++) registry.create();
    expect(registry.ensure(first.state.id)).toBe(first);
  });

  it('makes room again after idle rooms are swept', () => {
    for (let i = 0; i < MAX_ROOMS; i++) registry.create();
    expect(registry.create()).toBeNull();
    // every room is empty, so a sweep far in the future reclaims them
    expect(registry.sweep(Date.now() + 3_600_000)).toBe(MAX_ROOMS);
    expect(registry.create()).not.toBeNull();
  });
});

/** Round 3, finding 2: the per-room upload quota survives concurrency. */
describe('upload quota under concurrency', () => {
  it('accepts exactly the cap when 33 uploads race', async () => {
    const room = new RoomRuntime('race1', backend, config);
    const results = await Promise.all(Array.from({ length: 33 }, (_, i) => room.addImage(png(8, 8, `#00${(i % 90) + 10}44`), 'image/png')));
    expect(results.filter((r) => 'imageId' in r)).toHaveLength(32);
    expect(results.filter((r) => 'error' in r)).toHaveLength(1);
    expect(room.state.images.size).toBe(32);
    room.dispose();
  });

  it('releases the reservation when a racing upload fails to decode', async () => {
    const room = new RoomRuntime('race2', backend, config);
    const bad = Buffer.alloc(24);
    bad.writeUInt32BE(0x89504e47, 0);
    bad.writeUInt32BE(0x0d0a1a0a, 4);
    bad.write('IHDR', 12, 'ascii');
    bad.writeUInt32BE(8, 16);
    bad.writeUInt32BE(8, 20);

    const results = await Promise.all([
      ...Array.from({ length: 10 }, () => room.addImage(bad, 'image/png')),
      ...Array.from({ length: 32 }, (_, i) => room.addImage(png(8, 8, `#11${(i % 90) + 10}55`), 'image/png')),
    ]);
    // the ten failures cost nothing: all 32 real uploads still fit
    expect(results.filter((r) => 'imageId' in r)).toHaveLength(32);
    expect(room.state.images.size).toBe(32);
    room.dispose();
  });
});
