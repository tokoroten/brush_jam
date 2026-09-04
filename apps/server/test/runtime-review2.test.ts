import { createCanvas } from '@napi-rs/canvas';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MockBackend } from '../src/ai/backends/index.js';
import type { BackendCapabilities } from '../src/ai/backends/types.js';
import { loadConfig } from '../src/config.js';
import { RoomRegistry, RoomRuntime, MAX_ROOMS, looksLikeLimitError } from '../src/runtime.js';
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

/** Full-canvas results are whole-canvas PNGs; only a couple are kept. */
describe('patch retention', () => {
  it('keeps at most two results in full mode', async () => {
    const full = loadConfig({ AI_BACKEND: 'mock', AI_MODE: 'full', AI_DEBOUNCE_MS: '0', CANVAS_SIZE: '512' } as NodeJS.ProcessEnv);
    const room = new RoomRuntime('fullpatches', new MockBackend(0), full);
    const userId = addMember(room.state, 'Alice').userId;
    const layerId = room.state.layers[0]!.id;

    const urls: string[] = [];
    for (let i = 0; i < 6; i++) {
      applyClientMessage(room.state, userId, {
        t: 'stroke_start',
        stroke: { id: `s${i}`, layerId, tool: 'pen', color: '#000000', width: 8, points: [{ x: 10 + i, y: 10 }] },
      });
      applyClientMessage(room.state, userId, { t: 'stroke_end', strokeId: `s${i}`, points: [{ x: 60 + i, y: 60 }] });
      room.scheduler.markDirty([{ x: 0, y: 0, width: 64, height: 64 }]);
      await new Promise((r) => setTimeout(r, 120));
      const last = room.state.aiRevision;
      if (last > 0) urls.push(String(last));
    }
    expect(room.state.aiRevision).toBeGreaterThan(0);
    expect(urls.length).toBeGreaterThan(2);
    expect(room.patchCount).toBeGreaterThan(0);
    expect(room.patchCount).toBeLessThanOrEqual(2);
    room.dispose();
  });
});

/** Dirty regions outside the canvas can never be generated. */
describe('dirty region clipping', () => {
  it('clips a stroke on a far-moved layer to the canvas', async () => {
    const patch = loadConfig({ AI_BACKEND: 'mock', AI_MODE: 'patch', CANVAS_SIZE: '1024', AI_DEBOUNCE_MS: '100000' } as NodeJS.ProcessEnv);
    const room = new RoomRuntime('clip', new MockBackend(0), patch);
    const userId = addMember(room.state, 'Alice').userId;
    const layerId = room.state.layers[0]!.id;
    room.handle(userId, JSON.stringify({ t: 'layer_update', id: layerId, patch: { offsetX: 2000, offsetY: 2000 } }));
    room.handle(
      userId,
      JSON.stringify({
        t: 'stroke_start',
        stroke: { id: 'far', layerId, tool: 'pen', color: '#000000', width: 8, points: [{ x: 10, y: 10 }] },
      }),
    );
    room.handle(userId, JSON.stringify({ t: 'stroke_end', strokeId: 'far', points: [{ x: 40, y: 40 }] }));

    // the stroke lands at ~2010,2010 in world space: entirely off a 1024 canvas
    for (const r of room.scheduler.dirtyRegions) {
      expect(r.x).toBeLessThan(1024);
      expect(r.y).toBeLessThan(1024);
      expect(r.x + r.width).toBeGreaterThan(0);
    }
    expect(room.scheduler.dirtyRegions).toHaveLength(0);
    room.dispose();
  });
});

/**
 * Review 7 finding 2: a worker that restarts with different limits leaves every
 * room asking for a size it will now refuse - a 400 on every generation.
 */
describe('backend limits changing under a live room', () => {
  /** Enough of a socket for join(): the room only ever sends to it. */
  class FakeSocket {
    readonly sent: string[] = [];
    readyState = 1;
    send(data: string): void {
      this.sent.push(data);
    }
    close(): void {
      this.readyState = 3;
    }
  }

  function room(): RoomRuntime {
    const config = loadConfig({ AI_BACKEND: 'mock', CANVAS_SIZE: '1024' } as NodeJS.ProcessEnv);
    return new RoomRuntime('limits', new MockBackend(1), config, {
      profiles: ['fast', 'quality'],
      maxDenoise: 0.95,
      maxResolution: 1024,
    });
  }

  it('clamps a room into smaller limits and tells everyone', () => {
    const rt = room();
    const socket = new FakeSocket();
    rt.join(socket as never, 'Alice', 'tok-a');
    // fast default: starts at 768 with room to reach 1024
    expect(rt.state.aiResolution).toBe(768);
    expect(rt.state.aiResolutionMax).toBe(1024);

    rt.applyLimits({ profiles: ['fast'], maxDenoise: 0.8, maxResolution: 512 });

    expect(rt.state.aiResolutionMax).toBe(512);
    expect(rt.state.aiResolution).toBe(512);
    expect(rt.state.maxDenoise).toBe(0.8);
    expect(rt.state.denoise).toBeLessThanOrEqual(0.8);
    expect(rt.state.aiProfiles).toEqual(['fast']);
    expect(rt.state.aiProfile).toBe('fast');
    // and everyone in the room is told, not just the next joiner
    const settings = socket.sent.map((m) => JSON.parse(m) as { t: string }).filter((m) => m.t === 'ai_settings_changed');
    expect(settings.at(-1)).toMatchObject({ aiResolution: 512, aiProfile: 'fast' });
  });

  it('raises the ceiling again when the worker comes back bigger', () => {
    const rt = room();
    rt.join(new FakeSocket() as never, 'Alice', 'tok-a');
    rt.applyLimits({ profiles: ['fast'], maxDenoise: 0.8, maxResolution: 512 });
    expect(rt.state.aiResolutionMax).toBe(512);
    expect(rt.state.aiResolution).toBe(512);

    rt.applyLimits({ profiles: ['fast', 'quality'], maxDenoise: 0.95, maxResolution: 1024 });
    expect(rt.state.aiResolutionMax).toBe(1024);
    expect(rt.state.aiProfiles).toEqual(['fast', 'quality']);
    // the room stays where it was clamped to; the user can raise it again
    expect(rt.state.aiResolution).toBe(512);
  });

  it('keeps a room that is already inside the new limits untouched', () => {
    const rt = room();
    rt.join(new FakeSocket() as never, 'Alice', 'tok-a');
    const before = { ...rt.state };
    rt.applyLimits({ profiles: ['fast', 'quality'], maxDenoise: 0.95, maxResolution: 1024 });
    expect(rt.state.aiResolution).toBe(before.aiResolution);
    expect(rt.state.aiProfile).toBe(before.aiProfile);
  });
});

/** Review 7 finding 2, registry half: notice the change and push it out. */
describe('capability re-probing', () => {
  class ShiftingBackend extends MockBackend {
    maxResolution = 1024;
    profiles: ('fast' | 'quality')[] = ['fast', 'quality'];
    probes = 0;
    override async capabilities(): Promise<BackendCapabilities> {
      this.probes += 1;
      return {
        profiles: this.profiles,
        maxResolution: this.maxResolution,
        maxDenoise: 0.95,
        negativePromptActive: { fast: true, quality: true },
      };
    }
  }

  const cfg = loadConfig({ AI_BACKEND: 'mock', CANVAS_SIZE: '1024' } as NodeJS.ProcessEnv);

  it('pushes smaller limits into every live room', async () => {
    const backend = new ShiftingBackend(1);
    const registry = new RoomRegistry(backend, cfg, { profiles: ['fast', 'quality'], maxDenoise: 0.95, maxResolution: 1024 });
    const a = registry.ensure('rooma')!;
    const b = registry.ensure('roomb')!;

    backend.maxResolution = 512;
    await registry.refreshCapabilities();

    expect(a.state.aiResolutionMax).toBe(512);
    expect(b.state.aiResolutionMax).toBe(512);
    expect(registry.backendLimits.maxResolution).toBe(512);
  });

  it('does nothing when the limits are unchanged', async () => {
    const backend = new ShiftingBackend(1);
    const registry = new RoomRegistry(backend, cfg, { profiles: ['fast', 'quality'], maxDenoise: 0.95, maxResolution: 1024 });
    const room = registry.ensure('roomc')!;
    const before = room.state.aiResolutionMax;
    await registry.refreshCapabilities();
    expect(room.state.aiResolutionMax).toBe(before);
  });

  it('applies the new limits to rooms created afterwards too', async () => {
    const backend = new ShiftingBackend(1);
    const registry = new RoomRegistry(backend, cfg, { profiles: ['fast', 'quality'], maxDenoise: 0.95, maxResolution: 1024 });
    backend.profiles = ['fast'];
    backend.maxResolution = 768;
    await registry.refreshCapabilities();
    expect(registry.ensure('roomd')!.state.aiProfiles).toEqual(['fast']);
    expect(registry.ensure('roomd')!.state.aiResolutionMax).toBe(768);
  });

  it('survives a probe that throws, keeping the last known limits', async () => {
    const backend = new ShiftingBackend(1);
    const registry = new RoomRegistry(backend, cfg, { profiles: ['fast', 'quality'], maxDenoise: 0.95, maxResolution: 1024 });
    const room = registry.ensure('roome')!;
    backend.capabilities = async () => {
      throw new Error('worker is down');
    };
    await registry.refreshCapabilities();
    expect(room.state.aiResolutionMax).toBe(1024);
  });

  it('only re-probes for errors that look like a limit problem', () => {
    expect(looksLikeLimitError('stream worker /generate failed: 400 size 1024 out of range [256, 768]')).toBe(true);
    expect(looksLikeLimitError('this profile is not supported')).toBe(true);
    expect(looksLikeLimitError('generation timed out')).toBe(false);
    expect(looksLikeLimitError('ECONNREFUSED')).toBe(false);
  });
});

/** The capability must reach a live room, not just the snapshot of a new one. */
describe('negative prompt activity through the registry', () => {
  class NegBackend extends MockBackend {
    active = { fast: false, quality: true };
    override async capabilities(): Promise<BackendCapabilities> {
      return { profiles: ['fast', 'quality'], maxResolution: 1024, maxDenoise: 0.95, negativePromptActive: this.active };
    }
  }

  const cfg = loadConfig({ AI_BACKEND: 'mock', CANVAS_SIZE: '1024' } as NodeJS.ProcessEnv);
  const limits = { profiles: ['fast', 'quality'] as const, maxDenoise: 0.95, maxResolution: 1024 };

  it('pushes a changed negative-prompt capability into live rooms', async () => {
    const backend = new NegBackend(1);
    const registry = new RoomRegistry(backend, cfg, { ...limits, profiles: [...limits.profiles] });
    const room = registry.ensure('negreg')!;
    room.state.aiProfile = 'fast';

    await registry.refreshCapabilities();

    expect(room.state.negativeActive).toEqual({ fast: false, quality: true });
  });

  it('treats a changed capability alone as a reason to re-broadcast', async () => {
    const backend = new NegBackend(1);
    const registry = new RoomRegistry(backend, cfg, {
      ...limits,
      profiles: [...limits.profiles],
      negativePromptActive: { fast: false, quality: true },
    });
    const room = registry.ensure('negreg2')!;
    backend.active = { fast: true, quality: true };
    await registry.refreshCapabilities();
    expect(room.state.negativeActive.fast).toBe(true);
  });
});
