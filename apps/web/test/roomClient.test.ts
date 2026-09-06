import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it, vi } from 'vitest';
import { CANVAS_SIZE, CLOSE_CAPACITY, CLOSE_SUPERSEDED, type Layer, type RoomSnapshot, type ServerMessage } from '@brushjam/shared';
import { setScratchCanvasFactory } from '../src/raster.js';
import { CAPACITY_RECONNECT_MS, RoomClient, RECONNECT_MS, type ClientDeps, type SocketLike } from '../src/roomClient.js';

class FakeSocket implements SocketLike {
  readyState = 0;
  onopen: (() => void) | null = null;
  onclose: ((event?: { code?: number }) => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  readonly sent: string[] = [];
  constructor(readonly url: string) {}
  open(): void {
    this.readyState = 1;
    this.onopen?.();
  }
  close(): void {
    this.closed = true;
    this.readyState = 3;
    this.onclose?.();
  }
  send(data: string): void {
    this.sent.push(data);
  }
}

function sockets(): { deps: Pick<ClientDeps, 'openSocket'>; all: FakeSocket[] } {
  const all: FakeSocket[] = [];
  return {
    all,
    deps: {
      openSocket: (url) => {
        const s = new FakeSocket(url);
        all.push(s);
        return s;
      },
    },
  };
}

setScratchCanvasFactory((w, h) => createCanvas(w, h) as unknown as HTMLCanvasElement);

const tick = (ms = 0): Promise<void> => new Promise((r) => setTimeout(r, ms));

interface Loader {
  deps: ClientDeps;
  /** Resolve the pending load for `src`. */
  resolve(src: string): void;
  reject(src: string): void;
  pending(): string[];
  requested: string[];
  aborted: string[];
}

function makeLoader(): Loader {
  const waiters = new Map<string, { resolve: () => void; reject: () => void }>();
  const requested: string[] = [];
  const aborted: string[] = [];
  const deps: ClientDeps = {
    createRaster: (size = CANVAS_SIZE) => createCanvas(size, size) as unknown as HTMLCanvasElement,
    openSocket: (url) => new FakeSocket(url),
    loadImage: (src, options) =>
      new Promise((resolve, reject) => {
        requested.push(src);
        const canvas = createCanvas(8, 8);
        const cctx = canvas.getContext('2d');
        cctx.fillStyle = '#ff8800';
        cctx.fillRect(0, 0, 8, 8);
        const image = canvas as unknown as HTMLImageElement;
        waiters.set(src, { resolve: () => resolve(image), reject: () => reject(new Error('failed')) });
        options?.signal?.addEventListener('abort', () => {
          aborted.push(src);
          reject(new Error('cancelled'));
        });
      }),
  };
  const find = (needle: string): string | undefined => [...waiters.keys()].find((k) => k.includes(needle));
  return {
    deps,
    requested,
    aborted,
    pending: () => [...waiters.keys()],
    resolve: (needle) => {
      const key = find(needle)!;
      waiters.get(key)!.resolve();
      waiters.delete(key);
    },
    reject: (needle) => {
      const key = find(needle)!;
      waiters.get(key)!.reject();
      waiters.delete(key);
    },
  };
}

const layer = (id: string, extra: Partial<Layer> = {}): Layer => ({
  id,
  name: id,
  kind: 'draw',
  visible: true,
  locked: false,
  opacity: 1,
  order: 0,
  includeInAI: true,
  ...extra,
});

const snapshot = (extra: Partial<RoomSnapshot> = {}): ServerMessage => ({
  t: 'snapshot',
  snapshot: {
    roomId: 'r1',
    youUserId: 'me',
    prompt: 'p',
    humanRevision: 0,
    aiRevision: 0,
    aiGeneration: 0,
    negativePromptActive: true,
    canvasSize: CANVAS_SIZE,
    aiWindow: 1024,
    aiApply: 768,
    denoise: 0.55,
    seed: 1234,
    negativePrompt: '',
    aiResolution: 1024,
    aiResolutionMax: 1024,
    aiResolutionAdjustable: true,
    aiProfile: 'fast' as const,
    aiProfiles: ['fast', 'quality'] as const,
    maxDenoise: 0.95,
    members: [{ userId: 'me', name: 'Me', color: '#fff' }],
    layers: [layer('l1')],
    strokes: [],
    undone: [],
    aiState: 'idle',
    ...extra,
  },
});

const rect = { x: 0, y: 0, width: 64, height: 64 };
const aiResult = (n: number, latencyMs = 10, profile: 'fast' | 'quality' = 'fast'): ServerMessage => ({
  t: 'ai_result',
  aiGeneration: n,
  profile,
  rect,
  url: `/patch-${n}.png`,
  aiRevision: n,
  crop: rect,
  apply: rect,
  latencyMs,
});

/** Finding B7: a snapshot must wipe the previous room's AI pixels. */
describe('snapshot resets AI state', () => {
  it('clears the canvas, crop and apply rect every time', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot());
    await tick();

    client.receive(aiResult(3));
    await tick();
    loader.resolve('/patch-3.png');
    await tick();
    expect(client.lastCrop).not.toBeNull();
    expect(client.lastApply).not.toBeNull();
    expect(client.aiRevision).toBe(3);

    const ctx = (client.aiCanvas as unknown as ReturnType<typeof createCanvas>).getContext('2d');
    expect(ctx.getImageData(1, 1, 1, 1).data[3]).toBeGreaterThan(0);

    // room restarted / recreated: revision back to zero
    client.receive(snapshot());
    await tick();
    expect(client.lastCrop).toBeNull();
    expect(client.lastApply).toBeNull();
    expect(ctx.getImageData(1, 1, 1, 1).data[3]).toBe(0);
    // and nothing was fetched, because there is no AI output to fetch
    expect(loader.requested.some((s) => s.includes('ai.png'))).toBe(false);
    client.dispose();
  });

  it('fetches ai.png when the snapshot does have output', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiRevision: 5, aiGeneration: 2 }));
    await tick();
    expect(loader.pending().some((s) => s.includes('ai.png'))).toBe(true);
    loader.resolve('ai.png');
    client.dispose();
  });
});

/** Finding B6: a stalled asset must not freeze the ordered message queue. */
describe('asset loads never block the queue', () => {
  it('keeps applying messages while a reference image is still loading', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot());
    await tick();

    client.receive({
      t: 'layer_created',
      layer: layer('ref1', { kind: 'reference', imageId: 'img1', includeInAI: false, order: 1 }),
      humanRevision: 1,
    });
    client.receive({ t: 'prompt_changed', prompt: 'still flowing' });
    client.receive({ t: 'presence', members: [{ userId: 'me', name: 'Me', color: '#fff' }] });
    await tick();

    // the image request is outstanding...
    expect(loader.pending().some((s) => s.includes('img1'))).toBe(true);
    // ...and the queue kept going anyway
    expect(client.prompt).toBe('still flowing');
    expect(client.layers).toHaveLength(2);
    client.dispose();
  });

  it('recovers from a failed AI patch by refreshing ai.png', async () => {
    vi.useFakeTimers();
    try {
      const loader = makeLoader();
      const client = new RoomClient('r1', 'Me', loader.deps);
      client.receive(snapshot());
      await vi.advanceTimersByTimeAsync(0);

      client.receive(aiResult(2));
      await vi.advanceTimersByTimeAsync(0);
      loader.reject('/patch-2.png');
      await vi.advanceTimersByTimeAsync(0);

      // the failed patch did not advance the AI revision
      expect(client.aiRevision).toBe(0);
      await vi.advanceTimersByTimeAsync(1100);
      expect(loader.requested.some((s) => s.includes('ai.png'))).toBe(true);
      client.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('applies patches in order even when the first is slower', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot());
    await tick();

    client.receive(aiResult(1));
    client.receive(aiResult(2));
    await tick();

    // only the first patch is being fetched: the queue is strictly ordered
    expect(loader.pending().filter((s) => s.startsWith('/patch')).length).toBe(1);
    loader.resolve('/patch-1.png');
    await tick();
    expect(client.aiRevision).toBe(1);
    loader.resolve('/patch-2.png');
    await tick();
    expect(client.aiRevision).toBe(2);
    client.dispose();
  });

  it('cancels outstanding loads on dispose', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot());
    await tick();
    client.receive({
      t: 'layer_created',
      layer: layer('ref1', { kind: 'reference', imageId: 'img9', order: 1 }),
      humanRevision: 1,
    });
    await tick();
    expect(loader.pending().some((s) => s.includes('img9'))).toBe(true);

    client.dispose();
    await tick();
    expect(loader.aborted.some((s) => s.includes('img9'))).toBe(true);
  });
});

/** Finding 14 (client half): ghosts are dropped. */
describe('live stroke cleanup', () => {
  it('drops a cancelled stroke and strokes from members who left', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ members: [{ userId: 'me', name: 'Me', color: '#fff' }, { userId: 'bob', name: 'Bob', color: '#0f0' }] }));
    await tick();

    const stroke = { id: 'bob:s1', layerId: 'l1', tool: 'pen' as const, color: '#000000', width: 4, points: [{ x: 1, y: 1 }] };
    client.receive({ t: 'stroke_start', userId: 'bob', stroke });
    client.receive({ t: 'stroke_start', userId: 'bob', stroke: { ...stroke, id: 'bob:s2' } });
    await tick();
    expect(client.live.size).toBe(2);

    client.receive({ t: 'stroke_cancel', userId: 'bob', strokeId: 'bob:s1', reason: 'layer removed' });
    await tick();
    expect(client.live.size).toBe(1);

    client.receive({ t: 'presence', members: [{ userId: 'me', name: 'Me', color: '#fff' }] });
    await tick();
    expect(client.live.size).toBe(0);
    client.dispose();
  });
});

/** Regression: StrictMode mounts, unmounts and remounts the effect. */
describe('connection lifecycle', () => {
  it('reconnects after dispose, as StrictMode double-mounting requires', () => {
    const { deps, all } = sockets();
    const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });

    client.connect();
    all[0]!.open();
    expect(client.connected).toBe(true);

    client.dispose();
    expect(all[0]!.closed).toBe(true);
    expect(client.connected).toBe(false);

    client.connect();
    expect(all).toHaveLength(2);
    all[1]!.open();
    expect(client.connected).toBe(true);

    // and the live socket is the new one
    client.send({ t: 'set_prompt', prompt: 'hi' });
    expect(all[1]!.sent).toHaveLength(1);
    expect(all[0]!.sent).toHaveLength(0);
    client.dispose();
  });

  it('does not schedule a reconnect once disposed', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = sockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      all[0]!.open();
      client.dispose();

      // the closing socket's onclose must not resurrect the connection
      vi.advanceTimersByTime(10 * RECONNECT_MS);
      expect(all).toHaveLength(1);
      expect(client.connected).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('reconnects on its own when the server drops the socket', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = sockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      all[0]!.open();
      all[0]!.close();
      expect(client.connected).toBe(false);

      vi.advanceTimersByTime(RECONNECT_MS + 10);
      expect(all).toHaveLength(2);
      all[1]!.open();
      expect(client.connected).toBe(true);
      client.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('drops a stale socket instead of letting it clear the new connection', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = sockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      const stale = all[0]!;
      client.connect(); // remount before the first socket ever opened
      all[1]!.open();
      expect(client.connected).toBe(true);

      stale.readyState = 3;
      stale.onclose?.(); // arrives late; already detached, so it is a no-op
      vi.advanceTimersByTime(5 * RECONNECT_MS);
      expect(client.connected).toBe(true);
      expect(all).toHaveLength(2);
      client.dispose();
    } finally {
      vi.useRealTimers();
    }
  });
});

/** Round 3, finding 3: a late full raster must not undo a newer patch. */
describe('stale ai.png loads', () => {
  it('discards a full raster that lost the race to a newer patch', async () => {
    vi.useFakeTimers();
    try {
      const loader = makeLoader();
      const client = new RoomClient('r1', 'Me', loader.deps);
      client.receive(snapshot());
      await vi.advanceTimersByTimeAsync(0);

      // a patch fails, so a full refresh is scheduled...
      client.receive(aiResult(1));
      await vi.advanceTimersByTimeAsync(0);
      loader.reject('/patch-1.png');
      await vi.advanceTimersByTimeAsync(1100);
      expect(loader.pending().some((s) => s.includes('ai.png'))).toBe(true);

      // ...but a newer patch lands and is painted while it is still in flight
      client.receive(aiResult(7));
      await vi.advanceTimersByTimeAsync(0);
      loader.resolve('/patch-7.png');
      await vi.advanceTimersByTimeAsync(0);
      expect(client.aiRevision).toBe(7);

      const ctx = (client.aiCanvas as unknown as ReturnType<typeof createCanvas>).getContext('2d');
      // paint a marker inside the patch rect so we can see if it is wiped
      ctx.fillStyle = '#00ff00';
      ctx.fillRect(0, 0, 4, 4);

      loader.resolve('ai.png');
      await vi.advanceTimersByTimeAsync(10);
      const px = ctx.getImageData(1, 1, 1, 1).data;
      expect([px[0], px[1], px[2]]).toEqual([0, 255, 0]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('still applies a full raster when nothing newer was painted', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiRevision: 4, aiGeneration: 1 }));
    await tick();
    loader.resolve('ai.png');
    await tick();
    const ctx = (client.aiCanvas as unknown as ReturnType<typeof createCanvas>).getContext('2d');
    expect(ctx.getImageData(1, 1, 1, 1).data[3]).toBeGreaterThan(0);
    client.dispose();
  });
});

/** Feature: room-level AI settings arrive like any other shared state. */
describe('ai settings', () => {
  it('takes them from the snapshot and from later broadcasts', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ denoise: 0.7, negativePrompt: 'no text', aiResolution: 512, aiResolutionMax: 768 }));
    await tick();
    expect(client.denoise).toBe(0.7);
    expect(client.aiResolution).toBe(512);
    expect(client.aiResolutionMax).toBe(768);
    expect(client.negativePrompt).toBe('no text');

    client.receive({ t: 'ai_settings_changed', denoise: 0.35, negativePrompt: '', aiResolution: 768, aiProfile: 'fast', negativePromptActive: true, seed: 1234 });
    await tick();
    expect(client.denoise).toBe(0.35);
    expect(client.negativePrompt).toBe('');
    expect(client.aiResolution).toBe(768);
    client.dispose();
  });
});

/** Full-canvas mode: the world size comes from the server, never a constant. */
describe('canvas size', () => {
  it('adopts the snapshot canvas size and resizes the AI raster', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ canvasSize: 1024 }));
    await tick();
    expect(client.canvasSize).toBe(1024);
    expect(client.aiCanvas.width).toBe(1024);
    expect(client.aiCanvas.height).toBe(1024);
    // layer rasters follow the new size
    expect(client.layerCanvas('l1').width).toBe(1024);
    client.dispose();
  });
});

/** Noise previews are incremental and disappear with their stroke. */
describe('live stroke previews', () => {
  const noiseStart = (id: string): ServerMessage => ({
    t: 'stroke_start',
    userId: 'bob',
    stroke: { id, layerId: 'l1', tool: 'noise', color: '#000000', width: 16, points: [{ x: 10, y: 10 }] },
  });

  it('extends the raster as points arrive instead of redrawing it', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ canvasSize: 256, members: [{ userId: 'bob', name: 'Bob', color: '#0f0' }] }));
    await tick();
    client.receive(noiseStart('bob:n1'));
    await tick();

    const first = client.previewRaster('bob:n1');
    expect(first).not.toBeNull();
    expect(first!.width).toBe(256);
    const ctx = (first as unknown as ReturnType<typeof createCanvas>).getContext('2d');
    const painted = (): number => {
      const d = ctx.getImageData(0, 0, 256, 256).data;
      let n = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i + 3]! > 0) n += 1;
      return n;
    };
    const afterStart = painted();
    expect(afterStart).toBeGreaterThan(0);

    client.receive({ t: 'stroke_chunk', userId: 'bob', strokeId: 'bob:n1', points: [{ x: 120, y: 120 }] });
    await tick();
    // the same raster object grows; it is not reallocated
    expect(client.previewRaster('bob:n1')).toBe(first);
    expect(painted()).toBeGreaterThan(afterStart);
    client.dispose();
  });

  it('forgets the raster when the stroke commits or is cancelled', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ canvasSize: 256, members: [{ userId: 'bob', name: 'Bob', color: '#0f0' }] }));
    await tick();

    client.receive(noiseStart('bob:n1'));
    await tick();
    const first = client.previewRaster('bob:n1');
    client.receive({ t: 'stroke_cancel', userId: 'bob', strokeId: 'bob:n1', reason: 'gone' });
    await tick();
    expect(client.previewRaster('bob:n1')).toBeNull();

    // a new stroke with the same id gets a fresh raster
    client.receive(noiseStart('bob:n1'));
    await tick();
    expect(client.previewRaster('bob:n1')).not.toBe(first);
    client.dispose();
  });
});

/** The fast/quality switch is shared room state like the prompt. */
describe('AI profile', () => {
  const changed = (aiProfile: 'fast' | 'quality'): ServerMessage => ({
    t: 'ai_settings_changed',
    denoise: 0.7,
    negativePrompt: '',
    aiResolution: aiProfile === 'fast' ? 768 : 1024,
    aiProfile,
    negativePromptActive: aiProfile === 'quality',
    seed: 1234,
  });

  it('takes the profile from the snapshot', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiProfile: 'quality' }));
    await tick();
    expect(client.aiProfile).toBe('quality');
  });

  it('follows another user switching it', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiProfile: 'fast' }));
    await tick();
    client.receive(changed('quality'));
    await tick();
    expect(client.aiProfile).toBe('quality');
    expect(client.aiResolution).toBe(1024);
  });

  it('remembers what each profile actually cost on this machine', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiProfile: 'fast' }));
    await tick();

    client.receive(aiResult(1, 3700, 'fast'));
    await tick();
    loader.resolve('/patch-1.png');
    await tick();
    expect(client.profileLatency.fast).toBe(3700);
    expect(client.profileLatency.quality).toBeUndefined();

    client.receive(changed('quality'));
    await tick();
    client.receive(aiResult(2, 10_200, 'quality'));
    await tick();
    loader.resolve('/patch-2.png');
    await tick();
    expect(client.profileLatency.quality).toBe(10_200);
    // and the fast measurement survives the switch
    expect(client.profileLatency.fast).toBe(3700);
  });
});

/** The client mirrors what the backend can do so the UI can follow it. */
describe('backend capabilities', () => {
  it('takes the supported profiles and the denoise ceiling from the snapshot', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiProfiles: ['fast'], maxDenoise: 0.9, aiProfile: 'fast' }));
    await tick();
    expect(client.aiProfiles).toEqual(['fast']);
    expect(client.maxDenoise).toBe(0.9);
  });

  it('defaults to both profiles before a snapshot arrives', () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    expect(client.aiProfiles).toEqual(['fast', 'quality']);
    expect(client.maxDenoise).toBe(0.95);
  });
});

/** Review 7 findings 3 and 8. */
describe('AI result bookkeeping', () => {
  it('fetches the full raster whenever anything has been generated, even at revision 0', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    // a settings-triggered generation in an untouched room: revision 0, but
    // there is a raster waiting.
    client.receive(snapshot({ aiRevision: 0, aiGeneration: 3 }));
    await tick();
    expect(loader.pending().some((u) => u.includes('ai.png'))).toBe(true);
  });

  it('does not fetch anything in a room that has never generated', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiRevision: 0, aiGeneration: 0 }));
    await tick();
    expect(loader.pending().some((u) => u.includes('ai.png'))).toBe(false);
  });

  it('attributes latency to the profile the result was generated with', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ aiProfile: 'fast' }));
    await tick();

    // the room switches to quality while a fast run is still in flight
    client.receive({ t: 'ai_settings_changed', denoise: 0.7, negativePrompt: '', aiResolution: 1024, aiProfile: 'quality', negativePromptActive: true, seed: 1234 });
    await tick();
    client.receive(aiResult(1, 3700, 'fast'));
    await tick();
    loader.resolve('/patch-1.png');
    await tick();

    expect(client.profileLatency.fast).toBe(3700);
    expect(client.profileLatency.quality).toBeUndefined();
  });
});

/** The room tells the client whether the negative prompt reaches the sampler. */
describe('negative prompt activity', () => {
  it('defaults to active', () => {
    expect(new RoomClient('r1', 'Me', makeLoader().deps).negativePromptActive).toBe(true);
  });

  it('takes it from the snapshot', async () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.receive(snapshot({ negativePromptActive: false }));
    await tick();
    expect(client.negativePromptActive).toBe(false);
  });

  it('follows a profile switch broadcast', async () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.receive(snapshot({ negativePromptActive: false }));
    await tick();
    client.receive({
      t: 'ai_settings_changed',
      denoise: 0.7,
      negativePrompt: '',
      aiResolution: 1024,
      aiProfile: 'quality',
      negativePromptActive: true,
      seed: 1234,
    });
    await tick();
    expect(client.negativePromptActive).toBe(true);
  });
});

/**
 * Review 8 finding B1: after a worker restarts smaller, an open client whose
 * controls still offer the old profiles just sends requests the server refuses.
 */
describe('ai_capabilities', () => {
  const caps = (extra: Partial<Extract<ServerMessage, { t: 'ai_capabilities' }>> = {}): ServerMessage => ({
    t: 'ai_capabilities',
    aiProfiles: ['fast'],
    maxDenoise: 0.8,
    aiResolutionMax: 768,
    negativePromptActive: false,
    ...extra,
  });

  it('narrows the controls when the backend loses a profile', async () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.receive(snapshot());
    await tick();
    client.receive(caps());
    await tick();
    expect(client.aiProfiles).toEqual(['fast']);
    expect(client.maxDenoise).toBe(0.8);
    expect(client.aiResolutionMax).toBe(768);
    expect(client.negativePromptActive).toBe(false);
  });

  it('widens them again when the backend comes back bigger', async () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.receive(snapshot());
    await tick();
    client.receive(caps());
    await tick();
    client.receive(caps({ aiProfiles: ['fast', 'quality'], maxDenoise: 0.95, aiResolutionMax: 1024, negativePromptActive: true }));
    await tick();
    expect(client.aiProfiles).toEqual(['fast', 'quality']);
    expect(client.aiResolutionMax).toBe(1024);
  });

  it('notifies subscribers so the panel re-renders', async () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.receive(snapshot());
    await tick();
    let bumps = 0;
    client.subscribe(() => (bumps += 1));
    client.receive(caps());
    await tick();
    expect(bumps).toBeGreaterThan(0);
  });
});

/**
 * Review 10 finding 4: a room where people try several reference photos held
 * every one of them, fully decoded, in every browser for the whole session.
 */
describe('reference image retention', () => {
  const ref = (id: string, imageId: string): Layer => layer(id, { kind: 'reference', imageId, order: 1 });

  it('keeps the image a layer still references', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ layers: [layer('l1'), ref('l2', 'img-a')] }));
    await tick();
    loader.resolve('images/img-a');
    await tick();
    expect(client.images.has('img-a')).toBe(true);
  });

  it('drops the image when its layer is deleted', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ layers: [layer('l1'), ref('l2', 'img-a')] }));
    await tick();
    loader.resolve('images/img-a');
    await tick();
    expect(client.images.has('img-a')).toBe(true);

    client.receive({ t: 'layer_deleted', id: 'l2', humanRevision: 2 });
    await tick();
    expect(client.images.has('img-a')).toBe(false);
  });

  it('drops images no longer referenced after a snapshot replaces the room', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ layers: [layer('l1'), ref('l2', 'img-a')] }));
    await tick();
    loader.resolve('images/img-a');
    await tick();

    // a fresh snapshot in which that reference is gone
    client.receive(snapshot({ layers: [layer('l1')] }));
    await tick();
    expect(client.images.has('img-a')).toBe(false);
    expect(client.images.size).toBe(0);
  });

  it('drops the old image when a layer is repointed at a different one', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ layers: [layer('l1'), ref('l2', 'img-a')] }));
    await tick();
    loader.resolve('images/img-a');
    await tick();

    client.receive({ t: 'layer_updated', layer: ref('l2', 'img-b'), humanRevision: 3 });
    await tick();
    expect(client.images.has('img-a')).toBe(false);
  });

  it('keeps an image two layers share until both are gone', async () => {
    const loader = makeLoader();
    const client = new RoomClient('r1', 'Me', loader.deps);
    client.receive(snapshot({ layers: [ref('l1', 'img-a'), ref('l2', 'img-a')] }));
    await tick();
    loader.resolve('images/img-a');
    await tick();

    client.receive({ t: 'layer_deleted', id: 'l2', humanRevision: 2 });
    await tick();
    expect(client.images.has('img-a')).toBe(true);

    client.receive({ t: 'layer_deleted', id: 'l1', humanRevision: 3 });
    await tick();
    expect(client.images.has('img-a')).toBe(false);
  });
});

/** Review 10 finding 6: a refusal nobody can see may as well not be sent. */
describe('action errors', () => {
  it('starts with nothing to show', () => {
    expect(new RoomClient('r1', 'Me', makeLoader().deps).actionError).toBeNull();
  });

  it('surfaces a server error message as a toast', async () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.receive(snapshot());
    await tick();
    client.receive({ t: 'error', message: 'layer is locked' });
    await tick();
    expect(client.actionError?.message).toBe('layer is locked');
  });

  it('notifies subscribers so the toast appears without another event', async () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.receive(snapshot());
    await tick();
    let bumps = 0;
    client.subscribe(() => (bumps += 1));
    client.noteActionError('paste failed: server said 413');
    expect(bumps).toBeGreaterThan(0);
    expect(client.actionError?.message).toContain('413');
  });

  it('can be cleared, and clearing twice is harmless', () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.noteActionError('boom');
    client.clearActionError();
    expect(client.actionError).toBeNull();
    client.clearActionError();
    expect(client.actionError).toBeNull();
  });

  it('replaces an older error rather than queueing', () => {
    const client = new RoomClient('r1', 'Me', makeLoader().deps);
    client.noteActionError('first');
    client.noteActionError('second');
    expect(client.actionError?.message).toBe('second');
  });
});

/** Astra finding 3: two tabs sharing a session token evicted each other forever. */
describe('close codes', () => {
  class CodedSocket extends FakeSocket {
    closeWith(code: number): void {
      this.readyState = 3;
      this.onclose?.({ code });
    }
  }

  function codedSockets(): { deps: Pick<ClientDeps, 'openSocket'>; all: CodedSocket[] } {
    const all: CodedSocket[] = [];
    return { all, deps: { openSocket: (url) => { const s = new CodedSocket(url); all.push(s); return s; } } };
  }

  it('stops reconnecting when another tab takes the identity', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = codedSockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      all[0]!.open();
      all[0]!.closeWith(CLOSE_SUPERSEDED);

      expect(client.superseded).toBe(true);
      expect(client.connected).toBe(false);
      vi.advanceTimersByTime(60 * RECONNECT_MS);
      expect(all).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('reconnects on demand after being superseded', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = codedSockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      all[0]!.open();
      all[0]!.closeWith(CLOSE_SUPERSEDED);

      client.connect();
      expect(all).toHaveLength(2);
      expect(client.superseded).toBe(false);
      all[1]!.open();
      expect(client.connected).toBe(true);
      client.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('reports whether a message actually went out', () => {
    const { deps, all } = sockets();
    const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
    // Nothing sent before connect(): there is no socket at all.
    expect(client.send({ t: 'undo' })).toBe(false);
    client.connect();
    // ...nor while the handshake is still open.
    expect(client.send({ t: 'set_prompt', prompt: 'a hill' })).toBe(false);
    all[0]!.open();
    expect(client.send({ t: 'set_prompt', prompt: 'a hill' })).toBe(true);
    expect(all[0]!.sent).toEqual([JSON.stringify({ t: 'set_prompt', prompt: 'a hill' })]);
    client.dispose();
    expect(client.send({ t: 'undo' })).toBe(false);
  });

  it('counts a session per snapshot, so the fields know a connection was lost', () => {
    const { deps, all } = sockets();
    const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
    client.connect();
    expect(client.sessionEpoch).toBe(0);
    all[0]!.open();
    all[0]!.onmessage?.({ data: JSON.stringify(snapshot()) });
    return tick().then(() => {
      expect(client.sessionEpoch).toBe(1);
      all[0]!.onmessage?.({ data: JSON.stringify(snapshot({ prompt: 'q' })) });
      return tick().then(() => {
        expect(client.sessionEpoch).toBe(2);
        client.dispose();
      });
    });
  });

  it('backs off when the server says it is full', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = codedSockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      all[0]!.closeWith(CLOSE_CAPACITY);

      // A full server is not a blip: nothing at one second, a retry at five.
      vi.advanceTimersByTime(RECONNECT_MS + 10);
      expect(all).toHaveLength(1);
      vi.advanceTimersByTime(CAPACITY_RECONNECT_MS);
      expect(all).toHaveLength(2);

      // Still full: the wait doubles.
      all[1]!.closeWith(CLOSE_CAPACITY);
      vi.advanceTimersByTime(CAPACITY_RECONNECT_MS + 10);
      expect(all).toHaveLength(2);
      vi.advanceTimersByTime(CAPACITY_RECONNECT_MS);
      expect(all).toHaveLength(3);
      client.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('goes back to the one-second retry once a connection succeeds', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = codedSockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      all[0]!.closeWith(CLOSE_CAPACITY);
      vi.advanceTimersByTime(CAPACITY_RECONNECT_MS + 10);
      all[1]!.open();

      all[1]!.closeWith(1006); // a transport failure, not capacity
      vi.advanceTimersByTime(RECONNECT_MS + 10);
      expect(all).toHaveLength(3);
      client.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('still reconnects after a close with no code at all', () => {
    vi.useFakeTimers();
    try {
      const { deps, all } = codedSockets();
      const client = new RoomClient('r1', 'Me', { ...makeLoader().deps, ...deps });
      client.connect();
      all[0]!.open();
      all[0]!.close();
      vi.advanceTimersByTime(RECONNECT_MS + 10);
      expect(all).toHaveLength(2);
      client.dispose();
    } finally {
      vi.useRealTimers();
    }
  });
});
