import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it, vi } from 'vitest';
import { CANVAS_SIZE, type Layer, type RoomSnapshot, type ServerMessage } from '@brushjam/shared';
import { setScratchCanvasFactory } from '../src/raster.js';
import { RoomClient, RECONNECT_MS, type ClientDeps, type SocketLike } from '../src/roomClient.js';

class FakeSocket implements SocketLike {
  readyState = 0;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
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
    canvasSize: CANVAS_SIZE,
    aiWindow: 1024,
    aiApply: 768,
    denoise: 0.55,
    negativePrompt: '',
    aiResolution: 1024,
    aiResolutionMax: 1024,
    members: [{ userId: 'me', name: 'Me', color: '#fff' }],
    layers: [layer('l1')],
    strokes: [],
    undone: [],
    aiState: 'idle',
    ...extra,
  },
});

const rect = { x: 0, y: 0, width: 64, height: 64 };
const aiResult = (n: number): ServerMessage => ({
  t: 'ai_result',
  rect,
  url: `/patch-${n}.png`,
  aiRevision: n,
  crop: rect,
  apply: rect,
  latencyMs: 10,
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
    client.receive(snapshot({ aiRevision: 5 }));
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
    client.receive(snapshot({ aiRevision: 4 }));
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

    client.receive({ t: 'ai_settings_changed', denoise: 0.35, negativePrompt: '', aiResolution: 768 });
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
