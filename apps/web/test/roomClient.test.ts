import { createCanvas } from '@napi-rs/canvas';
import { describe, expect, it, vi } from 'vitest';
import { CANVAS_SIZE, type Layer, type RoomSnapshot, type ServerMessage } from '@brushjam/shared';
import { RoomClient, type ClientDeps } from '../src/roomClient.js';

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
    createRaster: () => createCanvas(CANVAS_SIZE, CANVAS_SIZE) as unknown as HTMLCanvasElement,
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
