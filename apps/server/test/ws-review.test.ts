import type { AddressInfo } from 'node:net';
import { createCanvas } from '@napi-rs/canvas';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import type { ServerMessage } from '@brushjam/shared';
import { MockBackend } from '../src/ai/backends/index.js';
import { loadConfig } from '../src/config.js';
import { createBrushJamServer, type BrushJamServer } from '../src/server.js';

let app: BrushJamServer;
let port = 0;

const config = loadConfig({
  AI_BACKEND: 'mock',
  AI_DEBOUNCE_MS: '100000',
  WEB_DIST: 'nonexistent-dir',
  ROOM_IDLE_MS: '10000',
} as NodeJS.ProcessEnv);

beforeEach(async () => {
  app = createBrushJamServer(config, new MockBackend(0));
  await new Promise<void>((resolve) => app.server.listen(0, '127.0.0.1', resolve));
  port = (app.server.address() as AddressInfo).port;
});

afterEach(async () => {
  await app.close();
});

class Client {
  readonly received: ServerMessage[] = [];
  closed = false;
  private constructor(readonly socket: WebSocket) {}

  static async connect(roomId: string, name: string, token?: string): Promise<Client> {
    const socket = new WebSocket(`ws://127.0.0.1:${port}/ws/rooms/${roomId}?name=${name}${token ? `&token=${token}` : ''}`);
    const client = new Client(socket);
    socket.on('message', (data) => client.received.push(JSON.parse(data.toString()) as ServerMessage));
    socket.on('close', () => {
      client.closed = true;
    });
    await new Promise((resolve, reject) => {
      socket.once('open', resolve);
      socket.once('error', reject);
    });
    await client.waitFor('snapshot');
    return client;
  }

  sendRaw(data: string): void {
    this.socket.send(data);
  }

  send(msg: unknown): void {
    this.socket.send(JSON.stringify(msg));
  }

  async waitFor<T extends ServerMessage['t']>(type: T, timeoutMs = 3000): Promise<Extract<ServerMessage, { t: T }>> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const hit = this.received.find((m) => m.t === type);
      if (hit) return hit as Extract<ServerMessage, { t: T }>;
      if (Date.now() > deadline) throw new Error(`timed out waiting for ${type}; got ${this.received.map((m) => m.t).join(',')}`);
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  get userId(): string {
    const snap = this.received.find((m) => m.t === 'snapshot');
    return snap && snap.t === 'snapshot' ? snap.snapshot.youUserId : '';
  }

  clear(): void {
    this.received.length = 0;
  }

  close(): void {
    this.socket.close();
  }
}

/** Finding 1: malformed frames answer with an error and never kill the process. */
describe('malformed messages', () => {
  const payloads = [
    '{"t":"layer_create"}',
    '{"t":"layer_create","layer":null}',
    '{"t":"layer_create","layer":"draw"}',
    '{"t":"layer_update","id":"x"}',
    '{"t":"layer_update","id":"x","patch":null}',
    '{"t":"stroke_start"}',
    '{"t":"stroke_start","stroke":{}}',
    '{"t":"stroke_chunk","strokeId":"a","points":"lots"}',
    '{"t":"cursor"}',
    '{"t":"layer_reorder"}',
    '{"t":"unknown_thing"}',
    'not json at all',
    'null',
    '[]',
    '42',
  ];

  it('answers every malformed payload with an error and keeps serving', async () => {
    const alice = await Client.connect('crashme', 'Alice');
    for (const payload of payloads) {
      alice.clear();
      alice.sendRaw(payload);
      const error = await alice.waitFor('error');
      expect(error.message).toBeTruthy();
    }

    // still alive: a normal action works afterwards
    alice.clear();
    alice.send({ t: 'set_prompt', prompt: 'still here' });
    expect((await alice.waitFor('prompt_changed')).prompt).toBe('still here');

    const health = (await (await fetch(`http://127.0.0.1:${port}/healthz`)).json()) as { ok: boolean };
    expect(health.ok).toBe(true);
    alice.close();
  });

  it('keeps other clients in the room working after a bad frame', async () => {
    const alice = await Client.connect('crashtwo', 'Alice');
    const bob = await Client.connect('crashtwo', 'Bob');
    alice.sendRaw('{"t":"layer_create","layer":null}');
    await alice.waitFor('error');

    bob.clear();
    alice.send({ t: 'set_prompt', prompt: 'unaffected' });
    expect((await bob.waitFor('prompt_changed')).prompt).toBe('unaffected');
    alice.close();
    bob.close();
  });
});

/** Finding 11: a reconnect with the same token keeps the identity and the undo stack. */
describe('reconnect identity', () => {
  it('resumes the same userId and can undo pre-disconnect strokes', async () => {
    const first = await Client.connect('resumeroom', 'Alice', 'tok-abcdefgh');
    const layerId = (await first.waitFor('snapshot')).snapshot.layers[0]!.id;
    const originalUserId = first.userId;

    first.send({ t: 'stroke_start', stroke: { id: 's1', layerId, tool: 'pen', color: '#ff0000', width: 10, points: [{ x: 10, y: 10 }] } });
    first.send({ t: 'stroke_end', strokeId: 's1', points: [{ x: 40, y: 40 }] });
    await first.waitFor('stroke_committed');
    first.close();
    await new Promise((r) => setTimeout(r, 50));

    const second = await Client.connect('resumeroom', 'Alice', 'tok-abcdefgh');
    expect(second.userId).toBe(originalUserId);
    second.clear();
    second.send({ t: 'undo' });
    expect((await second.waitFor('undo_applied')).strokeId).toBe(`${originalUserId}:s1`);
    second.close();
  });

  it('gives a fresh identity without a token', async () => {
    const first = await Client.connect('freshroom', 'Alice');
    const firstId = first.userId;
    first.close();
    await new Promise((r) => setTimeout(r, 50));
    const second = await Client.connect('freshroom', 'Alice');
    expect(second.userId).not.toBe(firstId);
    second.close();
  });
});

/** Finding 14: a disconnect mid-stroke tells everyone else to drop the ghost. */
describe('stroke cancellation', () => {
  it('broadcasts stroke_cancel when the author disconnects mid-stroke', async () => {
    const alice = await Client.connect('cancelroom', 'Alice');
    const bob = await Client.connect('cancelroom', 'Bob');
    const layerId = (await alice.waitFor('snapshot')).snapshot.layers[0]!.id;

    alice.send({ t: 'stroke_start', stroke: { id: 'ghost', layerId, tool: 'pen', color: '#00ff00', width: 10, points: [{ x: 5, y: 5 }] } });
    await bob.waitFor('stroke_start');
    bob.clear();
    alice.close();

    const cancel = await bob.waitFor('stroke_cancel');
    expect(cancel.strokeId).toContain('ghost');
    bob.close();
  });
});

/** Findings 2 and 3: uploads are validated before a room is allocated. */
describe('upload guards', () => {
  const png = (w: number, h: number): Buffer => {
    const canvas = createCanvas(w, h);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ff00ff';
    ctx.fillRect(0, 0, w, h);
    return canvas.toBuffer('image/png');
  };

  it('rejects an unsupported content type without creating the room', async () => {
    const before = (await (await fetch(`http://127.0.0.1:${port}/healthz`)).json()) as { rooms: number };
    const res = await fetch(`http://127.0.0.1:${port}/rooms/newroomx/images`, {
      method: 'POST',
      headers: { 'content-type': 'application/pdf' },
      body: Buffer.from('%PDF'),
    });
    expect(res.status).toBe(415);
    const after = (await (await fetch(`http://127.0.0.1:${port}/healthz`)).json()) as { rooms: number };
    expect(after.rooms).toBe(before.rooms);
  });

  it('rejects a header that claims impossible dimensions', async () => {
    const bomb = png(1, 1);
    bomb.writeUInt32BE(50_000, 16);
    bomb.writeUInt32BE(50_000, 20);
    const res = await fetch(`http://127.0.0.1:${port}/rooms/bombroom/images`, {
      method: 'POST',
      headers: { 'content-type': 'image/png' },
      body: bomb,
    });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toMatch(/larger than/);
  });

  it('accepts a valid image', async () => {
    const res = await fetch(`http://127.0.0.1:${port}/rooms/goodroom/images`, {
      method: 'POST',
      headers: { 'content-type': 'image/png' },
      body: png(32, 16),
    });
    expect(await res.json()).toMatchObject({ width: 32, height: 16 });
  });
});

/** Finding 2: the 64 MiB AI raster is not allocated until there is a result. */
describe('room lifecycle', () => {
  it('does not serve (or allocate) an AI canvas before the first result', async () => {
    const alice = await Client.connect('lazyroom', 'Alice');
    const res = await fetch(`http://127.0.0.1:${port}/rooms/lazyroom/ai.png`);
    expect(res.status).toBe(404);
    alice.close();
  });

  it('evicts idle empty rooms', async () => {
    const alice = await Client.connect('idleroom', 'Alice');
    alice.close();
    await new Promise((r) => setTimeout(r, 50));
    expect(app.registry.get('idleroom')).toBeDefined();

    // ROOM_IDLE_MS is 10s in this suite; sweep with a clock far in the future
    const removed = app.registry.sweep(Date.now() + 60_000);
    expect(removed).toBeGreaterThan(0);
    expect(app.registry.get('idleroom')).toBeUndefined();
  });

  it('never evicts a room that still has members', async () => {
    const alice = await Client.connect('busyroom', 'Alice');
    expect(app.registry.sweep(Date.now() + 3_600_000)).toBe(0);
    expect(app.registry.get('busyroom')).toBeDefined();
    alice.close();
  });
});

/**
 * Regression: React StrictMode connects, disposes and reconnects with the same
 * session token, so the superseded socket's close arrives *after* the
 * replacement has joined. It must not evict the participant.
 */
describe('superseded sockets', () => {
  it('keeps the member when an old socket for the same identity closes late', async () => {
    const first = await Client.connect('strict1', 'Alice', 'tokenstrictmodeaaaa');
    const second = await Client.connect('strict1', 'Alice', 'tokenstrictmodeaaaa');
    expect(second.userId).toBe(first.userId);

    // the first socket closes only now, well after the replacement joined
    first.close();
    await new Promise((r) => setTimeout(r, 150));

    const observer = await Client.connect('strict1', 'Bob');
    expect(observer.received.find((m) => m.t === 'snapshot')).toBeDefined();
    const presence = await observer.waitFor('snapshot');
    expect(presence.snapshot.members.map((m) => m.name).sort()).toEqual(['Alice', 'Bob']);

    // and the surviving socket is still usable
    second.clear();
    second.send({ t: 'set_prompt', prompt: 'still here' });
    expect((await second.waitFor('prompt_changed')).prompt).toBe('still here');
    second.close();
    observer.close();
  });

  it('still removes the member when the live socket closes', async () => {
    const alice = await Client.connect('strict2', 'Alice', 'tokenstrictmodebbbb');
    const bob = await Client.connect('strict2', 'Bob');
    bob.clear();
    alice.close();
    const presence = await bob.waitFor('presence');
    expect(presence.members.map((m) => m.name)).toEqual(['Bob']);
    bob.close();
  });
});

/** Moving a draw layer reaches everyone else. */
describe('draw layer offsets over the wire', () => {
  it('broadcasts the offset and shows it in a later snapshot', async () => {
    const alice = await Client.connect('moveroom', 'Alice');
    const snap = await alice.waitFor('snapshot');
    const layerId = snap.snapshot.layers[0]!.id;
    const bob = await Client.connect('moveroom', 'Bob');
    bob.clear();

    alice.send({ t: 'layer_update', id: layerId, patch: { offsetX: 240, offsetY: -60 } });
    const updated = await bob.waitFor('layer_updated');
    expect(updated.layer).toMatchObject({ id: layerId, offsetX: 240, offsetY: -60 });

    const late = await Client.connect('moveroom', 'Carol');
    expect((await late.waitFor('snapshot')).snapshot.layers[0]).toMatchObject({ offsetX: 240, offsetY: -60 });
    alice.close();
    bob.close();
    late.close();
  });
});
