import type { AddressInfo } from 'node:net';
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
  AI_MODE: 'patch',
  CANVAS_SIZE: '4096',
  AI_DEBOUNCE_MS: '100000',
  WEB_DIST: 'nonexistent-dir',
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
  private constructor(readonly socket: WebSocket) {}

  static async connect(roomId: string, name: string, token?: string): Promise<Client> {
    const query = `name=${name}${token ? `&token=${token}` : ''}`;
    const socket = new WebSocket(`ws://127.0.0.1:${port}/ws/rooms/${roomId}?${query}`);
    const client = new Client(socket);
    socket.on('message', (data) => client.received.push(JSON.parse(data.toString()) as ServerMessage));
    await new Promise((resolve, reject) => {
      socket.once('open', resolve);
      socket.once('error', reject);
    });
    await client.waitFor('snapshot');
    return client;
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

async function stroke(c: Client, layerId: string, id: string, x: number): Promise<void> {
  c.send({ t: 'stroke_start', stroke: { id, layerId, tool: 'pen', color: '#ff0000', width: 12, points: [{ x, y: 100 }] } });
  c.send({ t: 'stroke_chunk', strokeId: id, points: [{ x: x + 10, y: 120 }] });
  c.send({ t: 'stroke_end', strokeId: id, points: [{ x: x + 20, y: 140 }] });
}

describe('websocket room', () => {
  it('creates rooms over HTTP and reports health', async () => {
    const res = await fetch(`http://127.0.0.1:${port}/api/rooms`, { method: 'POST' });
    const { roomId } = (await res.json()) as { roomId: string };
    expect(roomId).toMatch(/^[a-z0-9]{8}$/);
    const health = await (await fetch(`http://127.0.0.1:${port}/healthz`)).json();
    expect(health).toMatchObject({ ok: true, backend: 'mock' });
  });

  it('sends a snapshot on join and presence to everyone', async () => {
    const alice = await Client.connect('roomone', 'Alice');
    const snap = await alice.waitFor('snapshot');
    expect(snap.snapshot.layers).toHaveLength(1);
    expect(snap.snapshot.canvasSize).toBe(4096);
    // finding 17: the client must be told the configured sizes
    expect(snap.snapshot.aiWindow).toBe(1024);
    expect(snap.snapshot.aiApply).toBe(768);

    alice.clear();
    const bob = await Client.connect('roomone', 'Bob');
    const presence = await alice.waitFor('presence');
    expect(presence.members.map((m) => m.name).sort()).toEqual(['Alice', 'Bob']);
    alice.close();
    bob.close();
  });

  it('relays strokes between two clients and commits them once', async () => {
    const alice = await Client.connect('roomtwo', 'Alice');
    const bob = await Client.connect('roomtwo', 'Bob');
    const layerId = (await bob.waitFor('snapshot')).snapshot.layers[0]!.id;
    bob.clear();

    await stroke(alice, layerId, 'a1', 100);
    const start = await bob.waitFor('stroke_start');
    expect(start.userId).toBe(alice.userId);
    expect(await bob.waitFor('stroke_chunk')).toMatchObject({ strokeId: `${alice.userId}:a1` });
    const committed = await bob.waitFor('stroke_committed');
    expect(committed.stroke.id).toBe(`${alice.userId}:a1`);
    expect(committed.stroke.points).toHaveLength(3);
    expect(committed.humanRevision).toBe(1);
    expect(bob.received.filter((m) => m.t === 'stroke_committed')).toHaveLength(1);

    alice.close();
    bob.close();
  });

  it('undo removes the sender own last stroke, not the other user latest', async () => {
    const alice = await Client.connect('roomthree', 'Alice');
    const bob = await Client.connect('roomthree', 'Bob');
    const layerId = (await alice.waitFor('snapshot')).snapshot.layers[0]!.id;

    await stroke(alice, layerId, 'a100', 100);
    await bob.waitFor('stroke_committed');
    await stroke(bob, layerId, 'b101', 300);
    await stroke(alice, layerId, 'a102', 500);
    await bob.waitFor('stroke_committed');
    while (bob.received.filter((m) => m.t === 'stroke_committed').length < 3) {
      await new Promise((r) => setTimeout(r, 10));
    }
    bob.clear();

    alice.send({ t: 'undo' });
    const undone = await bob.waitFor('undo_applied');
    expect(undone.strokeId).toBe(`${alice.userId}:a102`);

    alice.close();
    bob.close();
  });

  it('shares prompt changes and layer operations', async () => {
    const alice = await Client.connect('roomfour', 'Alice');
    const bob = await Client.connect('roomfour', 'Bob');
    bob.clear();

    alice.send({ t: 'set_prompt', prompt: 'a haunted lighthouse' });
    expect((await bob.waitFor('prompt_changed')).prompt).toBe('a haunted lighthouse');

    alice.send({ t: 'layer_create', layer: { kind: 'draw' } });
    const created = await bob.waitFor('layer_created');
    expect(created.layer.kind).toBe('draw');

    alice.send({ t: 'layer_update', id: created.layer.id, patch: { name: 'Sky', opacity: 0.25 } });
    expect((await bob.waitFor('layer_updated')).layer).toMatchObject({ name: 'Sky', opacity: 0.25 });

    alice.close();
    bob.close();
  });

  it('a late joiner receives the full stroke log', async () => {
    const alice = await Client.connect('roomfive', 'Alice');
    const layerId = (await alice.waitFor('snapshot')).snapshot.layers[0]!.id;
    await stroke(alice, layerId, 's1', 200);
    await alice.waitFor('stroke_committed');

    const carol = await Client.connect('roomfive', 'Carol');
    const snap = await carol.waitFor('snapshot');
    expect(snap.snapshot.strokes).toHaveLength(1);
    expect(snap.snapshot.humanRevision).toBe(1);
    expect(snap.snapshot.members).toHaveLength(2);

    alice.close();
    carol.close();
  });

  it('drops a member from presence on disconnect', async () => {
    const alice = await Client.connect('roomsix', 'Alice');
    const bob = await Client.connect('roomsix', 'Bob');
    alice.clear();
    bob.close();
    const deadline = Date.now() + 3000;
    for (;;) {
      const last = [...alice.received].reverse().find((m) => m.t === 'presence');
      if (last && last.t === 'presence' && last.members.length === 1) break;
      if (Date.now() > deadline) throw new Error('presence never shrank');
      await new Promise((r) => setTimeout(r, 10));
    }
    alice.close();
  });

  it('accepts a pasted image and turns it into a reference layer', async () => {
    const alice = await Client.connect('roomsvn', 'Alice');
    const { createCanvas } = await import('@napi-rs/canvas');
    const canvas = createCanvas(64, 32);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(0, 0, 64, 32);
    const bytes = canvas.toBuffer('image/png');

    const res = await fetch(`http://127.0.0.1:${port}/rooms/roomsvn/images`, {
      method: 'POST',
      headers: { 'content-type': 'image/png' },
      body: bytes,
    });
    const stored = (await res.json()) as { imageId: string; width: number; height: number };
    expect(stored).toMatchObject({ width: 64, height: 32 });

    alice.send({ t: 'layer_create', layer: { kind: 'reference', imageId: stored.imageId, x: 10, y: 20 } });
    const created = await alice.waitFor('layer_created');
    expect(created.layer).toMatchObject({ kind: 'reference', includeInAI: false, imageWidth: 64 });

    const fetched = await fetch(`http://127.0.0.1:${port}/rooms/roomsvn/images/${stored.imageId}`);
    expect(fetched.headers.get('content-type')).toBe('image/png');
    alice.close();
  });

  it('rejects a websocket upgrade on a bad room path', async () => {
    const socket = new WebSocket(`ws://127.0.0.1:${port}/ws/rooms/NOPE!!`);
    await expect(new Promise((resolve, reject) => {
      socket.once('open', resolve);
      socket.once('error', reject);
    })).rejects.toBeTruthy();
  });
});
