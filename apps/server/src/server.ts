import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer, type WebSocket } from 'ws';
import type { AIBackend } from './ai/backends/index.js';
import type { Config } from './config.js';
import { RoomRegistry } from './runtime.js';

const ROOM_ID = /^[a-z0-9]{4,16}$/;
const SESSION_TOKEN = /^[A-Za-z0-9_-]{8,64}$/;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
/** One frame is never legitimately larger than this (points are batched, not streamed). */
const MAX_WS_PAYLOAD = 1024 * 1024;

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.json': 'application/json; charset=utf-8',
  '.ico': 'image/x-icon',
};

function json(res: ServerResponse, status: number, body: unknown): void {
  const data = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(data) });
  res.end(data);
}

function png(res: ServerResponse, body: Buffer, cacheSeconds = 0): void {
  res.writeHead(200, {
    'content-type': 'image/png',
    'content-length': body.length,
    'cache-control': cacheSeconds > 0 ? `public, max-age=${cacheSeconds}` : 'no-store',
  });
  res.end(body);
}

function readBody(req: IncomingMessage, limit: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => {
      size += c.length;
      if (size > limit) {
        reject(new Error('payload too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function defaultWebDist(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  return path.resolve(here, '..', '..', 'web', 'dist');
}

export interface BrushJamServer {
  server: Server;
  registry: RoomRegistry;
  close(): Promise<void>;
}

export function createBrushJamServer(config: Config, backend: AIBackend): BrushJamServer {
  const registry = new RoomRegistry(backend, config);
  registry.startSweeper();
  const webDist = config.webDist ?? defaultWebDist();
  const hasWeb = existsSync(path.join(webDist, 'index.html'));

  const server = createServer((req, res) => {
    void handleRequest(req, res).catch((err: unknown) => {
      json(res, 500, { error: err instanceof Error ? err.message : 'internal error' });
    });
  });

  async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', 'http://localhost');
    const parts = url.pathname.split('/').filter(Boolean);

    if (req.method === 'POST' && url.pathname === '/api/rooms') {
      const room = registry.create();
      if (!room) return json(res, 429, { error: 'the server is holding too many rooms right now' });
      json(res, 200, { roomId: room.state.id });
      return;
    }

    if (parts[0] === 'rooms' && parts[1] && ROOM_ID.test(parts[1])) {
      const roomId = parts[1];
      const room = registry.get(roomId);

      if (req.method === 'GET' && parts[2] === 'ai.png' && parts.length === 3) {
        if (!room || !room.hasAi()) return json(res, 404, { error: 'no AI output yet' });
        return png(res, room.aiPng());
      }
      if (req.method === 'GET' && parts[2] === 'patches' && parts[3]) {
        const patch = room?.patch(parts[3].replace(/\.png$/, ''));
        if (!patch) return json(res, 404, { error: 'patch not found' });
        return png(res, patch, 300);
      }
      if (req.method === 'POST' && parts[2] === 'images' && parts.length === 3) {
        const mime = req.headers['content-type'] ?? 'image/png';
        // Validate before touching the registry: `ensure` would allocate a room.
        if (!/^image\/(png|jpeg|webp)$/.test(mime)) return json(res, 415, { error: 'unsupported image type' });
        const declaredLength = Number(req.headers['content-length'] ?? 0);
        if (declaredLength > MAX_IMAGE_BYTES) return json(res, 413, { error: 'image too large' });
        const bytes = await readBody(req, MAX_IMAGE_BYTES);
        const target = registry.ensure(roomId);
        if (!target) return json(res, 429, { error: 'the server is holding too many rooms right now' });
        const stored = await target.addImage(bytes, mime);
        if ('error' in stored) return json(res, 400, stored);
        return json(res, 200, stored);
      }
      if (req.method === 'GET' && parts[2] === 'images' && parts[3]) {
        const stored = room?.state.images.get(parts[3]);
        if (!stored) return json(res, 404, { error: 'image not found' });
        res.writeHead(200, { 'content-type': stored.mime, 'content-length': stored.bytes.length, 'cache-control': 'public, max-age=3600' });
        res.end(stored.bytes);
        return;
      }
    }

    if (req.method === 'GET' && url.pathname === '/healthz') {
      return json(res, 200, { ok: true, backend: backend.name, rooms: registry.size });
    }

    if (hasWeb && req.method === 'GET') {
      const rel = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
      const file = path.resolve(webDist, rel);
      if (file.startsWith(webDist) && existsSync(file) && statSync(file).isFile()) {
        res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' });
        createReadStream(file).pipe(res);
        return;
      }
      // SPA fallback
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      createReadStream(path.join(webDist, 'index.html')).pipe(res);
      return;
    }

    json(res, 404, { error: 'not found' });
  }

  const wss = new WebSocketServer({ noServer: true, maxPayload: MAX_WS_PAYLOAD });

  server.on('upgrade', (req, socket, head) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    const match = /^\/ws\/rooms\/([a-z0-9]{4,16})$/.exec(url.pathname);
    if (!match) {
      socket.destroy();
      return;
    }
    const roomId = match[1]!;
    wss.handleUpgrade(req, socket, head, (ws: WebSocket) => {
      const room = registry.ensure(roomId);
      if (!room) {
        ws.close(1013, 'server is holding too many rooms');
        return;
      }
      const rawToken = url.searchParams.get('token') ?? '';
      const token = SESSION_TOKEN.test(rawToken) ? rawToken : undefined;
      const userId = room.join(ws, url.searchParams.get('name') ?? '', token);
      ws.on('message', (data) => room.handle(userId, data.toString()));
      ws.on('close', () => room.leave(userId, ws));
      ws.on('error', () => room.leave(userId, ws));
    });
  });

  return {
    server,
    registry,
    close: () =>
      new Promise<void>((resolve) => {
        registry.dispose();
        wss.close();
        for (const client of wss.clients) client.terminate();
        // Open ws upgrades and keep-alive HTTP sockets otherwise hold the
        // listener open, which is what made a tsx-watch restart hit EADDRINUSE.
        server.closeAllConnections?.();
        let settled = false;
        const done = (): void => {
          if (settled) return;
          settled = true;
          resolve();
        };
        server.close(done);
        // last resort: never let shutdown hang a watch restart
        setTimeout(done, 1500).unref();
      }),
  };
}
