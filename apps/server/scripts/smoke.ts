/**
 * End-to-end smoke test: boots the room server, joins over WebSocket, draws a
 * stroke, and waits for a real AI patch. Run with `pnpm --filter @brushjam/server smoke`.
 * Uses whatever backend the config selects (ComfyUI when it is reachable).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import WebSocket from 'ws';
import type { ServerMessage } from '@brushjam/shared';
import { createBackend } from '../src/ai/backends/index.js';
import { loadConfig } from '../src/config.js';
import { createBrushJamServer } from '../src/server.js';

const config = loadConfig();
const backend = await createBackend(config);
const app = createBrushJamServer(config, backend);
await new Promise<void>((resolve) => app.server.listen(0, '127.0.0.1', resolve));
const port = (app.server.address() as AddressInfo).port;
console.log(`[smoke] server on ${port}, backend ${backend.name}, window ${config.aiWindow}`);

const socket = new WebSocket(`ws://127.0.0.1:${port}/ws/rooms/smoketest?name=Smoke`);
const started = Date.now();
let layerId = '';

socket.on('message', (raw) => {
  const msg = JSON.parse(raw.toString()) as ServerMessage;
  if (msg.t === 'snapshot') {
    layerId = msg.snapshot.layers[0]!.id;
    socket.send(JSON.stringify({ t: 'set_prompt', prompt: 'anime style, fantasy town, vibrant colors' }));
    const points = Array.from({ length: 40 }, (_, i) => ({ x: 1800 + i * 12, y: 2000 + Math.sin(i / 3) * 180 }));
    socket.send(JSON.stringify({ t: 'stroke_start', stroke: { id: 'smoke1', layerId, tool: 'pen', color: '#2244cc', width: 28, points: points.slice(0, 1) } }));
    socket.send(JSON.stringify({ t: 'stroke_end', strokeId: 'smoke1', points: points.slice(1) }));
    console.log('[smoke] stroke sent, waiting for AI...');
  }
  if (msg.t === 'ai_status') console.log(`[smoke] ai_status ${msg.state}${msg.message ? `: ${msg.message}` : ''}`);
  if (msg.t === 'ai_result') {
    console.log(`[smoke] ai_result rev=${msg.aiRevision} latency=${msg.latencyMs}ms total=${Date.now() - started}ms crop=${JSON.stringify(msg.crop)}`);
    void fetch(`http://127.0.0.1:${port}${msg.url}`)
      .then(async (r) => Buffer.from(await r.arrayBuffer()))
      .then((buf) => {
        const out = path.resolve('smoke-out');
        mkdirSync(out, { recursive: true });
        writeFileSync(path.join(out, 'patch.png'), buf);
        console.log(`[smoke] patch saved (${buf.length} bytes) to ${path.join(out, 'patch.png')}`);
        socket.close();
        void app.close().then(() => process.exit(0));
      });
  }
});

setTimeout(() => {
  console.error('[smoke] timed out after 240s');
  process.exit(1);
}, 240_000);
