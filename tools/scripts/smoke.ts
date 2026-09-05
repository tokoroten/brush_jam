/**
 * Is this server alive, and does one stroke come back as a picture?
 *
 *   pnpm smoke -- [--url http://127.0.0.1:8787] [--out smoke-out] [--timeout 240]
 *
 * Runs against an ALREADY RUNNING server, like every other script here. The
 * version this replaced booted a Node server in-process, which tested the
 * process it started rather than the one serving anybody; this one exercises
 * whatever is actually listening, whichever implementation that is.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import WebSocket from 'ws';
import type { ServerMessage } from '@brushjam/shared';

interface Options {
  url: string;
  out: string;
  timeoutMs: number;
  room: string;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    url: process.env.BRUSHJAM_URL ?? 'http://127.0.0.1:8787',
    out: 'smoke-out',
    timeoutMs: 240_000,
    room: `smoke${Math.random().toString(36).slice(2, 7)}`,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === '--url' && value) opts.url = value;
    else if (arg === '--out' && value) opts.out = value;
    else if (arg === '--room' && value) opts.room = value;
    else if (arg === '--timeout' && value) opts.timeoutMs = Math.max(1000, Number(value) * 1000 || 1000);
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
const base = opts.url.replace(/\/+$/, '');

let health: { ok?: boolean; backend?: string } = {};
try {
  const res = await fetch(`${base}/healthz`, { signal: AbortSignal.timeout(3000) });
  health = (await res.json()) as typeof health;
} catch (err) {
  console.error(`[smoke] ${base}/healthz did not answer: ${err instanceof Error ? err.message : String(err)}`);
  console.error('[smoke] start a server first, e.g. `uv run brushjam` in apps/brushjam');
  process.exit(1);
}

console.log(`[smoke] server ${base}, backend ${health.backend ?? 'unknown'}, room ${opts.room}`);

const socket = new WebSocket(`${base.replace(/^http/, 'ws')}/ws/rooms/${opts.room}?name=Smoke`);
const started = Date.now();

const fail = (message: string): never => {
  console.error(`[smoke] ${message}`);
  process.exit(1);
};

const deadline = setTimeout(() => fail(`timed out after ${opts.timeoutMs / 1000}s`), opts.timeoutMs);

socket.on('error', (err) => fail(`socket error: ${err.message}`));

socket.on('message', (raw) => {
  const msg = JSON.parse(raw.toString()) as ServerMessage;

  if (msg.t === 'snapshot') {
    const layer = msg.snapshot.layers.find((l) => l.kind === 'draw');
    if (!layer) fail('the room has no draw layer');
    const size = msg.snapshot.canvasSize;
    // A stroke across the middle of whatever canvas this server is configured
    // for, rather than coordinates from one particular deployment.
    const points = Array.from({ length: 40 }, (_, i) => ({
      x: Math.round(size * 0.2 + i * (size * 0.015)),
      y: Math.round(size * 0.5 + Math.sin(i / 3) * (size * 0.08)),
    }));
    socket.send(JSON.stringify({ t: 'set_prompt', prompt: 'anime style, fantasy town, vibrant colors' }));
    socket.send(
      JSON.stringify({
        t: 'stroke_start',
        stroke: { id: 'smoke1', layerId: layer!.id, tool: 'pen', color: '#2244cc', width: 28, points: points.slice(0, 1) },
      }),
    );
    socket.send(JSON.stringify({ t: 'stroke_end', strokeId: 'smoke1', points: points.slice(1) }));
    console.log('[smoke] stroke sent, waiting for AI...');
    return;
  }

  if (msg.t === 'error') console.error(`[smoke] server error: ${msg.message}`);
  if (msg.t === 'ai_status') {
    console.log(`[smoke] ai_status ${msg.state}${msg.message ? `: ${msg.message}` : ''}`);
    return;
  }

  if (msg.t === 'ai_result') {
    console.log(
      `[smoke] ai_result rev=${msg.aiRevision} latency=${msg.latencyMs}ms total=${Date.now() - started}ms crop=${JSON.stringify(msg.crop)}`,
    );
    void fetch(`${base}${msg.url}`)
      .then(async (r) => Buffer.from(await r.arrayBuffer()))
      .then((buf) => {
        const out = path.resolve(opts.out);
        mkdirSync(out, { recursive: true });
        const file = path.join(out, 'patch.png');
        writeFileSync(file, buf);
        console.log(`[smoke] patch saved (${buf.length} bytes) to ${file}`);
        clearTimeout(deadline);
        socket.close();
        process.exit(0);
      })
      .catch((err) => fail(`could not fetch the result: ${err instanceof Error ? err.message : String(err)}`));
  }
});
