/**
 * Per-edit latency measurement against an ALREADY RUNNING server.
 *
 *   pnpm --filter @brushjam/server latency [--url http://127.0.0.1:8787] [--n 10]
 *
 * Joins a fresh room as a WebSocket client, draws N short strokes one at a
 * time, and times each one from the moment `stroke_end` is sent to the
 * `ai_result` that follows. That is the number a person actually feels: it
 * includes the debounce, the render, the backend, and the compositing - not
 * just the backend's own `latencyMs`, which is reported separately.
 *
 * It never starts a server or a backend of its own, so it measures whatever is
 * really running (the /healthz line says which backend that is).
 */
import WebSocket from 'ws';
import type { ServerMessage } from '@brushjam/shared';

interface Options {
  url: string;
  /** Stream worker to ask about sampling settings, when that is the backend. */
  workerUrl: string;
  count: number;
  room: string;
  timeoutMs: number;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    url: process.env.BRUSHJAM_URL ?? 'http://127.0.0.1:8787',
    workerUrl: process.env.STREAM_URL ?? 'http://127.0.0.1:8790',
    count: 10,
    room: `lat${Math.random().toString(36).slice(2, 8)}`,
    timeoutMs: 300_000,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === '--url' && value) opts.url = value;
    else if ((arg === '--n' || arg === '--count') && value) opts.count = Math.max(1, Number(value) || 1);
    else if (arg === '--room' && value) opts.room = value;
    else if (arg === '--timeout' && value) opts.timeoutMs = Math.max(1000, Number(value) || 1000);
    else if (arg === '--worker' && value) opts.workerUrl = value;
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
const base = opts.url.replace(/\/+$/, '');

interface Health {
  ok?: boolean;
  backend?: string;
  rooms?: number;
}

let health: Health = {};
try {
  const res = await fetch(`${base}/healthz`, { signal: AbortSignal.timeout(3000) });
  health = (await res.json()) as Health;
} catch (err) {
  console.error(`[latency] ${base}/healthz did not answer: ${err instanceof Error ? err.message : String(err)}`);
  console.error('[latency] start a server first, e.g. `pnpm --filter @brushjam/server dev`');
  process.exit(1);
}
/**
 * For a stream backend, say how the worker is actually configured: its steps,
 * guidance, VAE and model decide the number being measured, and none of them
 * come from this server's config.
 */
let workerNote = '';
async function describeWorker(workerUrl: string): Promise<void> {
  if (health.backend !== 'stream') return;
  try {
    const res = await fetch(`${workerUrl.replace(/\/+$/, '')}/healthz`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return;
    const w = (await res.json()) as Record<string, unknown>;
    const parts = ['steps', 'guidance', 'vae', 'model', 'lora']
      .filter((k) => w[k] !== undefined && w[k] !== '')
      .map((k) => `${k} ${String(w[k])}`);
    if (parts.length > 0) workerNote = ` (${parts.join(', ')})`;
  } catch {
    /* the worker is optional information, never a reason to fail the run */
  }
}

await describeWorker(opts.workerUrl);
console.log(
  `[latency] server ${base}, backend ${health.backend ?? 'unknown'}${workerNote}, ${opts.count} edits, room ${opts.room}`,
);

const wsUrl = `${base.replace(/^http/, 'ws')}/ws/rooms/${opts.room}?name=Latency`;
const socket = new WebSocket(wsUrl);

interface Sample {
  edit: number;
  /** stroke_end until the PNG is fetched: what a person waits for. */
  totalMs: number;
  /** stroke_end until the ai_result message, before the image is fetched. */
  notifiedMs: number;
  /** The server's own latencyMs: input render + backend + composite. */
  pipelineMs: number;
  bytes: number;
}

const samples: Sample[] = [];
let layerId = '';
let canvasSize = 1024;
let edit = 0;
let sentAt = 0;
let waiting = false;

const fail = (message: string): never => {
  console.error(`[latency] ${message}`);
  socket.close();
  process.exit(1);
};

const deadline = setTimeout(() => fail(`gave up after ${opts.timeoutMs} ms with ${samples.length}/${opts.count} samples`), opts.timeoutMs);

/** One short stroke, placed so successive edits land in different places. */
function drawNext(): void {
  edit += 1;
  const step = canvasSize / (opts.count + 2);
  const x = Math.round(step * (edit + 0.5));
  const y = Math.round(canvasSize / 2 + Math.sin(edit) * (canvasSize / 6));
  const id = `lat${edit}`;
  socket.send(
    JSON.stringify({
      t: 'stroke_start',
      stroke: { id, layerId, tool: 'pen', color: '#1b1b1b', width: 24, points: [{ x, y }] },
    }),
  );
  const tail = Array.from({ length: 8 }, (_, i) => ({ x: x + i * 6, y: y + i * 9 }));
  sentAt = Date.now();
  waiting = true;
  socket.send(JSON.stringify({ t: 'stroke_end', strokeId: id, points: tail }));
}

socket.on('message', (raw) => {
  let msg: ServerMessage;
  try {
    msg = JSON.parse(raw.toString()) as ServerMessage;
  } catch {
    return;
  }

  if (msg.t === 'snapshot') {
    const draw = msg.snapshot.layers.find((l) => l.kind === 'draw');
    if (!draw) fail('the room has no draw layer');
    layerId = draw!.id;
    canvasSize = msg.snapshot.canvasSize;
    console.log(`[latency] canvas ${canvasSize}, ai resolution ${msg.snapshot.aiResolution}, prompt "${msg.snapshot.prompt}"`);
    drawNext();
    return;
  }

  if (msg.t === 'error') console.error(`[latency] server error: ${msg.message}`);
  if (msg.t === 'ai_status' && msg.state === 'error') console.error(`[latency] ai error: ${msg.message ?? 'unknown'}`);

  if (msg.t === 'ai_result' && waiting) {
    waiting = false;
    const notifiedMs = Date.now() - sentAt;
    // The message is not the picture: a client still has to fetch and decode
    // the PNG before anything appears, so that download is part of the wait.
    void fetch(`${base}${msg.url}`)
      .then(async (r) => (await r.arrayBuffer()).byteLength)
      .catch(() => 0)
      .then((bytes) => {
        const totalMs = Date.now() - sentAt;
        samples.push({ edit, totalMs, notifiedMs, pipelineMs: msg.latencyMs, bytes });
        console.log(
          `[latency] edit ${edit}/${opts.count}: ${totalMs} ms to pixels (notified ${notifiedMs} ms, server pipeline ${msg.latencyMs} ms)`,
        );
        if (samples.length >= opts.count) {
          clearTimeout(deadline);
          report();
          socket.close();
          return;
        }
        drawNext();
      });
  }
});

socket.on('error', (err) => fail(`socket error: ${err.message}`));
socket.on('close', () => {
  if (samples.length < opts.count) fail(`connection closed after ${samples.length}/${opts.count} samples`);
});

function summarise(values: number[]): { min: number; median: number; max: number } {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? Math.round((sorted[mid - 1]! + sorted[mid]!) / 2) : sorted[mid]!;
  return { min: sorted[0]!, median, max: sorted[sorted.length - 1]! };
}

function report(): void {
  const total = summarise(samples.map((s) => s.totalMs));
  const notified = summarise(samples.map((s) => s.notifiedMs));
  const pipeline = summarise(samples.map((s) => s.pipelineMs));
  const line = (label: string, v: { min: number; median: number; max: number }): string =>
    `[latency] ${label.padEnd(34)} min ${v.min} ms | median ${v.median} ms | max ${v.max} ms`;
  console.log('');
  console.log(`[latency] backend: ${health.backend ?? 'unknown'}${workerNote}  samples: ${samples.length}`);
  // Named for what they actually measure. `latencyMs` from the server is NOT
  // backend-only: it starts before the input render and ends after compositing.
  console.log(line('stroke_end -> pixels on screen', total));
  console.log(line('stroke_end -> ai_result message', notified));
  console.log(line('server pipeline (render+gen+composite)', pipeline));
}
