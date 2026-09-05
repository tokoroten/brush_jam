/**
 * Playtest simulation: N clients draw in one room for a few minutes while this
 * measures what a person would actually notice - how often the AI answers, how
 * long each answer took, whether anyone got an error, and whether everyone is
 * still looking at the same picture at the end.
 *
 *   pnpm --filter @brushjam/server playtest-sim -- \
 *     --url http://127.0.0.1:8787 --users 3 --minutes 2
 *
 * It talks to a RUNNING server over the real WebSocket protocol - no imports of
 * server internals - so it exercises validation, broadcast and the scheduler
 * exactly as a browser would. Exits non-zero if anything errored, if the
 * clients disagree about the stroke count, or if presence is short a member.
 */
import { WebSocket } from 'ws';
import type { ServerMessage, StrokeInit, Point } from '@brushjam/shared';

interface Options {
  url: string;
  users: number;
  minutes: number;
  room: string | null;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = { url: 'http://127.0.0.1:8787', users: 3, minutes: 2, room: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    const value = argv[i + 1];
    if (arg === '--url' && value) opts.url = value.replace(/\/$/, ''), i++;
    else if (arg === '--users' && value) opts.users = Number(value), i++;
    else if (arg === '--minutes' && value) opts.minutes = Number(value), i++;
    else if (arg === '--room' && value) opts.room = value, i++;
    else if (arg === '--help') {
      console.log('usage: playtest-sim [--url http://host:port] [--users 3] [--minutes 2] [--room id]');
      process.exit(0);
    }
  }
  if (!Number.isFinite(opts.users) || opts.users < 1) throw new Error('--users must be at least 1');
  if (!Number.isFinite(opts.minutes) || opts.minutes <= 0) throw new Error('--minutes must be positive');
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
const wsBase = opts.url.replace(/^http/, 'ws');

// Rhythm of a person drawing: a short stroke every second or so, an occasional
// noise scribble for the AI to invent into, an occasional undo.
const STROKE_MIN_MS = 500;
const STROKE_MAX_MS = 2000;
const NOISE_EVERY_MS = 20_000;
const UNDO_EVERY_MS = 30_000;
const CHUNK_MS = 60;
const COLORS = ['#e05252', '#4f8ef7', '#3fb56b', '#e0a83a', '#a86fe0'];
const PROMPT_A = 'anime style, fantasy town, vibrant colors';
const PROMPT_B = 'anime style, seaside village at sunset, warm light';

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
const rnd = (lo: number, hi: number): number => lo + Math.random() * (hi - lo);
const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(Math.random() * xs.length)]!;

let strokeSeq = 0;
const strokeId = (): string => `sim${(strokeSeq += 1).toString(36)}${Math.random().toString(36).slice(2, 8)}`;

/** A short squiggle somewhere on the canvas, as a list of points. */
function squiggle(canvasSize: number, length: number): Point[] {
  const x0 = rnd(canvasSize * 0.15, canvasSize * 0.85);
  const y0 = rnd(canvasSize * 0.15, canvasSize * 0.85);
  const pts: Point[] = [];
  let x = x0;
  let y = y0;
  let dx = rnd(-14, 14);
  let dy = rnd(-14, 14);
  for (let i = 0; i < length; i++) {
    dx += rnd(-4, 4);
    dy += rnd(-4, 4);
    x = Math.max(0, Math.min(canvasSize, x + dx));
    y = Math.max(0, Math.min(canvasSize, y + dy));
    pts.push({ x: Math.round(x), y: Math.round(y) });
  }
  return pts;
}

interface ResultEvent {
  /** ms since the run started. */
  at: number;
  /** Time from the newest stroke_end this client had sent or seen. */
  sinceStrokeMs: number | null;
  latencyMs: number | undefined;
}

class SimClient {
  readonly errors: string[] = [];
  readonly results: ResultEvent[] = [];
  /** Strokes this client believes exist, by id, from the server's broadcasts. */
  readonly committed = new Set<string>();
  readonly undone = new Set<string>();
  members = 0;
  canvasSize = 4096;
  layerId = '';
  userId = '';
  private socket: WebSocket | null = null;
  private ready: Promise<void>;
  /** When the room last finished a stroke, from anyone. */
  private lastStrokeEnd: number | null = null;
  closedUnexpectedly: string | null = null;

  constructor(
    readonly name: string,
    private readonly roomId: string,
    private readonly startedAt: number,
  ) {
    const url = `${wsBase}/ws/rooms/${roomId}?name=${encodeURIComponent(name)}&token=${encodeURIComponent(`sim-${name}`)}`;
    const socket = new WebSocket(url);
    this.socket = socket;
    this.ready = new Promise<void>((resolve, reject) => {
      const onOpen = (): void => resolve();
      socket.once('open', onOpen);
      socket.once('error', (err: Error) => reject(err));
      setTimeout(() => reject(new Error(`${name}: socket did not open within 10 s`)), 10_000).unref();
    });
    socket.on('message', (raw: Buffer) => this.receive(raw.toString()));
    socket.on('close', (code: number) => {
      if (code !== 1000 && code !== 1005) this.closedUnexpectedly = `closed with code ${code}`;
      this.socket = null;
    });
    socket.on('error', (err: Error) => this.errors.push(`socket: ${err.message}`));
  }

  waitOpen(): Promise<void> {
    return this.ready;
  }

  private receive(raw: string): void {
    let msg: ServerMessage;
    try {
      msg = JSON.parse(raw) as ServerMessage;
    } catch {
      this.errors.push('unparseable server message');
      return;
    }
    switch (msg.t) {
      case 'snapshot':
        this.userId = msg.snapshot.youUserId;
        this.canvasSize = msg.snapshot.canvasSize;
        this.layerId = msg.snapshot.layers[0]?.id ?? '';
        this.members = msg.snapshot.members.length;
        for (const s of msg.snapshot.strokes) this.committed.add(s.id);
        for (const id of msg.snapshot.undone) this.undone.add(id);
        break;
      case 'presence':
        this.members = msg.members.length;
        break;
      case 'stroke_committed':
        this.committed.add(msg.stroke.id);
        this.lastStrokeEnd = Date.now();
        break;
      case 'undo_applied':
        this.undone.add(msg.strokeId);
        break;
      case 'stroke_cancel':
        this.errors.push(`stroke cancelled: ${msg.reason}`);
        break;
      case 'ai_result':
        this.results.push({
          at: Date.now() - this.startedAt,
          sinceStrokeMs: this.lastStrokeEnd === null ? null : Date.now() - this.lastStrokeEnd,
          latencyMs: msg.latencyMs,
        });
        break;
      case 'ai_status':
        if (msg.state === 'error') this.errors.push(`ai_status error: ${msg.message ?? '(no message)'}`);
        break;
      case 'error':
        this.errors.push(msg.message);
        break;
      default:
        break;
    }
  }

  private send(msg: unknown): void {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(msg));
  }

  /** One stroke, sent the way the client does it: start, chunks, end. */
  async draw(tool: 'pen' | 'noise'): Promise<void> {
    const points = squiggle(this.canvasSize, tool === 'noise' ? 14 : 10);
    const stroke: StrokeInit = {
      id: strokeId(),
      layerId: this.layerId,
      tool,
      color: tool === 'noise' ? '#000000' : pick(COLORS),
      width: tool === 'noise' ? 64 : 14,
      points: points.slice(0, 2),
    };
    this.send({ t: 'stroke_start', stroke });
    for (let i = 2; i < points.length; i += 3) {
      await sleep(CHUNK_MS);
      this.send({ t: 'stroke_chunk', strokeId: stroke.id, points: points.slice(i, i + 3) });
    }
    await sleep(CHUNK_MS);
    this.send({ t: 'stroke_end', strokeId: stroke.id, points });
    this.lastStrokeEnd = Date.now();
  }

  undo(): void {
    this.send({ t: 'undo' });
  }

  setPrompt(prompt: string): void {
    this.send({ t: 'set_prompt', prompt });
  }

  close(): void {
    this.socket?.close(1000);
  }

  /** What this client thinks is on the canvas: committed minus undone. */
  get visibleStrokes(): number {
    let n = 0;
    for (const id of this.committed) if (!this.undone.has(id)) n += 1;
    return n;
  }
}

async function createRoom(): Promise<string> {
  if (opts.room) return opts.room;
  const res = await fetch(`${opts.url}/api/rooms`, { method: 'POST' });
  if (!res.ok) throw new Error(`POST /api/rooms failed: ${res.status} ${await res.text()}`);
  const body = (await res.json()) as { roomId?: string };
  if (!body.roomId) throw new Error('server did not return a roomId');
  return body.roomId;
}

/** One user's whole session: draw, occasionally noise, occasionally undo. */
async function drive(client: SimClient, endAt: number, index: number): Promise<void> {
  let nextNoise = Date.now() + NOISE_EVERY_MS * (0.5 + index / Math.max(1, opts.users));
  let nextUndo = Date.now() + UNDO_EVERY_MS * (0.5 + index / Math.max(1, opts.users));
  // Stagger the users so they are not all drawing on the same beat.
  await sleep(index * 250);
  while (Date.now() < endAt) {
    const now = Date.now();
    if (now >= nextNoise) {
      await client.draw('noise');
      nextNoise = now + NOISE_EVERY_MS;
    } else {
      await client.draw('pen');
    }
    if (Date.now() >= nextUndo) {
      client.undo();
      nextUndo = Date.now() + UNDO_EVERY_MS;
    }
    await sleep(rnd(STROKE_MIN_MS, STROKE_MAX_MS));
  }
}

function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))]!;
}

const fmt = (ms: number | null | undefined): string => (ms === null || ms === undefined ? 'n/a' : `${(ms / 1000).toFixed(2)} s`);

async function main(): Promise<void> {
  const roomId = await createRoom();
  const startedAt = Date.now();
  const durationMs = opts.minutes * 60_000;
  console.log(`[sim] room ${roomId} at ${opts.url}: ${opts.users} users for ${opts.minutes} min`);

  const clients: SimClient[] = [];
  for (let i = 0; i < opts.users; i++) clients.push(new SimClient(`Sim${i + 1}`, roomId, startedAt));
  await Promise.all(clients.map((c) => c.waitOpen()));
  // The snapshot arrives right after open and carries the layer id everything
  // is drawn on, so wait for it rather than racing it.
  await sleep(500);
  const missingLayer = clients.filter((c) => !c.layerId);
  if (missingLayer.length > 0) throw new Error(`no snapshot for ${missingLayer.map((c) => c.name).join(', ')}`);

  const endAt = startedAt + durationMs;
  // One prompt change at the midpoint, from the first user, like someone
  // changing their mind halfway through.
  const promptTimer = setTimeout(() => {
    console.log('[sim] prompt change at the midpoint');
    clients[0]!.setPrompt(PROMPT_B);
  }, durationMs / 2);
  clients[0]!.setPrompt(PROMPT_A);

  const progress = setInterval(() => {
    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(0);
    const results = clients[0]!.results.length;
    console.log(`[sim] ${elapsed}s: ${clients[0]!.visibleStrokes} strokes visible, ${results} ai results`);
  }, 15_000);
  progress.unref();

  await Promise.all(clients.map((c, i) => drive(c, endAt, i)));
  clearTimeout(promptTimer);
  clearInterval(progress);
  // Let the last broadcasts and any in-flight generation land.
  await sleep(3000);

  const elapsedMin = (Date.now() - startedAt) / 60_000;
  const results = clients[0]!.results;
  const gaps: number[] = [];
  for (let i = 1; i < results.length; i++) gaps.push(results[i]!.at - results[i - 1]!.at);
  const sinceStroke = results.map((r) => r.sinceStrokeMs).filter((v): v is number => v !== null);
  const reported = results.map((r) => r.latencyMs).filter((v): v is number => v !== undefined);

  const strokeCounts = clients.map((c) => c.visibleStrokes);
  const converged = strokeCounts.every((n) => n === strokeCounts[0]);
  const presenceOk = clients.every((c) => c.members === opts.users);
  const errors = clients.flatMap((c) => c.errors.map((e) => `${c.name}: ${e}`));
  const dropped = clients.filter((c) => c.closedUnexpectedly).map((c) => `${c.name}: ${c.closedUnexpectedly}`);

  console.log('');
  console.log('=== playtest simulation summary ===');
  console.log(`room                 ${roomId} at ${opts.url}`);
  console.log(`users / duration     ${opts.users} / ${elapsedMin.toFixed(2)} min`);
  console.log(`strokes drawn        ${strokeSeq} (visible per client: ${strokeCounts.join(', ')})`);
  console.log(`ai results           ${results.length} (${(results.length / elapsedMin).toFixed(1)}/min)`);
  console.log(`  gap between        median ${fmt(percentile(gaps, 0.5))}, p90 ${fmt(percentile(gaps, 0.9))}, max ${fmt(percentile(gaps, 1))}`);
  console.log(`  since last stroke  median ${fmt(percentile(sinceStroke, 0.5))}, p90 ${fmt(percentile(sinceStroke, 0.9))}`);
  console.log(`  server latencyMs   median ${fmt(percentile(reported, 0.5))}, p90 ${fmt(percentile(reported, 0.9))}`);
  console.log(`presence             ${clients.map((c) => c.members).join(', ')} of ${opts.users} ${presenceOk ? 'OK' : 'MISMATCH'}`);
  console.log(`convergence          ${converged ? 'OK - all clients agree' : `DIVERGED: ${strokeCounts.join(' vs ')}`}`);
  console.log(`errors               ${errors.length === 0 ? 'none' : errors.length}`);
  for (const e of errors.slice(0, 10)) console.log(`  ${e}`);
  for (const d of dropped) console.log(`  dropped: ${d}`);

  for (const c of clients) c.close();
  await sleep(200);

  const failed = errors.length > 0 || dropped.length > 0 || !converged || !presenceOk;
  if (results.length === 0) {
    console.log('note: no ai_result arrived - is a real backend configured?');
  }
  console.log(failed ? 'RESULT: FAIL' : 'RESULT: PASS');
  process.exit(failed ? 1 : 0);
}

await main();
