/**
 * Denoise sweep over synthetic drawings, straight through a backend.
 *
 *   pnpm --filter @brushjam/server quality-grid [--res 768] [--drawings a,b,c,d]
 *                                               [--denoise 0.5,0.65,0.8,0.9] [--out docs/experiments]
 *
 * No server and no WebSocket: it builds the four test drawings itself, calls
 * `createBackend(loadConfig())` and drives `generate()` directly, so the same
 * script works against the stream worker, ComfyUI or the mock. The seed is
 * fixed and the mask is full white (the full-canvas path), so the only thing
 * that varies across a row is the denoise.
 *
 * Results land in <out>/<date>/ as `<drawing>_d<denoise>.png` plus a contact
 * sheet and results.json. Try it with AI_BACKEND=mock first - it is instant.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { createCanvas, loadImage, type Canvas } from '@napi-rs/canvas';
import { DEFAULT_NEGATIVE_PROMPT, renderStrokes, type RenderableStroke } from '@brushjam/shared';
import { createBackend } from '../src/ai/backends/index.js';
import { loadConfig } from '../src/config.js';
import { buildFullMask } from '../src/raster.js';

const DRAW_SIZE = 1024;
const CELL = 384;
const LABEL = 34;
const PROMPT = 'anime style, fantasy town, vibrant colors';
const SEED = 424242;

interface Options {
  res: number;
  out: string;
  drawings: string[];
  denoises: number[];
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    res: 768,
    out: path.resolve(process.cwd(), '../../docs/experiments'),
    drawings: ['a', 'b', 'c', 'd'],
    denoises: [0.5, 0.65, 0.8, 0.9],
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === '--res' && value) opts.res = Math.max(256, Number(value) || 768);
    else if (arg === '--out' && value) opts.out = path.resolve(value);
    // Split on whitespace too: PowerShell turns an unquoted `a,b` into `a b`.
    else if (arg === '--drawings' && value) opts.drawings = value.split(/[\s,]+/).filter(Boolean);
    else if (arg === '--denoise' && value) {
      opts.denoises = value
        .split(/[\s,]+/)
        .map((s) => Number(s))
        .filter((n) => Number.isFinite(n) && n > 0 && n <= 1);
    }
  }
  return opts;
}

// --- the four test drawings -------------------------------------------------
// World coordinates are 0..1024. Every stroke is deterministic, so a rerun
// compares like with like (noise strokes hash their id + world position).

function line(id: string, color: string, width: number, pts: [number, number][]): RenderableStroke {
  return { id, tool: 'pen', color, width, points: pts.map(([x, y]) => ({ x, y })) };
}

function noise(id: string, width: number, pts: [number, number][]): RenderableStroke {
  return { id, tool: 'noise', color: '#808080', width, points: pts.map(([x, y]) => ({ x, y })) };
}

/** Straight run of points so a stroke actually covers ground. */
function run(x0: number, y0: number, x1: number, y1: number, steps = 24): [number, number][] {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps;
    return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t] as [number, number];
  });
}

function houseAndTree(prefix = 'a'): RenderableStroke[] {
  const ink = '#222222';
  const w = 7;
  return [
    line(`${prefix}-ground`, ink, w, run(60, 800, 964, 800)),
    line(`${prefix}-h1`, ink, w, run(220, 800, 220, 520)),
    line(`${prefix}-h2`, ink, w, run(220, 520, 520, 520)),
    line(`${prefix}-h3`, ink, w, run(520, 520, 520, 800)),
    line(`${prefix}-r1`, ink, w, run(190, 520, 370, 380)),
    line(`${prefix}-r2`, ink, w, run(370, 380, 550, 520)),
    line(`${prefix}-d1`, ink, w, run(300, 800, 300, 660)),
    line(`${prefix}-d2`, ink, w, run(300, 660, 380, 660)),
    line(`${prefix}-d3`, ink, w, run(380, 660, 380, 800)),
    line(`${prefix}-w1`, ink, w, run(430, 600, 480, 600)),
    line(`${prefix}-w2`, ink, w, run(430, 600, 430, 650)),
    line(`${prefix}-w3`, ink, w, run(430, 650, 480, 650)),
    line(`${prefix}-w4`, ink, w, run(480, 600, 480, 650)),
    line(`${prefix}-t1`, '#5a3a1a', 14, run(740, 800, 740, 600)),
    line(`${prefix}-t2`, '#2f7a3a', 10, [
      [740, 600],
      [660, 560],
      [680, 470],
      [740, 420],
      [800, 470],
      [820, 560],
      [740, 600],
    ]),
    line(`${prefix}-t3`, '#2f7a3a', 10, run(690, 540, 790, 540, 12)),
  ];
}

function stickFigure(): RenderableStroke[] {
  const ink = '#222222';
  const w = 9;
  return [
    line('b-ground', ink, 6, run(60, 860, 964, 860)),
    line(
      'b-head',
      ink,
      w,
      Array.from({ length: 33 }, (_, i) => {
        const a = (i / 32) * Math.PI * 2;
        return [420 + Math.cos(a) * 70, 250 + Math.sin(a) * 70] as [number, number];
      }),
    ),
    line('b-spine', ink, w, run(420, 320, 420, 600)),
    line('b-arms', ink, w, run(300, 420, 540, 420)),
    line('b-leg1', ink, w, run(420, 600, 350, 780)),
    line('b-leg2', ink, w, run(420, 600, 490, 780)),
    line('b-blob', '#e0473c', 120, [
      [720, 500],
      [760, 470],
      [800, 500],
      [820, 560],
      [780, 610],
      [730, 590],
      [710, 545],
      [740, 520],
    ]),
    line('b-blob2', '#f0a51e', 70, run(730, 540, 800, 560, 8)),
  ];
}

function houseWithNoisySky(): RenderableStroke[] {
  return [
    ...houseAndTree('c'),
    noise('c-sky1', 150, run(60, 150, 964, 150, 40)),
    noise('c-sky2', 150, run(60, 280, 964, 280, 40)),
  ];
}

function mostlyNoise(): RenderableStroke[] {
  const strokes: RenderableStroke[] = [];
  for (let i = 0; i < 6; i++) {
    const y = 140 + i * 130;
    strokes.push(noise(`d-n${i}`, 130, run(60, y, 964, y + (i % 2 === 0 ? 60 : -60), 40)));
  }
  strokes.push(line('d-l1', '#111111', 8, run(120, 820, 900, 820)));
  strokes.push(line('d-l2', '#111111', 8, run(300, 820, 300, 480)));
  strokes.push(line('d-l3', '#111111', 8, run(300, 480, 620, 480)));
  return strokes;
}

const DRAWINGS: Record<string, { title: string; strokes: () => RenderableStroke[] }> = {
  a: { title: 'line-art house + tree', strokes: () => houseAndTree() },
  b: { title: 'stick figure + blob', strokes: stickFigure },
  c: { title: 'house + noise sky', strokes: houseWithNoisySky },
  d: { title: 'mostly noise', strokes: mostlyNoise },
};

function renderDrawing(strokes: RenderableStroke[]): Canvas {
  const canvas = createCanvas(DRAW_SIZE, DRAW_SIZE);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
  renderStrokes(ctx as never, strokes, {
    createCanvas: (w, h) => createCanvas(w, h) as never,
    bounds: { width: DRAW_SIZE, height: DRAW_SIZE },
  });
  return canvas;
}

/** Scale a square canvas to `size`, high quality only when it really resizes. */
function resample(source: Canvas, size: number): Canvas {
  if (source.width === size) return source;
  const out = createCanvas(size, size);
  const ctx = out.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, size, size);
  return out;
}

// --- run --------------------------------------------------------------------

const opts = parseArgs(process.argv.slice(2));
const unknown = opts.drawings.filter((k) => !DRAWINGS[k]);
if (unknown.length > 0) {
  console.error(`[grid] unknown drawing(s): ${unknown.join(', ')} (have ${Object.keys(DRAWINGS).join(', ')})`);
  process.exit(1);
}
if (opts.denoises.length === 0) {
  console.error('[grid] --denoise did not contain any value in (0, 1]');
  process.exit(1);
}

const config = loadConfig();
const backend = await createBackend(config);
// Local date, not UTC: the folder should match the day the operator ran it.
const now = new Date();
const date = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
const outDir = path.join(opts.out, date);
mkdirSync(outDir, { recursive: true });

console.log(`[grid] backend ${backend.name}, res ${opts.res}, steps ${config.aiSteps}, seed ${SEED}`);
console.log(`[grid] drawings ${opts.drawings.join(',')} x denoise ${opts.denoises.join(',')} -> ${outDir}`);

const mask = buildFullMask(opts.res);
const inputs = new Map<string, Canvas>();
for (const key of opts.drawings) {
  const full = renderDrawing(DRAWINGS[key]!.strokes());
  // Saved at the generation size, so the input file is byte-for-byte what the
  // backend was handed (noise drawings compress badly, so this matters).
  const scaled = resample(full, opts.res);
  writeFileSync(path.join(outDir, `${key}_input.png`), scaled.toBuffer('image/png'));
  inputs.set(key, scaled);
}

interface Cell {
  drawing: string;
  denoise: number;
  file: string;
  latencyMs: number;
  bytes: number;
  error?: string;
}

const cells: Cell[] = [];
const controller = new AbortController();

for (const key of opts.drawings) {
  for (const denoise of opts.denoises) {
    const file = `${key}_d${denoise.toFixed(2)}.png`;
    const started = Date.now();
    try {
      const png = await backend.generate(
        {
          prompt: PROMPT,
          negativePrompt: DEFAULT_NEGATIVE_PROMPT,
          imagePng: inputs.get(key)!.toBuffer('image/png'),
          maskPng: mask.png,
          size: opts.res,
          denoise,
          steps: config.aiSteps,
          seed: SEED,
          tag: `grid-${key}-${denoise}`,
        },
        controller.signal,
      );
      const latencyMs = Date.now() - started;
      writeFileSync(path.join(outDir, file), png);
      cells.push({ drawing: key, denoise, file, latencyMs, bytes: png.length });
      console.log(`[grid] ${key} d=${denoise} -> ${file} (${latencyMs} ms, ${png.length} bytes)`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      cells.push({ drawing: key, denoise, file, latencyMs: Date.now() - started, bytes: 0, error: message });
      console.error(`[grid] ${key} d=${denoise} FAILED: ${message}`);
    }
  }
}

// --- contact sheet ----------------------------------------------------------
// Column 0 is the input, then one column per denoise. Cells are 384 px so a
// full 4x5 sheet stays well under a couple of MB.

const cols = 1 + opts.denoises.length;
const rows = opts.drawings.length;
const sheet = createCanvas(cols * CELL, LABEL + rows * (CELL + LABEL));
const sctx = sheet.getContext('2d');
sctx.fillStyle = '#f5f5f5';
sctx.fillRect(0, 0, sheet.width, sheet.height);
sctx.fillStyle = '#111111';
sctx.font = '20px sans-serif';
sctx.textBaseline = 'middle';
sctx.fillText('input', 12, LABEL / 2);
opts.denoises.forEach((d, i) => sctx.fillText(`denoise ${d}`, (i + 1) * CELL + 12, LABEL / 2));

for (let r = 0; r < rows; r++) {
  const key = opts.drawings[r]!;
  const top = LABEL + r * (CELL + LABEL);
  sctx.fillStyle = '#111111';
  sctx.font = '18px sans-serif';
  sctx.fillText(`${key}: ${DRAWINGS[key]!.title}`, 12, top + CELL + LABEL / 2);
  sctx.drawImage(inputs.get(key)!, 0, top, CELL, CELL);
  for (let c = 0; c < opts.denoises.length; c++) {
    const cell = cells.find((x) => x.drawing === key && x.denoise === opts.denoises[c]);
    const x = (c + 1) * CELL;
    if (!cell || cell.error) {
      sctx.fillStyle = '#ddaaaa';
      sctx.fillRect(x, top, CELL, CELL);
      sctx.fillStyle = '#661111';
      sctx.fillText('failed', x + 12, top + CELL / 2);
      continue;
    }
    const img = await loadImage(path.join(outDir, cell.file));
    sctx.drawImage(img, x, top, CELL, CELL);
  }
}
writeFileSync(path.join(outDir, 'grid.png'), sheet.toBuffer('image/png'));

const results = {
  date,
  backend: backend.name,
  prompt: PROMPT,
  negativePrompt: DEFAULT_NEGATIVE_PROMPT,
  resolution: opts.res,
  steps: config.aiSteps,
  cfg: config.aiCfg,
  seed: SEED,
  drawings: opts.drawings.map((k) => ({ key: k, title: DRAWINGS[k]!.title })),
  denoises: opts.denoises,
  cells,
};
writeFileSync(path.join(outDir, 'results.json'), `${JSON.stringify(results, null, 2)}\n`);

// --- summary ----------------------------------------------------------------
const header = ['drawing', ...opts.denoises.map((d) => `d=${d}`)];
const widths = header.map((h) => h.length);
const table = opts.drawings.map((key) => {
  const row = [
    key,
    ...opts.denoises.map((d) => {
      const cell = cells.find((x) => x.drawing === key && x.denoise === d);
      return cell ? (cell.error ? 'FAILED' : `${cell.latencyMs} ms`) : '-';
    }),
  ];
  row.forEach((v, i) => (widths[i] = Math.max(widths[i]!, v.length)));
  return row;
});
const pad = (row: string[]): string => row.map((v, i) => v.padEnd(widths[i]!)).join('  ');
console.log('');
console.log(pad(header));
for (const row of table) console.log(pad(row));
const ok = cells.filter((c) => !c.error);
if (ok.length > 0) {
  const times = ok.map((c) => c.latencyMs).sort((a, b) => a - b);
  console.log('');
  console.log(
    `[grid] ${ok.length}/${cells.length} cells | min ${times[0]} ms | median ${times[Math.floor(times.length / 2)]} ms | max ${times[times.length - 1]} ms`,
  );
}
console.log(`[grid] wrote ${outDir}`);
process.exit(cells.some((c) => c.error) ? 1 : 0);
