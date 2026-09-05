/**
 * Live smoke test for the RunPod serverless backend - one real generation
 * against a real endpoint, with no room server and no WebSocket in the way.
 *
 *   RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=... \
 *     pnpm --filter @brushjam/server runpod-smoke [--res 768] [--profile fast|quality]
 *                                                 [--denoise 0.65] [--out <dir>]
 *
 * It builds a small drawing plus a full-white mask itself, drives
 * `RunpodBackend.generate()` directly, and writes the PNG out so the result can
 * be looked at. Two numbers are printed: the wall clock of this call, and
 * whether it looked like a cold start (anything over ~40 s is the worker
 * booting and loading the checkpoint off the network volume, not sampling).
 *
 * This is the script to run first after touching the endpoint, the volume or
 * the worker image. It costs one generation on the endpoint's GPU.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { createCanvas } from '@napi-rs/canvas';
import { DEFAULT_NEGATIVE_PROMPT, type AIProfileName } from '@brushjam/shared';
import { RunpodBackend } from '../src/ai/backends/runpod.js';
import { DEFAULT_FAST_LORA } from '../src/ai/backends/comfyui.js';

interface Options {
  res: number;
  profile: AIProfileName;
  denoise: number;
  out: string;
  runs: number;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    res: 768,
    profile: 'fast',
    denoise: 0.65,
    out: path.resolve(process.cwd(), 'smoke-out/runpod'),
    runs: 1,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === '--res' && value) opts.res = Math.max(256, Number(value) || 768);
    else if (arg === '--profile' && value) opts.profile = value === 'quality' ? 'quality' : 'fast';
    else if (arg === '--denoise' && value) opts.denoise = Number(value) || 0.65;
    else if (arg === '--out' && value) opts.out = path.resolve(value);
    else if (arg === '--runs' && value) opts.runs = Math.max(1, Number(value) || 1);
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));

const endpointId = process.env.RUNPOD_ENDPOINT_ID ?? '';
const apiKey = process.env.RUNPOD_API_KEY ?? '';
if (!endpointId || !apiKey) {
  console.error('[runpod-smoke] set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY (both live in the repo-root .env)');
  process.exit(1);
}

/** A house and a tree, drawn straight onto the canvas - no room, no strokes. */
function drawing(size: number): Buffer {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  const s = size / 1024;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const stroke = (color: string, width: number, pts: [number, number][]): void => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width * s;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x * s, y * s) : ctx.lineTo(x * s, y * s)));
    ctx.stroke();
  };
  stroke('#222222', 7, [[60, 800], [964, 800]]);
  stroke('#222222', 7, [[220, 800], [220, 520], [520, 520], [520, 800]]);
  stroke('#222222', 7, [[190, 520], [370, 380], [550, 520]]);
  stroke('#222222', 7, [[300, 800], [300, 660], [380, 660], [380, 800]]);
  stroke('#5a3a1a', 14, [[740, 800], [740, 600]]);
  stroke('#2f7a3a', 10, [[740, 600], [660, 560], [680, 470], [740, 420], [800, 470], [820, 560], [740, 600]]);
  return canvas.toBuffer('image/png');
}

/** Full white: regenerate everything, the same as the server's full-canvas mode. */
function fullMask(size: number): Buffer {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);
  return canvas.toBuffer('image/png');
}

const backend = new RunpodBackend({
  endpointId,
  apiKey,
  checkpoint: process.env.COMFYUI_CHECKPOINT ?? 'waiNSFWIllustrious_v150.safetensors',
  cfg: Number(process.env.AI_CFG ?? 5.5),
  vaeTile: Number(process.env.AI_VAE_TILE ?? 512),
  fastLora: process.env.COMFYUI_FAST_LORA ?? DEFAULT_FAST_LORA,
  timeoutMs: Number(process.env.RUNPOD_TIMEOUT_MS ?? 300_000),
});

const caps = await backend.capabilities();
console.log(`[runpod-smoke] endpoint ${endpointId}, profiles ${caps.profiles.join('/')}, ${opts.profile} @ ${opts.res}px denoise ${opts.denoise}`);

mkdirSync(opts.out, { recursive: true });
const imagePng = drawing(opts.res);
const maskPng = fullMask(opts.res);
writeFileSync(path.join(opts.out, 'input.png'), imagePng);

const times: number[] = [];
for (let i = 0; i < opts.runs; i++) {
  const started = Date.now();
  const png = await backend.generate(
    {
      profile: opts.profile,
      prompt: 'anime style, fantasy town, vibrant colors',
      negativePrompt: DEFAULT_NEGATIVE_PROMPT,
      imagePng,
      maskPng,
      size: opts.res,
      denoise: opts.denoise,
      steps: opts.profile === 'fast' ? Number(process.env.AI_FAST_STEPS ?? 4) : Number(process.env.AI_STEPS ?? 14),
      // A fresh seed per run: ComfyUI caches by prompt hash, so repeating the
      // identical workflow returns the previous result in ~200 ms and would
      // make the 'warm' figure a measurement of the cache, not the GPU.
      seed: 424242 + i,
      tag: `smoke${i}`,
    },
    new AbortController().signal,
  );
  const ms = Date.now() - started;
  times.push(ms);
  const file = path.join(opts.out, `${opts.profile}_${opts.res}_d${opts.denoise}_${i}.png`);
  writeFileSync(file, png);
  console.log(`[runpod-smoke] run ${i + 1}/${opts.runs}: ${ms} ms, ${png.length} bytes${i === 0 && ms > 40_000 ? ' (cold start)' : ''} -> ${file}`);
}

if (times.length > 1) {
  const warm = times.slice(1).sort((a, b) => a - b);
  console.log(`[runpod-smoke] cold ${times[0]} ms, warm median ${warm[Math.floor(warm.length / 2)]} ms over ${warm.length} runs`);
}
