import { afterEach, describe, expect, it, vi } from 'vitest';
import { ComfyUIBackend } from '../src/ai/backends/comfyui.js';
import type { GenerateRequest } from '../src/ai/backends/types.js';

const req: GenerateRequest = {
  prompt: 'a town',
  negativePrompt: 'lowres',
  imagePng: Buffer.from('image'),
  maskPng: Buffer.from('mask'),
  size: 1024,
  denoise: 0.55,
  steps: 14,
  seed: 1,
  tag: 'room_r1',
};

interface Call {
  path: string;
  method: string;
  body?: string;
}

interface Options {
  running?: string[];
  pending?: string[];
  /** Paths that never settle until aborted. */
  hang?: string[];
  /** Number of /history calls that fail transiently before succeeding. */
  historyFailures?: number;
  /** /history keeps answering "not done yet" forever. */
  neverFinishes?: boolean;
}

function stub(options: Options = {}): Call[] {
  const calls: Call[] = [];
  let historyFailures = options.historyFailures ?? 0;
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    const url = new URL(String(input));
    const path = url.pathname;
    calls.push({ path, method: init?.method ?? 'GET', body: typeof init?.body === 'string' ? init.body : undefined });

    if (options.hang?.some((h) => path.startsWith(h))) {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('stalled'), { name: 'TimeoutError' })), {
          once: true,
        });
      });
    }
    if (path === '/upload/image') return new Response(JSON.stringify({ name: 'in.png', subfolder: '', type: 'input' }), { status: 200 });
    if (path === '/prompt') return new Response(JSON.stringify({ prompt_id: 'pid-1' }), { status: 200 });
    if (path === '/interrupt') return new Response('{}', { status: 200 });
    if (path === '/queue') {
      if (init?.method === 'POST') return new Response('{}', { status: 200 });
      return new Response(
        JSON.stringify({
          queue_running: (options.running ?? []).map((id) => [0, id]),
          queue_pending: (options.pending ?? []).map((id) => [0, id]),
        }),
        { status: 200 },
      );
    }
    if (path.startsWith('/history/')) {
      if (historyFailures > 0) {
        historyFailures -= 1;
        throw Object.assign(new Error('fetch failed'), { name: 'TypeError' });
      }
      if (options.neverFinishes) return new Response('{}', { status: 200 });
      return new Response(
        JSON.stringify({ 'pid-1': { outputs: { '11': { images: [{ filename: 'o.png', subfolder: '', type: 'output' }] } } } }),
        { status: 200 },
      );
    }
    if (path === '/view') return new Response(new Uint8Array([9, 9]), { status: 200 });
    return new Response('{}', { status: 200 });
  });
  return calls;
}

const backend = (extra: Partial<ConstructorParameters<typeof ComfyUIBackend>[0]> = {}): ComfyUIBackend =>
  new ComfyUIBackend({
    url: 'http://127.0.0.1:8188',
    checkpoint: 'c.safetensors',
    pollIntervalMs: 2,
    requestTimeoutMs: 25,
    timeoutMs: 300,
    ...extra,
  });

afterEach(() => vi.unstubAllGlobals());

/** Finding B3: a queued prompt must never be orphaned by a later failure. */
describe('ComfyUI prompt cleanup', () => {
  it('interrupts the running job when the view request fails', async () => {
    const calls = stub({ running: ['pid-1'], hang: ['/view'] });
    await expect(backend().generate(req, new AbortController().signal)).rejects.toThrow();
    expect(calls.some((c) => c.path === '/queue' && c.method === 'GET')).toBe(true);
    expect(calls.some((c) => c.path === '/interrupt')).toBe(true);
  });

  it('deletes the prompt from the queue when it is only pending', async () => {
    const calls = stub({ pending: ['pid-1'], hang: ['/view'] });
    await expect(backend().generate(req, new AbortController().signal)).rejects.toThrow();
    const del = calls.find((c) => c.path === '/queue' && c.method === 'POST');
    expect(del).toBeDefined();
    expect(JSON.parse(del!.body!)).toEqual({ delete: ['pid-1'] });
    expect(calls.some((c) => c.path === '/interrupt')).toBe(false);
  });

  it('cleans up after the overall generation deadline expires', async () => {
    const calls = stub({ running: ['pid-1'], neverFinishes: true });
    await expect(backend({ timeoutMs: 40 }).generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
    expect(calls.some((c) => c.path === '/interrupt')).toBe(true);
  });

  it('cleans up when the caller aborts', async () => {
    const calls = stub({ running: ['pid-1'], neverFinishes: true });
    const controller = new AbortController();
    const promise = backend({ timeoutMs: 5000 }).generate(req, controller.signal);
    setTimeout(() => controller.abort(), 10);
    await expect(promise).rejects.toThrow(/aborted/);
    expect(calls.some((c) => c.path === '/interrupt')).toBe(true);
  });

  it('does not touch a prompt that belongs to somebody else', async () => {
    const calls = stub({ running: ['other-job'], pending: ['another-job'], hang: ['/view'] });
    await expect(backend().generate(req, new AbortController().signal)).rejects.toThrow();
    expect(calls.some((c) => c.path === '/interrupt')).toBe(false);
    expect(calls.some((c) => c.path === '/queue' && c.method === 'POST')).toBe(false);
  });

  it('does not cancel anything when /prompt itself fails', async () => {
    const calls = stub({ hang: ['/prompt'] });
    await expect(backend().generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
    expect(calls.some((c) => c.path === '/queue')).toBe(false);
  });

  it('retries a transient /history failure instead of giving up', async () => {
    const calls = stub({ historyFailures: 3 });
    const out = await backend().generate(req, new AbortController().signal);
    expect(out).toEqual(Buffer.from([9, 9]));
    expect(calls.filter((c) => c.path.startsWith('/history/')).length).toBeGreaterThan(3);
    // succeeded, so nothing was cancelled
    expect(calls.some((c) => c.path === '/interrupt')).toBe(false);
  });

  it('gives up on a persistent history failure once the deadline passes, and cleans up', async () => {
    const calls = stub({ running: ['pid-1'], historyFailures: 1000, timeoutMs: 60 } as Options);
    await expect(backend({ timeoutMs: 60 }).generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
    expect(calls.some((c) => c.path === '/interrupt')).toBe(true);
  });

  it('does not retry a real execution error', async () => {
    vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
      const path = new URL(String(input)).pathname;
      if (path === '/upload/image') return new Response(JSON.stringify({ name: 'in.png', subfolder: '', type: 'input' }), { status: 200 });
      if (path === '/prompt') return new Response(JSON.stringify({ prompt_id: 'pid-1' }), { status: 200 });
      if (path.startsWith('/history/')) return new Response(JSON.stringify({ 'pid-1': { status: { status_str: 'error' } } }), { status: 200 });
      return new Response('{}', { status: 200 });
    });
    await expect(backend().generate(req, new AbortController().signal)).rejects.toThrow(/execution error/);
  });
});
