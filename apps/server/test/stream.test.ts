import { QUALITY_SUFFIX } from '@brushjam/shared';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { StreamBackend, streamReachable } from '../src/ai/backends/stream.js';
import { AbortedError, type GenerateRequest } from '../src/ai/backends/types.js';

const req: GenerateRequest = {
  prompt: 'anime style, fantasy town',
  negativePrompt: 'lowres',
  imagePng: Buffer.from('image-bytes'),
  maskPng: Buffer.from('mask-bytes'),
  size: 1024,
  denoise: 0.55,
  steps: 4,
  seed: 12345,
  tag: 'room1_r7',
};

interface Recorded { url: string; init?: RequestInit }

function stubFetch(handler: (path: string, init?: RequestInit) => Response | Promise<Response>): Recorded[] {
  const calls: Recorded[] = [];
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return handler(new URL(String(input)).pathname, init);
  });
  return calls;
}

const okResponse = (b64: string): Response =>
  new Response(JSON.stringify({ image_b64: b64, width: 1024, height: 1024, timings: { total_ms: 420 } }), { status: 200 });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('StreamBackend', () => {
  it('posts the MVP_PLAN contract and returns the decoded PNG', async () => {
    const png = Buffer.from('generated-png-bytes');
    const calls = stubFetch(() => okResponse(png.toString('base64')));

    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790/' });
    const out = await backend.generate(req, new AbortController().signal);

    expect(out.equals(png)).toBe(true);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe('http://127.0.0.1:8790/generate');
    const body = JSON.parse(String(calls[0]!.init!.body)) as Record<string, unknown>;
    expect(body).toMatchObject({
      prompt: 'anime style, fantasy town',
      negative_prompt: 'lowres',
      denoise: 0.55,
      steps: 4,
      seed: 12345,
      width: 1024,
      height: 1024,
    });
    expect(body.image_b64).toBe(req.imagePng.toString('base64'));
    expect(body.mask_b64).toBe(req.maskPng.toString('base64'));
  });

  it('does not send the quality suffix twice', async () => {
    const calls = stubFetch(() => okResponse(Buffer.from('x').toString('base64')));
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    await backend.generate({ ...req, prompt: `a town${QUALITY_SUFFIX}` }, new AbortController().signal);
    expect(JSON.parse(String(calls[0]!.init!.body)).prompt).toBe('a town');
  });

  it('accepts a data: URL payload', async () => {
    const png = Buffer.from('data-url-png');
    stubFetch(() => okResponse(`data:image/png;base64,${png.toString('base64')}`).clone());
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    const out = await backend.generate(req, new AbortController().signal);
    expect(out.toString()).toBe('data-url-png');
  });

  it('throws on a non-2xx response, including the worker message', async () => {
    stubFetch(() => new Response('checkpoint not found', { status: 503 }));
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/503 checkpoint not found/);
  });

  it('throws when the worker answers without an image', async () => {
    stubFetch(() => new Response(JSON.stringify({ timings: {} }), { status: 200 }));
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/no image/);
  });

  it('maps the caller abort to AbortedError', async () => {
    const controller = new AbortController();
    stubFetch(
      (_path, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })), { once: true });
        }),
    );
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    const promise = backend.generate(req, controller.signal);
    controller.abort();
    await expect(promise).rejects.toBeInstanceOf(AbortedError);
  });

  it('times out a stalled worker so the scheduler slot is released', async () => {
    stubFetch(
      (_path, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('timeout'), { name: 'TimeoutError' })), { once: true });
        }),
    );
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790', timeoutMs: 20 });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
  });

  it('streamReachable is true only when /healthz reports ok', async () => {
    stubFetch((path) => (path === '/healthz' ? new Response(JSON.stringify({ ok: true }), { status: 200 }) : new Response('', { status: 404 })));
    await expect(streamReachable('http://127.0.0.1:8790')).resolves.toBe(true);

    vi.unstubAllGlobals();
    stubFetch(() => new Response(JSON.stringify({ ok: false, error: 'load failed' }), { status: 200 }));
    await expect(streamReachable('http://127.0.0.1:8790')).resolves.toBe(false);

    vi.unstubAllGlobals();
    stubFetch(() => Promise.reject(new Error('fetch failed')));
    await expect(streamReachable('http://127.0.0.1:8790')).resolves.toBe(false);
  });
});
