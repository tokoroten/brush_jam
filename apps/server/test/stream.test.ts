import { createCanvas } from '@napi-rs/canvas';
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
  profile: 'quality' as const,
  tag: 'room1_r7',
};

/** A real PNG of the requested size: the backend now checks both. */
function pngOf(size: number): Buffer {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#3366aa';
  ctx.fillRect(0, 0, size, size);
  return canvas.toBuffer('image/png');
}

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
    const png = pngOf(1024);
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
    const calls = stubFetch(() => okResponse(pngOf(1024).toString('base64')));
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    await backend.generate({ ...req, prompt: `a town${QUALITY_SUFFIX}` }, new AbortController().signal);
    expect(JSON.parse(String(calls[0]!.init!.body)).prompt).toBe('a town');
  });

  it('accepts a data: URL payload', async () => {
    const png = pngOf(1024);
    stubFetch(() => okResponse(`data:image/png;base64,${png.toString('base64')}`).clone());
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    const out = await backend.generate(req, new AbortController().signal);
    expect(out.equals(png)).toBe(true);
  });

  // Review 6 finding 7: Buffer.from(x, 'base64') accepts nearly anything, so a
  // corrupt or wrong-sized result used to reach the compositor.
  describe('output validation', () => {
    const run = async (b64: string): Promise<Buffer> => {
      stubFetch(() => okResponse(b64));
      return new StreamBackend({ url: 'http://127.0.0.1:8790' }).generate(req, new AbortController().signal);
    };

    it('rejects a payload that is not base64 at all', async () => {
      await expect(run('not base64 !!!')).rejects.toThrow(/malformed base64/);
    });

    it('rejects an empty data: URL', async () => {
      await expect(run('data:image/png;base64,')).rejects.toThrow(/malformed base64|no image/);
    });

    it('rejects bytes that are not a PNG', async () => {
      await expect(run(Buffer.from('definitely not a png at all').toString('base64'))).rejects.toThrow(/not a PNG/);
    });

    it('rejects a valid PNG of the wrong size', async () => {
      await expect(run(pngOf(512).toString('base64'))).rejects.toThrow(/returned 512x512, expected 1024x1024/);
    });

    it('accepts a valid PNG of exactly the requested size', async () => {
      const png = pngOf(1024);
      expect((await run(png.toString('base64'))).equals(png)).toBe(true);
    });
  });

  // FastAPI reports errors as {"detail": "..."}; the raw envelope is noise.
  it('surfaces the FastAPI detail message rather than the JSON envelope', async () => {
    stubFetch(() => new Response(JSON.stringify({ detail: 'size 4096 out of range [256, 1024]' }), { status: 400 }));
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(
      /400 size 4096 out of range \[256, 1024\]/,
    );
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

  it('sends a request_id and queue flag', async () => {
    const calls = stubFetch(() => okResponse(pngOf(1024).toString('base64')));
    await new StreamBackend({ url: 'http://127.0.0.1:8790' }).generate(req, new AbortController().signal);
    const body = JSON.parse(String(calls[0]!.init!.body)) as Record<string, unknown>;
    expect(typeof body.request_id).toBe('string');
    expect((body.request_id as string).length).toBeGreaterThan(8);
    expect(body.queue).toBe(true);
  });

  it('cancels the GPU job when the caller aborts', async () => {
    // The whole point: dropping the HTTP request does not stop the diffusion
    // loop, so the backend must tell the worker to stop.
    const controller = new AbortController();
    const calls = stubFetch((path, init) => {
      if (path === '/cancel') return new Response('{}', { status: 200 });
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })), { once: true });
      });
    });
    const promise = new StreamBackend({ url: 'http://127.0.0.1:8790' }).generate(req, controller.signal);
    controller.abort();
    await expect(promise).rejects.toBeInstanceOf(AbortedError);

    const generateBody = JSON.parse(String(calls[0]!.init!.body)) as { request_id: string };
    const cancel = calls.find((c) => c.url.endsWith('/cancel'));
    expect(cancel, 'a /cancel call').toBeDefined();
    expect(JSON.parse(String(cancel!.init!.body))).toEqual({ request_id: generateBody.request_id });
  });

  it('cancels on its own timeout too', async () => {
    const calls = stubFetch((path, init) => {
      if (path === '/cancel') return new Response('{}', { status: 200 });
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('t'), { name: 'TimeoutError' })), { once: true });
      });
    });
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790', timeoutMs: 20 });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
    await vi.waitFor(() => expect(calls.some((c) => c.url.endsWith('/cancel'))).toBe(true));
  });

  it('a failing /cancel does not mask the original error', async () => {
    const controller = new AbortController();
    stubFetch((path, init) => {
      if (path === '/cancel') return Promise.reject(new Error('worker gone'));
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })), { once: true });
      });
    });
    const promise = new StreamBackend({ url: 'http://127.0.0.1:8790' }).generate(req, controller.signal);
    controller.abort();
    await expect(promise).rejects.toBeInstanceOf(AbortedError);
  });

  it('maps the worker 499 (cancelled) to AbortedError', async () => {
    stubFetch(() => new Response('cancelled (abc)', { status: 499 }));
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790' });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toBeInstanceOf(AbortedError);
  });

  it('reports a 409 as busy rather than a generic failure', async () => {
    stubFetch(() => new Response('busy with other-id', { status: 409 }));
    const backend = new StreamBackend({ url: 'http://127.0.0.1:8790', queue: false });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/busy/);
  });

  // Review 6 finding 2 (TS half): /cancel only asks the worker to stop at its
  // next diffusion step, so a retry issued immediately queues behind the job we
  // just abandoned and pays for it twice.
  describe('waiting for a cancelled job to actually stop', () => {
    /** A worker whose /generate hangs until the caller's signal aborts. */
    function stallingWorker(state: { busy: boolean }): string[] {
      const paths: string[] = [];
      let firstGenerate = true;
      // The worker keeps reporting the cancelled job as the running one until
      // it reaches its next diffusion step, which is the whole problem.
      let running = '';
      vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
        const path = new URL(String(input)).pathname;
        paths.push(path);
        if (path === '/cancel') {
          running = String((JSON.parse(String(init?.body)) as { request_id?: string }).request_id ?? '');
          return new Response('{}', { status: 200 });
        }
        if (path === '/healthz') {
          return new Response(JSON.stringify({ ok: true, warm: true, busy: state.busy, current_request_id: running }), {
            status: 200,
          });
        }
        if (!firstGenerate) return new Response(JSON.stringify({ image_b64: pngOf(1024).toString('base64') }), { status: 200 });
        firstGenerate = false;
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
        });
      });
      return paths;
    }

    it('does not start the next generation until the worker reports itself idle', async () => {
      const state = { busy: true };
      const paths = stallingWorker(state);
      const controller = new AbortController();
      const backend = new StreamBackend({ url: 'http://127.0.0.1:8790', settleTimeoutMs: 5000 });

      const first = backend.generate(req, controller.signal);
      controller.abort();
      await expect(first).rejects.toBeInstanceOf(AbortedError);
      expect(paths).toContain('/cancel');

      // The worker is still finishing the abandoned job, so the next request
      // must wait rather than queue behind it.
      const next = backend.generate(req, new AbortController().signal);
      const raced = await Promise.race([
        next.then(() => 'done'),
        new Promise((resolve) => setTimeout(() => resolve('pending'), 400)),
      ]);
      expect(raced).toBe('pending');

      state.busy = false;
      await expect(next).resolves.toBeInstanceOf(Buffer);
    });

    it('gives up waiting rather than blocking the room forever', async () => {
      // busy never clears: the deadline must win.
      stallingWorker({ busy: true });
      const controller = new AbortController();
      const backend = new StreamBackend({ url: 'http://127.0.0.1:8790', settleTimeoutMs: 600 });

      const first = backend.generate(req, controller.signal);
      controller.abort();
      await expect(first).rejects.toBeInstanceOf(AbortedError);

      const started = Date.now();
      await expect(backend.generate(req, new AbortController().signal)).resolves.toBeInstanceOf(Buffer);
      expect(Date.now() - started).toBeGreaterThanOrEqual(500);
    });
  });

  describe('health', () => {
    it('reports warmth, capacity and the running job', async () => {
      stubFetch(
        () =>
          new Response(
            JSON.stringify({ ok: true, warm: true, max_size: 1024, busy: true, backend: 'sdxl', current_request_id: 'abc' }),
            { status: 200 },
          ),
      );
      const health = await new StreamBackend({ url: 'http://127.0.0.1:8790' }).health();
      expect(health).toMatchObject({ ok: true, warm: true, maxSize: 1024, busy: true, currentRequestId: 'abc' });
    });

    it('explains why an unusable worker was rejected', async () => {
      stubFetch(() => new Response('nope', { status: 503 }));
      const health = await new StreamBackend({ url: 'http://127.0.0.1:8790' }).health();
      expect(health.ok).toBe(false);
      expect(health.reason).toMatch(/503/);
    });

    it('treats a worker that answers ok:false as unusable', async () => {
      stubFetch(() => new Response(JSON.stringify({ ok: false }), { status: 200 }));
      expect((await new StreamBackend({ url: 'http://127.0.0.1:8790' }).health()).ok).toBe(false);
    });
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
