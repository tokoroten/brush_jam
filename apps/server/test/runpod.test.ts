import { afterEach, describe, expect, it, vi } from 'vitest';
import { RunpodBackend } from '../src/ai/backends/runpod.js';
import { AbortedError, BackendHttpError, type GenerateRequest } from '../src/ai/backends/types.js';

const PNG = Buffer.from('fake-png-bytes');

const req: GenerateRequest = {
  prompt: 'anime style, fantasy town',
  negativePrompt: 'lowres',
  imagePng: PNG,
  maskPng: Buffer.from('fake-mask-bytes'),
  size: 768,
  denoise: 0.65,
  steps: 4,
  seed: 424242,
  profile: 'fast',
  tag: 'room1_r7',
};

function backend(extra: Partial<ConstructorParameters<typeof RunpodBackend>[0]> = {}): RunpodBackend {
  return new RunpodBackend({
    endpointId: 'ep1',
    apiKey: 'key',
    checkpoint: 'waiNSFWIllustrious_v150.safetensors',
    fastLora: 'dmd2_sdxl_4step_lora_fp16.safetensors',
    vaeTile: 512,
    pollIntervalMs: 1,
    timeoutMs: 5000,
    ...extra,
  });
}

interface Call {
  url: string;
  method: string;
  body: unknown;
  auth: string | undefined;
}

/** Answers every RunPod path from a scripted list of responses. */
function stubFetch(script: (path: string, call: number) => Response | Promise<Response>): Call[] {
  const calls: Call[] = [];
  let n = 0;
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    const url = String(input);
    const headers = new Headers((init?.headers ?? {}) as Record<string, string>);
    calls.push({
      url,
      method: init?.method ?? 'GET',
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : undefined,
      auth: headers.get('authorization') ?? undefined,
    });
    return script(new URL(url).pathname, n++);
  });
  return calls;
}

const completed = (data: string): Response =>
  new Response(JSON.stringify({ id: 'j1', status: 'COMPLETED', output: { images: [{ filename: 'x.png', type: 'base64', data }] } }), {
    status: 200,
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('RunpodBackend', () => {
  it('posts the worker-comfyui envelope to /runsync and returns the base64 image', async () => {
    const calls = stubFetch(() => completed(PNG.toString('base64')));
    const out = await backend().generate(req, new AbortController().signal);

    expect(out.equals(PNG)).toBe(true);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe('https://api.runpod.ai/v2/ep1/runsync');
    expect(calls[0]!.method).toBe('POST');
    expect(calls[0]!.auth).toBe('Bearer key');

    const body = calls[0]!.body as { input: { workflow: Record<string, { class_type: string; inputs: Record<string, unknown> }>; images: Array<{ name: string; image: string }> } };
    // Two images, referenced by name from the two LoadImage nodes.
    expect(body.input.images).toHaveLength(2);
    expect(body.input.images[0]!.image).toBe(PNG.toString('base64'));
    expect(body.input.workflow['4']!.inputs.image).toBe(body.input.images[0]!.name);
    expect(body.input.workflow['5']!.inputs.image).toBe(body.input.images[1]!.name);
    // The fast profile really loads the LoRA (same builder as the local backend).
    expect(body.input.workflow['12']!.class_type).toBe('LoraLoader');
    expect(body.input.workflow['10']!.class_type).toBe('VAEDecodeTiled');
    expect(body.input.workflow['1']!.inputs.ckpt_name).toBe('waiNSFWIllustrious_v150.safetensors');
  });

  it('falls back to /status polling when /runsync returns before the job finishes', async () => {
    const calls = stubFetch((path, n) => {
      if (path.endsWith('/runsync')) return new Response(JSON.stringify({ id: 'j1', status: 'IN_QUEUE' }), { status: 200 });
      if (n < 3) return new Response(JSON.stringify({ id: 'j1', status: 'IN_PROGRESS' }), { status: 200 });
      return completed(PNG.toString('base64'));
    });
    const out = await backend().generate(req, new AbortController().signal);

    expect(out.equals(PNG)).toBe(true);
    expect(calls.map((c) => new URL(c.url).pathname)).toEqual([
      '/v2/ep1/runsync',
      '/v2/ep1/status/j1',
      '/v2/ep1/status/j1',
      '/v2/ep1/status/j1',
    ]);
  });

  it('reports a FAILED job with the worker error, not an empty-image message', async () => {
    stubFetch(() => new Response(JSON.stringify({ id: 'j1', status: 'FAILED', error: 'ckpt not found' }), { status: 200 }));
    await expect(backend().generate(req, new AbortController().signal)).rejects.toThrow(/FAILED: ckpt not found/);
  });

  it('surfaces the HTTP status so the scheduler can tell 4xx from 5xx', async () => {
    stubFetch(() => new Response('nope', { status: 401 }));
    const err = await backend()
      .generate(req, new AbortController().signal)
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(BackendHttpError);
    expect((err as BackendHttpError).status).toBe(401);
  });

  it('cancels the job when the caller aborts mid-poll', async () => {
    const controller = new AbortController();
    const calls = stubFetch((path) => {
      if (path.endsWith('/runsync')) return new Response(JSON.stringify({ id: 'j1', status: 'IN_QUEUE' }), { status: 200 });
      if (path.includes('/status/')) {
        controller.abort();
        return new Response(JSON.stringify({ id: 'j1', status: 'IN_PROGRESS' }), { status: 200 });
      }
      return new Response('{}', { status: 200 });
    });
    await expect(backend().generate(req, controller.signal)).rejects.toBeInstanceOf(AbortedError);
    expect(calls.some((c) => c.url.endsWith('/cancel/j1') && c.method === 'POST')).toBe(true);
  });

  it('times out rather than polling forever, and cancels what it abandons', async () => {
    const calls = stubFetch((path) => {
      if (path.endsWith('/runsync')) return new Response(JSON.stringify({ id: 'j1', status: 'IN_QUEUE' }), { status: 200 });
      return new Response(JSON.stringify({ id: 'j1', status: 'IN_PROGRESS' }), { status: 200 });
    });
    await expect(backend({ timeoutMs: 30 }).generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
    expect(calls.some((c) => c.url.endsWith('/cancel/j1'))).toBe(true);
  });

  it('refuses an s3_url output instead of decoding a URL as base64', async () => {
    stubFetch(() =>
      new Response(JSON.stringify({ id: 'j1', status: 'COMPLETED', output: { images: [{ type: 's3_url', data: 'https://s3/x.png' }] } }), {
        status: 200,
      }),
    );
    await expect(backend().generate(req, new AbortController().signal)).rejects.toThrow(/S3 upload/);
  });

  it('offers both profiles with a LoRA and only quality without one', async () => {
    expect((await backend().capabilities()).profiles).toEqual(['fast', 'quality']);
    expect((await backend({ fastLora: undefined }).capabilities()).profiles).toEqual(['quality']);
    // DMD2 runs at cfg 1.0, where the negative branch is never evaluated.
    expect((await backend().capabilities()).negativePromptActive).toEqual({ fast: false, quality: true });
  });
});
