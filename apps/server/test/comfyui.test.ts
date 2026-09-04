import { afterEach, describe, expect, it, vi } from 'vitest';
import { ComfyUIBackend, buildWorkflow, comfyReachable } from '../src/ai/backends/comfyui.js';
import type { GenerateRequest } from '../src/ai/backends/types.js';

const req: GenerateRequest = {
  prompt: 'anime style, fantasy town',
  negativePrompt: 'lowres',
  imagePng: Buffer.from('image-bytes'),
  maskPng: Buffer.from('mask-bytes'),
  size: 1024,
  denoise: 0.55,
  steps: 14,
  seed: 12345,
  tag: 'room1_r7',
};

interface Recorded { url: string; init?: RequestInit }

interface StubOptions {
  /** What /queue reports as currently running. */
  running?: string[];
  /** What /queue reports as waiting. */
  pending?: string[];
  /** Paths that never resolve, to exercise the request timeouts. */
  hang?: string[];
  /** How many /history polls fail before the stub starts answering. */
  historyFailures?: number;
}

function stubFetch(historyPages: unknown[], options: StubOptions = {}): { calls: Recorded[]; bodies: unknown[] } {
  const calls: Recorded[] = [];
  const bodies: unknown[] = [];
  let historyIndex = 0;
  let historyFailures = options.historyFailures ?? 0;
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    const path = new URL(String(input)).pathname;
    if (options.hang?.some((h) => path.startsWith(h))) {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'TimeoutError' })), { once: true });
      });
    }
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
    if (path === '/interrupt') return new Response('{}', { status: 200 });
    if (String(input).includes('/upload/image')) {
      const form = init!.body as FormData;
      const file = form.get('image') as File;
      bodies.push({ upload: file.name, overwrite: form.get('overwrite') });
      return new Response(JSON.stringify({ name: file.name, subfolder: '', type: 'input' }), { status: 200 });
    }
    if (String(input).endsWith('/prompt')) {
      bodies.push(JSON.parse(String(init!.body)));
      return new Response(JSON.stringify({ prompt_id: 'pid-1' }), { status: 200 });
    }
    if (String(input).includes('/history/')) {
      if (historyFailures > 0) {
        historyFailures -= 1;
        throw Object.assign(new Error('fetch failed'), { name: 'TypeError' });
      }
      const page = historyPages[Math.min(historyIndex, historyPages.length - 1)];
      historyIndex += 1;
      return new Response(JSON.stringify(page), { status: 200 });
    }
    if (String(input).includes('/view?')) {
      return new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 });
    }
    return new Response('{}', { status: 200 });
  });
  return { calls, bodies };
}

afterEach(() => vi.unstubAllGlobals());

describe('buildWorkflow', () => {
  const wf = buildWorkflow({
    checkpoint: 'waiNSFWIllustrious_v150.safetensors',
    prompt: 'a town',
    negativePrompt: 'lowres',
    imageName: 'img.png',
    maskName: 'mask.png',
    seed: 7,
    steps: 14,
    cfg: 5.5,
    denoise: 0.55,
    filenamePrefix: 'brushjam/room1',
  });

  it('wires the img2img-with-noise-mask graph', () => {
    expect((wf['1'] as never as { inputs: { ckpt_name: string } }).inputs.ckpt_name).toBe('waiNSFWIllustrious_v150.safetensors');
    expect((wf['8'] as never as { class_type: string }).class_type).toBe('SetLatentNoiseMask');
    expect((wf['8'] as never as { inputs: unknown }).inputs).toEqual({ samples: ['7', 0], mask: ['6', 0] });
    expect((wf['6'] as never as { inputs: { channel: string } }).inputs.channel).toBe('red');
    expect((wf['7'] as never as { class_type: string }).class_type).toBe('VAEEncode');
  });

  it('appends the quality suffix to the positive prompt only', () => {
    expect((wf['2'] as never as { inputs: { text: string } }).inputs.text).toBe('a town, masterpiece, best quality');
    expect((wf['3'] as never as { inputs: { text: string } }).inputs.text).toBe('lowres');
  });

  it('passes sampler settings through', () => {
    expect((wf['9'] as never as { inputs: Record<string, unknown> }).inputs).toMatchObject({
      seed: 7, steps: 14, cfg: 5.5, denoise: 0.55, sampler_name: 'euler_ancestral', scheduler: 'normal',
    });
  });
});

describe('ComfyUIBackend', () => {
  it('uploads, queues, polls and fetches the output in order', async () => {
    const { calls, bodies } = stubFetch([
      {},
      { 'pid-1': { outputs: { '11': { images: [{ filename: 'out.png', subfolder: 'brushjam', type: 'output' }] } } } },
    ]);
    const backend = new ComfyUIBackend({ url: 'http://127.0.0.1:8188/', checkpoint: 'ckpt.safetensors', pollIntervalMs: 1 });
    const out = await backend.generate(req, new AbortController().signal);

    const paths = calls.map((c) => new URL(c.url).pathname);
    expect(paths.slice(0, 3)).toEqual(['/upload/image', '/upload/image', '/prompt']);
    expect(paths.filter((p) => p.startsWith('/history/'))).toHaveLength(2);
    expect(paths[paths.length - 1]).toBe('/view');
    expect(out).toEqual(Buffer.from([1, 2, 3, 4]));

    const query = new URL(calls[calls.length - 1]!.url).searchParams;
    expect(query.get('filename')).toBe('out.png');
    expect(query.get('subfolder')).toBe('brushjam');
    expect(query.get('type')).toBe('output');
  });

  it('sends the workflow with the uploaded filenames and a client id', async () => {
    const { bodies } = stubFetch([{ 'pid-1': { outputs: { '11': { images: [{ filename: 'o.png', subfolder: '', type: 'output' }] } } } }]);
    const backend = new ComfyUIBackend({ url: 'http://127.0.0.1:8188', checkpoint: 'ckpt.safetensors', pollIntervalMs: 1 });
    await backend.generate(req, new AbortController().signal);

    const uploads = bodies.filter((b) => (b as { upload?: string }).upload) as Array<{ upload: string; overwrite: string }>;
    expect(uploads).toHaveLength(2);
    expect(uploads[0]!.overwrite).toBe('true');
    expect(uploads[0]!.upload).toContain('_img.png');
    expect(uploads[1]!.upload).toContain('_mask.png');

    const queued = bodies.find((b) => (b as { prompt?: unknown }).prompt) as { prompt: Record<string, { inputs: Record<string, unknown> }>; client_id: string };
    expect(queued.client_id).toMatch(/[0-9a-f-]{36}/);
    expect(queued.prompt['4']!.inputs.image).toBe(uploads[0]!.upload);
    expect(queued.prompt['5']!.inputs.image).toBe(uploads[1]!.upload);
    expect(queued.prompt['9']!.inputs.seed).toBe(12345);
  });

  it('throws when ComfyUI reports an execution error', async () => {
    stubFetch([{ 'pid-1': { status: { status_str: 'error' } } }]);
    const backend = new ComfyUIBackend({ url: 'http://127.0.0.1:8188', checkpoint: 'c.safetensors', pollIntervalMs: 1 });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/execution error/);
  });

  it('interrupts only when our prompt is the one running', async () => {
    const { calls } = stubFetch([{}], { running: ['pid-1'] });
    const backend = new ComfyUIBackend({ url: 'http://127.0.0.1:8188', checkpoint: 'c.safetensors', pollIntervalMs: 5 });
    const controller = new AbortController();
    const promise = backend.generate(req, controller.signal);
    setTimeout(() => controller.abort(), 10);
    await expect(promise).rejects.toThrow(/aborted/);
    expect(calls.some((c) => c.url.endsWith('/interrupt'))).toBe(true);
  });

  it('does not interrupt somebody else running job (finding 7)', async () => {
    const { calls } = stubFetch([{}], { running: ['someone-elses-prompt'] });
    const backend = new ComfyUIBackend({ url: 'http://127.0.0.1:8188', checkpoint: 'c.safetensors', pollIntervalMs: 5 });
    const controller = new AbortController();
    const promise = backend.generate(req, controller.signal);
    setTimeout(() => controller.abort(), 10);
    await expect(promise).rejects.toThrow(/aborted/);
    expect(calls.some((c) => c.url.endsWith('/queue'))).toBe(true);
    expect(calls.some((c) => c.url.endsWith('/interrupt'))).toBe(false);
  });

  it.each(['/upload/image', '/prompt', '/view'])('times out a stalled %s instead of wedging', async (path) => {
    stubFetch([{ 'pid-1': { outputs: { '11': { images: [{ filename: 'o.png', subfolder: '', type: 'output' }] } } } }], { hang: [path] });
    const backend = new ComfyUIBackend({
      url: 'http://127.0.0.1:8188',
      checkpoint: 'c.safetensors',
      pollIntervalMs: 1,
      requestTimeoutMs: 30,
    });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
  });

  it('keeps polling a stalled /history until the generation deadline (finding B3)', async () => {
    const { calls } = stubFetch([{ 'pid-1': { outputs: { '11': { images: [{ filename: 'o.png', subfolder: '', type: 'output' }] } } } }], {
      hang: ['/history'],
    });
    const backend = new ComfyUIBackend({
      url: 'http://127.0.0.1:8188',
      checkpoint: 'c.safetensors',
      pollIntervalMs: 1,
      requestTimeoutMs: 20,
      timeoutMs: 120,
    });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/timed out/);
    // retried rather than failing on the first stall
    expect(calls.filter((c) => new URL(c.url).pathname.startsWith('/history/')).length).toBeGreaterThan(1);
  });

  it('rejects a response without a prompt_id', async () => {
    vi.stubGlobal('fetch', async (input: string) => {
      const path = new URL(String(input)).pathname;
      if (path === '/upload/image') return new Response(JSON.stringify({ name: 'x.png', subfolder: '', type: 'input' }), { status: 200 });
      if (path === '/prompt') return new Response(JSON.stringify({}), { status: 200 });
      return new Response('{}', { status: 200 });
    });
    const backend = new ComfyUIBackend({ url: 'http://127.0.0.1:8188', checkpoint: 'c.safetensors' });
    await expect(backend.generate(req, new AbortController().signal)).rejects.toThrow(/prompt_id/);
  });

  it('probes reachability without throwing', async () => {
    vi.stubGlobal('fetch', async () => new Response('{}', { status: 200 }));
    expect(await comfyReachable('http://127.0.0.1:8188')).toBe(true);
    vi.stubGlobal('fetch', async () => {
      throw new Error('ECONNREFUSED');
    });
    expect(await comfyReachable('http://127.0.0.1:8188')).toBe(false);
  });
});
