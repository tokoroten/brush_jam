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

function stubFetch(historyPages: unknown[]): { calls: Recorded[]; bodies: unknown[] } {
  const calls: Recorded[] = [];
  const bodies: unknown[] = [];
  let historyIndex = 0;
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    calls.push({ url: String(input), init });
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

  it('aborts and interrupts when the signal fires', async () => {
    const { calls } = stubFetch([{}]);
    const backend = new ComfyUIBackend({ url: 'http://127.0.0.1:8188', checkpoint: 'c.safetensors', pollIntervalMs: 5 });
    const controller = new AbortController();
    const promise = backend.generate(req, controller.signal);
    setTimeout(() => controller.abort(), 10);
    await expect(promise).rejects.toThrow(/aborted/);
    expect(calls.some((c) => c.url.endsWith('/interrupt'))).toBe(true);
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
