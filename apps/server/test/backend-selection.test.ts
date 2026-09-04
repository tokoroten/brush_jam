import { afterEach, describe, expect, it, vi } from 'vitest';
import { ComfyUIBackend, createBackend, MockBackend, PROBE_TIMEOUT_MS, StreamBackend } from '../src/ai/backends/index.js';
import { loadConfig, resolveBackendConfig } from '../src/config.js';

interface Probes {
  stream?: 'ok' | 'loading' | 'error' | 'hang';
  comfy?: 'ok' | 'error' | 'hang';
  /** What the worker says it can generate; 0 means it does not say. */
  maxSize?: number;
}

function stub(probes: Probes): { paths: string[]; timeouts: number[] } {
  const paths: string[] = [];
  const timeouts: number[] = [];
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    const url = new URL(String(input));
    paths.push(`${url.port}${url.pathname}`);
    // record how long the caller was prepared to wait
    const signal = init?.signal as (AbortSignal & { _t?: number }) | undefined;
    if (signal) timeouts.push(1);

    const answer = url.pathname === '/healthz' ? probes.stream : probes.comfy;
    if (answer === 'hang') {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('probe timed out'), { name: 'TimeoutError' })), {
          once: true,
        });
      });
    }
    if (answer === 'error' || answer === undefined) throw new Error('connection refused');
    if (url.pathname === '/healthz') {
      const maxSize = probes.maxSize ?? 1024;
      return new Response(JSON.stringify({ ok: true, warm: answer === 'ok', max_size: maxSize, busy: false }), {
        status: 200,
      });
    }
    return new Response('{}', { status: 200 });
  });
  return { paths, timeouts };
}

const config = (env: Record<string, string> = {}): ReturnType<typeof loadConfig> => loadConfig(env as NodeJS.ProcessEnv);

afterEach(() => vi.unstubAllGlobals());

describe('backend selection', () => {
  it('honours an explicit AI_BACKEND=stream even when the worker is down', async () => {
    stub({ stream: 'error' });
    const logs: string[] = [];
    const backend = await createBackend(config({ AI_BACKEND: 'stream' }), (m) => logs.push(m));
    expect(backend.name).toBe('stream');
    expect(logs[0]).toContain('AI_BACKEND=stream');
    expect(logs[0]).toContain('http://127.0.0.1:8790');
    // the choice is honoured, but the operator is told why it will fail
    expect(logs.join('\n')).toMatch(/warning: stream worker is not answering/);
  });

  it('warns when an explicitly chosen worker is too small for AI_WINDOW', async () => {
    stub({ stream: 'ok', maxSize: 512 });
    const logs: string[] = [];
    await createBackend(config({ AI_BACKEND: 'stream', AI_WINDOW: '1024' }), (m) => logs.push(m));
    expect(logs.join('\n')).toMatch(/max_size 512 is below AI_WINDOW 1024/);
  });

  it('honours an explicit AI_BACKEND=comfyui even when the worker is up', async () => {
    stub({ stream: 'ok', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config({ AI_BACKEND: 'comfyui' }), (m) => logs.push(m))).name).toBe('comfyui');
    expect(logs[0]).toContain('AI_BACKEND=comfyui');
  });

  // Review 6 finding 1: a reachable worker is not a usable one, and on this
  // card its being up at all means ComfyUI is starved. Auto leaves it alone.
  it('does not auto-select the stream worker even when it is warm', async () => {
    const { paths } = stub({ stream: 'ok', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config(), (m) => logs.push(m))).name).toBe('comfyui');
    expect(paths).toEqual(['8188/system_stats']);
    expect(logs[0]).toContain('stream is explicit-only');
  });

  it('considers the worker in auto only with AI_STREAM_AUTO=1', async () => {
    const { paths } = stub({ stream: 'ok', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config({ AI_STREAM_AUTO: '1' }), (m) => logs.push(m))).name).toBe('stream');
    expect(paths).toEqual(['8790/healthz']);
    expect(logs[0]).toContain('auto-detected');
  });

  it('skips a worker that is up but has no model loaded', async () => {
    stub({ stream: 'loading', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config({ AI_STREAM_AUTO: '1' }), (m) => logs.push(m))).name).toBe('comfyui');
    expect(logs.join('\n')).toMatch(/not warm/);
  });

  it('skips a worker capped below the configured window', async () => {
    stub({ stream: 'ok', comfy: 'ok', maxSize: 512 });
    const logs: string[] = [];
    const backend = await createBackend(config({ AI_STREAM_AUTO: '1', AI_WINDOW: '1024' }), (m) => logs.push(m));
    expect(backend.name).toBe('comfyui');
    expect(logs.join('\n')).toMatch(/max_size 512 < AI_WINDOW 1024/);
  });

  it('accepts a big enough worker', async () => {
    stub({ stream: 'ok', comfy: 'ok', maxSize: 1024 });
    const backend = await createBackend(config({ AI_STREAM_AUTO: '1', AI_WINDOW: '1024' }), () => {});
    expect(backend.name).toBe('stream');
  });

  it('falls back to ComfyUI when an opted-in worker does not answer', async () => {
    const { paths } = stub({ stream: 'error', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config({ AI_STREAM_AUTO: '1' }), (m) => logs.push(m))).name).toBe('comfyui');
    expect(paths).toEqual(['8790/healthz', '8188/system_stats']);
    expect(logs.join('\n')).toMatch(/skipping stream worker/);
  });

  it('falls back to the mock when ComfyUI does not answer', async () => {
    stub({ stream: 'error', comfy: 'error' });
    const logs: string[] = [];
    expect((await createBackend(config(), (m) => logs.push(m))).name).toBe('mock');
    expect(logs[0]).toContain('127.0.0.1:8188');
  });

  it('uses a custom STREAM_URL', async () => {
    const { paths } = stub({ stream: 'ok' });
    await createBackend(config({ AI_STREAM_AUTO: '1', STREAM_URL: 'http://127.0.0.1:9999/' }), () => {});
    expect(paths).toEqual(['9999/healthz']);
  });

  it('does not hang startup when a probe stalls', async () => {
    stub({ stream: 'hang', comfy: 'error' });
    // with AI_STREAM_AUTO the stalling worker is probed and must time out
    const started = Date.now();
    const backend = await createBackend(config({ AI_STREAM_AUTO: '1' }), () => {});
    // the probe aborts itself; without a deadline this would never return
    expect(Date.now() - started).toBeLessThan(PROBE_TIMEOUT_MS + 1500);
    expect(backend.name).toBe('mock');
  }, 10_000);

  it('still selects mock and runpod explicitly', async () => {
    stub({ stream: 'ok' });
    expect((await createBackend(config({ AI_BACKEND: 'mock' }), () => {})).name).toBe('mock');
    const runpod = config({ AI_BACKEND: 'runpod', RUNPOD_ENDPOINT_ID: 'e', RUNPOD_API_KEY: 'k' });
    expect((await createBackend(runpod, () => {})).name).toBe('runpod');
  });
});

describe('stream config', () => {
  it('defaults the worker url and timeout', () => {
    const c = config();
    expect(c.streamUrl).toBe('http://127.0.0.1:8790');
    expect(c.streamTimeoutMs).toBe(120_000);
  });

  it('trims a trailing slash and accepts overrides', () => {
    const c = config({ STREAM_URL: 'http://box:8790//', STREAM_TIMEOUT_MS: '30000' });
    expect(c.streamUrl).toBe('http://box:8790');
    expect(c.streamTimeoutMs).toBe(30_000);
  });

  it('rejects a malformed STREAM_URL and an out-of-range timeout', () => {
    expect(() => config({ STREAM_URL: 'not a url' })).toThrow(/STREAM_URL/);
    expect(() => config({ STREAM_TIMEOUT_MS: '10' })).toThrow(/STREAM_TIMEOUT_MS/);
  });

  it('keeps the worker out of auto unless AI_STREAM_AUTO is set', () => {
    expect(config().streamAuto).toBe(false);
    expect(config({ AI_STREAM_AUTO: '1' }).streamAuto).toBe(true);
    expect(config({ AI_STREAM_AUTO: 'yes' }).streamAuto).toBe(true);
    expect(config({ AI_STREAM_AUTO: '0' }).streamAuto).toBe(false);
  });

  it('accepts AI_BACKEND=stream and still rejects nonsense', () => {
    expect(config({ AI_BACKEND: 'stream' }).aiBackend).toBe('stream');
    expect(() => config({ AI_BACKEND: 'sideways' })).toThrow(/AI_BACKEND/);
  });
});

/**
 * A room may only offer what the running backend can do. The stream worker
 * holds one fused LCM LoRA: it has no quality profile at all, and asking for
 * 14 steps there would silently run 4.
 */
describe('backend capabilities', () => {
  it('comfyui offers both profiles when a LoRA is configured', async () => {
    const caps = await new ComfyUIBackend({ url: 'http://x', checkpoint: 'c', fastLora: 'lcm.safetensors' }).capabilities();
    expect(caps.profiles).toEqual(['fast', 'quality']);
    expect(caps.maxDenoise).toBe(0.95);
  });

  it('comfyui offers quality only without a LoRA', async () => {
    const caps = await new ComfyUIBackend({ url: 'http://x', checkpoint: 'c' }).capabilities();
    expect(caps.profiles).toEqual(['quality']);
  });

  it('the mock offers both', async () => {
    expect((await new MockBackend().capabilities()).profiles).toEqual(['fast', 'quality']);
  });

  it('the stream worker offers fast only, with its own limits', async () => {
    stub({ stream: 'ok', maxSize: 1024 });
    const caps = await new StreamBackend({ url: 'http://127.0.0.1:8790' }).capabilities();
    expect(caps.profiles).toEqual(['fast']);
    expect(caps.maxResolution).toBe(1024);
    // no max_denoise reported: fall back to the conservative default
    expect(caps.maxDenoise).toBe(0.9);
  });

  it('takes max_denoise from the worker when it reports one', async () => {
    vi.stubGlobal('fetch', async () =>
      new Response(JSON.stringify({ ok: true, warm: true, max_size: 768, max_denoise: 0.85 }), { status: 200 }),
    );
    const caps = await new StreamBackend({ url: 'http://127.0.0.1:8790' }).capabilities();
    expect(caps).toMatchObject({ profiles: ['fast'], maxResolution: 768, maxDenoise: 0.85 });
  });

  it('falls back to safe limits when the worker cannot be reached', async () => {
    stub({ stream: 'error' });
    const caps = await new StreamBackend({ url: 'http://127.0.0.1:8790' }).capabilities();
    expect(caps).toMatchObject({ profiles: ['fast'], maxResolution: 1024, maxDenoise: 0.9 });
  });

  it('reports the sampling settings the worker publishes', async () => {
    vi.stubGlobal('fetch', async () =>
      new Response(
        JSON.stringify({ ok: true, warm: true, max_size: 1024, steps: 4, guidance: 1.5, vae: 'fp16fix', model: 'sdxl' }),
        { status: 200 },
      ),
    );
    const health = await new StreamBackend({ url: 'http://127.0.0.1:8790' }).health();
    expect(health.sampling).toEqual({ steps: 4, guidance: 1.5, vae: 'fp16fix', model: 'sdxl', lora: undefined });
  });
});

/** The stream worker's defaults differ from ComfyUI's and must not be guessed. */
describe('backend defaults', () => {
  const caps = (over: Partial<{ profiles: ('fast' | 'quality')[]; maxResolution: number; maxDenoise: number }> = {}) => ({
    profiles: ['fast', 'quality'] as ('fast' | 'quality')[],
    maxResolution: 2048,
    maxDenoise: 0.95,
    ...over,
  });
  const streamCaps = caps({ profiles: ['fast'], maxResolution: 1024, maxDenoise: 0.9 });

  it('moves a stream server to 768 and denoise 0.8', () => {
    const c = resolveBackendConfig(config(), 'stream', streamCaps);
    expect(c.aiWindow).toBe(768);
    expect(c.aiDenoise).toBe(0.8);
    expect(c.aiProfile).toBe('fast');
  });

  it('leaves an explicit AI_WINDOW and AI_DENOISE alone', () => {
    const c = resolveBackendConfig(config({ AI_WINDOW: '512', AI_DENOISE: '0.6' }), 'stream', streamCaps);
    expect(c.aiWindow).toBe(512);
    expect(c.aiDenoise).toBe(0.6);
  });

  it('never generates larger than the canvas', () => {
    expect(resolveBackendConfig(config({ CANVAS_SIZE: '512' }), 'stream', streamCaps).aiWindow).toBe(512);
  });

  it('leaves comfyui and mock configuration untouched', () => {
    const before = config();
    expect(resolveBackendConfig(before, 'comfyui', caps())).toMatchObject({
      aiWindow: before.aiWindow,
      aiDenoise: before.aiDenoise,
      aiProfile: before.aiProfile,
    });
  });

  // Review 7 finding 6: defaults used to be applied AFTER validation, so they
  // could produce a combination that had never been validated together.
  it('validates the combination the server actually runs', () => {
    // Explicit apply 768 is fine against the default window of 768, but a
    // backend that only does 512 clamps the window under it. That combination
    // used to be applied after validation and run anyway.
    const cfg = config({ AI_MODE: 'patch', AI_APPLY: '768' });
    expect(() => resolveBackendConfig(cfg, 'stream', caps({ profiles: ['fast'], maxResolution: 512 }))).toThrow(/AI_APPLY/);
  });

  // Review 7 finding 6: an explicit choice is an instruction, not a hint.
  it('refuses an explicit profile the backend cannot run', () => {
    expect(() => resolveBackendConfig(config({ AI_PROFILE: 'quality' }), 'stream', streamCaps)).toThrow(
      /AI_PROFILE=quality is not supported/,
    );
  });

  it('falls back quietly when the profile was not pinned', () => {
    const c = resolveBackendConfig(config(), 'comfyui', caps({ profiles: ['quality'] }));
    expect(c.aiProfile).toBe('quality');
  });

  // Review 7 finding 1: an oversized window is a 400 on every generation.
  it('refuses an explicit AI_WINDOW the backend will not accept', () => {
    expect(() => resolveBackendConfig(config({ AI_WINDOW: '1024' }), 'stream', caps({ profiles: ['fast'], maxResolution: 768 }))).toThrow(
      /AI_WINDOW=1024 is larger than/,
    );
  });

  it('clamps an implicit window to what the backend accepts', () => {
    const c = resolveBackendConfig(config(), 'stream', caps({ profiles: ['fast'], maxResolution: 512, maxDenoise: 0.9 }));
    expect(c.aiWindow).toBe(512);
    expect(c.maxResolution).toBe(512);
  });

  it('caps denoise at the backend ceiling, or refuses an explicit one', () => {
    // the implicit 0.7 default is pulled down to a lower ceiling
    expect(resolveBackendConfig(config(), 'comfyui', caps({ maxDenoise: 0.6 })).aiDenoise).toBe(0.6);
    expect(() => resolveBackendConfig(config({ AI_DENOISE: '0.9' }), 'comfyui', caps({ maxDenoise: 0.75 }))).toThrow(
      /AI_DENOISE=0.9 is above/,
    );
  });

  // Review 7 finding 4: starting size and ceiling are different things.
  it('keeps a 1024 ceiling for a fast room so quality can still reach it', () => {
    const c = resolveBackendConfig(config(), 'comfyui', caps());
    expect(c.aiWindow).toBe(768);
    expect(c.maxResolution).toBe(1024);
  });

  it('treats an explicit AI_WINDOW as a hard ceiling', () => {
    const c = resolveBackendConfig(config({ AI_WINDOW: '768' }), 'comfyui', caps());
    expect(c.maxResolution).toBe(768);
  });

  it('never lets the ceiling exceed the backend', () => {
    expect(resolveBackendConfig(config(), 'stream', streamCaps).maxResolution).toBe(1024);
    expect(resolveBackendConfig(config(), 'stream', caps({ profiles: ['fast'], maxResolution: 768 })).maxResolution).toBe(768);
  });
});
