import { afterEach, describe, expect, it, vi } from 'vitest';
import { createBackend, PROBE_TIMEOUT_MS } from '../src/ai/backends/index.js';
import { loadConfig } from '../src/config.js';

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
