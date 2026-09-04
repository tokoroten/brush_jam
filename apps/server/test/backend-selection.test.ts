import { afterEach, describe, expect, it, vi } from 'vitest';
import { createBackend, PROBE_TIMEOUT_MS } from '../src/ai/backends/index.js';
import { loadConfig } from '../src/config.js';

interface Probes {
  stream?: 'ok' | 'loading' | 'error' | 'hang';
  comfy?: 'ok' | 'error' | 'hang';
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
      return new Response(JSON.stringify({ ok: true, warm: answer === 'ok' }), { status: 200 });
    }
    return new Response('{}', { status: 200 });
  });
  return { paths, timeouts };
}

const config = (env: Record<string, string> = {}): ReturnType<typeof loadConfig> => loadConfig(env as NodeJS.ProcessEnv);

afterEach(() => vi.unstubAllGlobals());

describe('backend selection', () => {
  it('honours an explicit AI_BACKEND=stream without probing anything', async () => {
    const { paths } = stub({});
    const logs: string[] = [];
    const backend = await createBackend(config({ AI_BACKEND: 'stream' }), (m) => logs.push(m));
    expect(backend.name).toBe('stream');
    expect(paths).toHaveLength(0);
    expect(logs[0]).toContain('AI_BACKEND=stream');
    expect(logs[0]).toContain('http://127.0.0.1:8790');
  });

  it('honours an explicit AI_BACKEND=comfyui even when the worker is up', async () => {
    stub({ stream: 'ok', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config({ AI_BACKEND: 'comfyui' }), (m) => logs.push(m))).name).toBe('comfyui');
    expect(logs[0]).toContain('AI_BACKEND=comfyui');
  });

  it('prefers the stream worker in auto mode', async () => {
    const { paths } = stub({ stream: 'ok', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config(), (m) => logs.push(m))).name).toBe('stream');
    expect(paths).toEqual(['8790/healthz']);
    expect(logs[0]).toContain('auto-detected');
  });

  it('accepts a worker that is up but still loading its model', async () => {
    stub({ stream: 'loading', comfy: 'ok' });
    expect((await createBackend(config(), () => {})).name).toBe('stream');
  });

  it('falls back to ComfyUI when the worker does not answer', async () => {
    const { paths } = stub({ stream: 'error', comfy: 'ok' });
    const logs: string[] = [];
    expect((await createBackend(config(), (m) => logs.push(m))).name).toBe('comfyui');
    expect(paths).toEqual(['8790/healthz', '8188/system_stats']);
    expect(logs[0]).toContain('no stream worker');
  });

  it('falls back to the mock when neither answers, naming both', async () => {
    stub({ stream: 'error', comfy: 'error' });
    const logs: string[] = [];
    expect((await createBackend(config(), (m) => logs.push(m))).name).toBe('mock');
    expect(logs[0]).toContain('127.0.0.1:8790');
    expect(logs[0]).toContain('127.0.0.1:8188');
  });

  it('uses a custom STREAM_URL', async () => {
    const { paths } = stub({ stream: 'ok' });
    await createBackend(config({ STREAM_URL: 'http://127.0.0.1:9999/' }), () => {});
    expect(paths).toEqual(['9999/healthz']);
  });

  it('does not hang startup when a probe stalls', async () => {
    stub({ stream: 'hang', comfy: 'error' });
    const started = Date.now();
    const backend = await createBackend(config(), () => {});
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

  it('accepts AI_BACKEND=stream and still rejects nonsense', () => {
    expect(config({ AI_BACKEND: 'stream' }).aiBackend).toBe('stream');
    expect(() => config({ AI_BACKEND: 'sideways' })).toThrow(/AI_BACKEND/);
  });
});
