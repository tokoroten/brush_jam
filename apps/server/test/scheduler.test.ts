import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Rect, ServerMessage } from '@brushjam/shared';
import { AIScheduler, type MaskHandle, type SchedulerHost } from '../src/ai/scheduler.js';
import type { AIBackend, GenerateRequest } from '../src/ai/backends/index.js';

class FakeBackend implements AIBackend {
  readonly name = 'fake';
  readonly calls: GenerateRequest[] = [];
  private resolvers: Array<(v: Buffer) => void> = [];
  private rejecters: Array<(e: Error) => void> = [];
  failNext = false;

  generate(req: GenerateRequest): Promise<Buffer> {
    this.calls.push(req);
    if (this.failNext) {
      this.failNext = false;
      return Promise.reject(new Error('gpu on fire'));
    }
    return new Promise<Buffer>((resolve, reject) => {
      this.resolvers.push(resolve);
      this.rejecters.push(reject);
    });
  }

  async finish(): Promise<void> {
    const resolve = this.resolvers.shift();
    this.rejecters.shift();
    resolve?.(Buffer.from('patch'));
    await vi.advanceTimersByTimeAsync(0);
  }

  get inFlight(): number {
    return this.resolvers.length;
  }
}

function makeHost(revision = { value: 10 }): {
  host: SchedulerHost;
  emitted: ServerMessage[];
  applied: Array<{ crop: Rect; forRevision: number }>;
  revision: { value: number };
  maskEmpty: { value: boolean };
} {
  const emitted: ServerMessage[] = [];
  const applied: Array<{ crop: Rect; forRevision: number }> = [];
  const maskEmpty = { value: false };
  const host: SchedulerHost = {
    getPrompt: () => 'a prompt',
    getRevision: () => revision.value,
    renderInput: async () => Buffer.from('input'),
    buildMask: (): MaskHandle => ({ png: Buffer.from('mask'), alpha: {}, empty: maskEmpty.value }),
    applyResult: async (_patch, crop, _mask, forRevision) => {
      applied.push({ crop, forRevision });
      return { rect: crop, url: '/patch.png' };
    },
    emit: (msg) => emitted.push(msg),
  };
  return { host, emitted, applied, revision, maskEmpty };
}

const opts = { window: 1024, apply: 768, steps: 14, denoise: 0.55, debounceMs: 400, canvasSize: 4096, errorBackoffMs: 2000 };
const R = (x: number, y: number, w = 20, h = 20): Rect => ({ x, y, width: w, height: h });

describe('AIScheduler', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('debounces activity into a single generation', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(200);
    s.markDirty([R(2050, 2050)]);
    await vi.advanceTimersByTimeAsync(399);
    expect(backend.calls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(2);
    expect(backend.calls).toHaveLength(1);
    s.stop();
  });

  it('keeps at most one request in flight and re-runs once afterwards', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls).toHaveLength(1);

    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(1000);
    expect(backend.calls).toHaveLength(1);
    expect(backend.inFlight).toBe(1);

    await backend.finish();
    await vi.advanceTimersByTimeAsync(10);
    expect(backend.calls).toHaveLength(2);
    s.stop();
  });

  it('centers the crop on the most recent dirty region', async () => {
    const { host, applied } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(200, 200)]);
    s.markDirty([R(3000, 3000)]);
    await vi.advanceTimersByTimeAsync(400);
    await backend.finish();
    expect(applied[0]!.crop).toEqual({ x: 2498, y: 2498, width: 1024, height: 1024 });
    s.stop();
  });

  it('discards a stale result', async () => {
    const { host, applied, revision } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);

    revision.value = 50;
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    revision.value = 60;
    s.markDirty([R(2000, 2000)]);
    await backend.finish();
    await vi.advanceTimersByTimeAsync(10);
    expect(applied).toHaveLength(1);
    expect(applied[0]!.forRevision).toBe(50);

    // second run is for revision 60; make its result arrive "after" a newer accept
    expect(backend.calls).toHaveLength(2);
    revision.value = 40;
    s.markDirty([R(2000, 2000)]);
    await backend.finish();
    await vi.advanceTimersByTimeAsync(10);
    expect(applied.map((a) => a.forRevision)).toEqual([50, 60]);
    s.stop();
  });

  it('emits generating then idle with a latency', async () => {
    const { host, emitted } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    await backend.finish();
    const states = emitted.filter((m) => m.t === 'ai_status').map((m) => (m as { state: string }).state);
    expect(states).toEqual(['queued', 'generating', 'idle']);
    expect(emitted.some((m) => m.t === 'ai_result')).toBe(true);
    s.stop();
  });

  it('backs off after an error and keeps the dirty region', async () => {
    const { host, emitted } = makeHost();
    const backend = new FakeBackend();
    backend.failNext = true;
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    await vi.advanceTimersByTimeAsync(10);
    const error = emitted.find((m) => m.t === 'ai_status' && m.state === 'error') as { message?: string } | undefined;
    expect(error?.message).toBe('gpu on fire');
    expect(s.dirtyRegions).toHaveLength(1);
    expect(backend.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1500);
    expect(backend.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(600);
    expect(backend.calls).toHaveLength(2);
    s.stop();
  });

  it('clears the repainted part of the dirty region after a result', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    await backend.finish();
    expect(s.dirtyRegions).toHaveLength(0);
    s.stop();
  });

  it('keeps the parts of a large region that were not repainted', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([{ x: 1000, y: 1000, width: 2000, height: 2000 }]);
    await vi.advanceTimersByTimeAsync(400);
    await backend.finish();
    expect(s.dirtyRegions.length).toBeGreaterThan(0);
    s.stop();
  });

  it('does nothing when the mask is empty', async () => {
    const { host, maskEmpty } = makeHost();
    maskEmpty.value = true;
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(500);
    expect(backend.calls).toHaveLength(0);
    expect(s.dirtyRegions).toHaveLength(0);
    s.stop();
  });

  it('stop() prevents further runs', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    s.stop();
    await vi.advanceTimersByTimeAsync(1000);
    expect(backend.calls).toHaveLength(0);
  });
});
