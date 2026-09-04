import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_NEGATIVE_PROMPT, type Rect, type ServerMessage } from '@brushjam/shared';
import { AIScheduler, type MaskHandle, type SchedulerHost } from '../src/ai/scheduler.js';
import { AbortedError, type AIBackend, type GenerateRequest } from '../src/ai/backends/index.js';

class FakeBackend implements AIBackend {
  readonly name = 'fake';
  readonly calls: GenerateRequest[] = [];
  private resolvers: Array<(v: Buffer) => void> = [];
  failNext = false;
  /** When true, generate() only settles if the abort signal fires. */
  hangForever = false;

  generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer> {
    this.calls.push(req);
    if (this.failNext) {
      this.failNext = false;
      return Promise.reject(new Error('gpu on fire'));
    }
    return new Promise<Buffer>((resolve, reject) => {
      signal.addEventListener('abort', () => reject(new AbortedError()), { once: true });
      if (!this.hangForever) this.resolvers.push(resolve);
    });
  }

  async finish(): Promise<void> {
    this.resolvers.shift()?.(Buffer.from('patch'));
    await vi.advanceTimersByTimeAsync(0);
  }

  get inFlight(): number {
    return this.resolvers.length;
  }
}

interface Harness {
  host: SchedulerHost;
  emitted: ServerMessage[];
  applied: Array<{ crop: Rect; apply: Rect; forRevision: number }>;
  masks: Array<{ crop: Rect; apply: Rect }>;
  revision: { value: number };
  prompt: { value: string };
  maskEmpty: { value: boolean };
  renderedAt: number[];
  logs: string[];
  settings: {
    denoise: number | undefined;
    negativePrompt: string | undefined;
    resolution: number | undefined;
    profile: 'fast' | 'quality' | undefined;
  };
  renderSizes: Array<{ rect: Rect; size: number }>;
}

function makeHost(): Harness {
  const emitted: ServerMessage[] = [];
  const applied: Array<{ crop: Rect; apply: Rect; forRevision: number }> = [];
  const masks: Array<{ crop: Rect; apply: Rect }> = [];
  const revision = { value: 10 };
  const prompt = { value: 'a prompt' };
  const settings = {
    denoise: undefined as number | undefined,
    negativePrompt: undefined as string | undefined,
    resolution: undefined as number | undefined,
    profile: undefined as 'fast' | 'quality' | undefined,
  };
  const renderSizes: Array<{ rect: Rect; size: number }> = [];
  const maskEmpty = { value: false };
  const renderedAt: number[] = [];
  const logs: string[] = [];
  const host: SchedulerHost = {
    getRevision: () => revision.value,
    beginJob: () => {
      const captured = {
        revision: revision.value,
        prompt: prompt.value,
        denoise: settings.denoise,
        negativePrompt: settings.negativePrompt,
        resolution: settings.resolution,
        profile: settings.profile,
      };
      return {
        ...captured,
        render: async (rect, size) => {
          renderSizes.push({ rect, size });
          renderedAt.push(revision.value);
          return Buffer.from('input');
        },
      };
    },
    buildMask: (_dirty, crop, _size, apply): MaskHandle => {
      masks.push({ crop, apply });
      return { png: Buffer.from('mask'), alpha: {}, empty: maskEmpty.value };
    },
    buildFullMask: (size): MaskHandle => {
      masks.push({ crop: { x: 0, y: 0, width: size, height: size }, apply: { x: 0, y: 0, width: size, height: size } });
      return { png: Buffer.from('full-mask'), alpha: {}, empty: false };
    },
    applyResult: async (_patch, crop, apply, _mask, forRevision) => {
      applied.push({ crop, apply, forRevision });
      return { rect: crop, url: '/patch.png' };
    },
    emit: (msg) => emitted.push(msg),
  };
  return { host, emitted, applied, masks, revision, prompt, maskEmpty, renderedAt, logs, settings, renderSizes };
}

const opts = {
  window: 1024,
  apply: 768,
  steps: 14,
  denoise: 0.55,
  debounceMs: 400,
  canvasSize: 4096,
  errorBackoffMs: 2000,
  watchdogMs: 10_000,
};
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

    // the edit made during the run still waits out its own debounce
    await vi.advanceTimersByTimeAsync(400);
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

  describe('canvas-edge regions (finding 4)', () => {
    it.each([
      ['top-left', R(20, 20, 30, 30)],
      ['bottom-right', R(4046, 4046, 30, 30)],
      ['left edge only', R(20, 2000, 30, 30)],
    ])('consumes a %s region instead of looping forever', async (_name, region) => {
      const { host, applied } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      s.markDirty([region]);
      await vi.advanceTimersByTimeAsync(400);
      await backend.finish();
      await vi.advanceTimersByTimeAsync(50);

      expect(applied).toHaveLength(1);
      const apply = applied[0]!.apply;
      expect(region.x).toBeGreaterThanOrEqual(apply.x);
      expect(region.x + region.width).toBeLessThanOrEqual(apply.x + apply.width);
      expect(s.dirtyRegions).toHaveLength(0);
      // and no runaway follow-up generation
      await vi.advanceTimersByTimeAsync(5000);
      expect(backend.calls).toHaveLength(1);
      s.stop();
    });

    it('uses one apply rect for the mask, the result and consumption', async () => {
      const { host, applied, masks, emitted } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      s.markDirty([R(20, 20, 30, 30)]);
      await vi.advanceTimersByTimeAsync(400);
      await backend.finish();

      const result = emitted.find((m) => m.t === 'ai_result') as { apply: Rect } | undefined;
      expect(masks[0]!.apply).toEqual(applied[0]!.apply);
      expect(result!.apply).toEqual(applied[0]!.apply);
      s.stop();
    });

    it('drops a region only after two identical runs made no progress', async () => {
      const { host, applied } = makeHost();
      const backend = new FakeBackend();
      const logs: string[] = [];
      // apply size 0 means nothing is ever consumed by subtraction
      const s = new AIScheduler(host, backend, { ...opts, apply: 0, log: (m) => logs.push(m) });
      s.markDirty([R(2000, 2000)]);
      await vi.advanceTimersByTimeAsync(400);
      await backend.finish();
      await vi.advanceTimersByTimeAsync(50);

      // first no-op run: the region is given one more chance
      expect(s.dirtyRegions).toHaveLength(1);
      expect(logs).toHaveLength(0);

      await backend.finish();
      await vi.advanceTimersByTimeAsync(50);
      expect(applied).toHaveLength(2);
      expect(s.dirtyRegions).toHaveLength(0);
      expect(logs.join(' ')).toMatch(/no-progress/);

      await vi.advanceTimersByTimeAsync(5000);
      expect(backend.calls).toHaveLength(2);
      s.stop();
    });

    it('never discards a second region that merely overlaps the crop (finding B2)', async () => {
      const { host, applied } = makeHost();
      const backend = new FakeBackend();
      const logs: string[] = [];
      // AI_WINDOW=2048 with AI_APPLY=128: the crop covers both regions but the
      // apply rect only covers the selected one.
      const s = new AIScheduler(host, backend, { ...opts, window: 2048, apply: 128, log: (m) => logs.push(m) });
      const far = { x: 50, y: 1000, width: 20, height: 20 };
      const near = { x: 1000, y: 1000, width: 20, height: 20 };
      s.markDirty([far]);
      s.markDirty([near]);
      expect(s.dirtyRegions).toHaveLength(2);

      await vi.advanceTimersByTimeAsync(400);
      await backend.finish();
      await vi.advanceTimersByTimeAsync(50);

      // the selected region was repainted; the other one survives untouched
      expect(applied).toHaveLength(1);
      expect(s.dirtyRegions).toEqual([far]);
      expect(logs).toHaveLength(0);

      // ...and it does get its own generation
      await backend.finish();
      await vi.advanceTimersByTimeAsync(50);
      expect(backend.calls).toHaveLength(2);
      expect(applied).toHaveLength(2);
      expect(applied[1]!.apply.x).toBeLessThanOrEqual(far.x);
      s.stop();
    });
  });

  describe('prompt changes (finding 12)', () => {
    it('re-runs the last applied area when nothing is dirty', async () => {
      const { host, applied, prompt } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      s.markDirty([R(2000, 2000)]);
      await vi.advanceTimersByTimeAsync(400);
      await backend.finish();
      expect(s.dirtyRegions).toHaveLength(0);

      prompt.value = 'watercolor town';
      s.nudge();
      await vi.advanceTimersByTimeAsync(400);
      expect(backend.calls).toHaveLength(2);
      expect(backend.calls[1]!.prompt).toBe('watercolor town');
      await backend.finish();
      expect(applied).toHaveLength(2);
      s.stop();
    });

    it('re-runs when the prompt changes during a generation (finding B5)', async () => {
      const { host, applied, prompt } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      prompt.value = 'fantasy town';
      s.markDirty([R(2000, 2000)]);
      await vi.advanceTimersByTimeAsync(400);
      expect(backend.calls).toHaveLength(1);

      // prompt edited while the first generation is still running
      prompt.value = 'watercolor town';
      s.nudge();
      await backend.finish();
      await vi.advanceTimersByTimeAsync(500);

      expect(backend.calls).toHaveLength(2);
      expect(backend.calls[1]!.prompt).toBe('watercolor town');
      await backend.finish();
      expect(applied).toHaveLength(2);
      expect(s.dirtyRegions).toHaveLength(0);
      s.stop();
    });

    it('does not loop when the prompt is unchanged during a generation', async () => {
      const { host } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      s.markDirty([R(2000, 2000)]);
      await vi.advanceTimersByTimeAsync(400);
      await backend.finish();
      await vi.advanceTimersByTimeAsync(3000);
      expect(backend.calls).toHaveLength(1);
      s.stop();
    });

    it('does nothing on a prompt change before anything has ever been generated', async () => {
      const { host } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      s.nudge();
      await vi.advanceTimersByTimeAsync(1000);
      expect(backend.calls).toHaveLength(0);
      s.stop();
    });
  });

  describe('atomic render snapshot (finding 10)', () => {
    it('tags the result with the revision captured before rendering', async () => {
      const { host, applied, revision, renderedAt } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      revision.value = 7;
      s.markDirty([R(2000, 2000)]);
      await vi.advanceTimersByTimeAsync(400);
      expect(renderedAt).toEqual([7]);

      revision.value = 9;
      await backend.finish();
      expect(applied[0]!.forRevision).toBe(7);
      s.stop();
    });

    it('uses the prompt captured with the job, not a later one', async () => {
      const { host, prompt } = makeHost();
      const backend = new FakeBackend();
      const s = new AIScheduler(host, backend, opts);
      prompt.value = 'first';
      s.markDirty([R(2000, 2000)]);
      await vi.advanceTimersByTimeAsync(400);
      prompt.value = 'second';
      expect(backend.calls[0]!.prompt).toBe('first');
      s.stop();
    });
  });

  describe('watchdog (finding 7)', () => {
    it('abandons a generation that never returns and recovers', async () => {
      const { host, emitted } = makeHost();
      const backend = new FakeBackend();
      backend.hangForever = true;
      const s = new AIScheduler(host, backend, opts);
      s.markDirty([R(2000, 2000)]);
      await vi.advanceTimersByTimeAsync(400);
      expect(backend.calls).toHaveLength(1);

      await vi.advanceTimersByTimeAsync(10_100);
      const error = emitted.find((m) => m.t === 'ai_status' && m.state === 'error') as { message?: string } | undefined;
      expect(error?.message).toBe('generation timed out');

      // not wedged: the next run still happens after the backoff
      backend.hangForever = false;
      await vi.advanceTimersByTimeAsync(2100);
      expect(backend.calls).toHaveLength(2);
      s.stop();
    });
  });
});

/** Feature: the room's AI settings reach the backend. */
describe('room AI settings', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('falls back to the server defaults when the room sets nothing', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls[0]).toMatchObject({ denoise: 0.55, negativePrompt: DEFAULT_NEGATIVE_PROMPT });
    s.stop();
  });

  it('passes the room denoise and negative prompt through', async () => {
    const { host, settings } = makeHost();
    settings.denoise = 0.85;
    settings.negativePrompt = 'no text, no watermark';
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls[0]).toMatchObject({ denoise: 0.85, negativePrompt: 'no text, no watermark' });
    s.stop();
  });

  it('treats a blank room negative prompt as "use the built-in list"', async () => {
    const { host, settings } = makeHost();
    settings.negativePrompt = '   ';
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls[0]?.negativePrompt).toBe(DEFAULT_NEGATIVE_PROMPT);
    s.stop();
  });
});

/** Full-canvas mode: one whole-canvas generation per quiet period. */
describe('AIScheduler full mode', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const fullOpts = { ...opts, mode: 'full' as const, canvasSize: 1024, window: 1024, apply: 1024 };

  it('debounces any change into a single whole-canvas run', async () => {
    const { host, masks, applied } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, fullOpts);
    s.markDirty([R(10, 10)]);
    await vi.advanceTimersByTimeAsync(200);
    s.markDirty([R(900, 900)]);
    await vi.advanceTimersByTimeAsync(399);
    expect(backend.calls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(2);
    expect(backend.calls).toHaveLength(1);
    // the mask and the applied rect are the whole canvas
    expect(masks[0]).toEqual({ crop: { x: 0, y: 0, width: 1024, height: 1024 }, apply: { x: 0, y: 0, width: 1024, height: 1024 } });
    await backend.finish();
    expect(applied[0]?.crop).toEqual({ x: 0, y: 0, width: 1024, height: 1024 });
    expect(applied[0]?.apply).toEqual({ x: 0, y: 0, width: 1024, height: 1024 });
    s.stop();
  });

  it('never keeps dirty regions or picks a crop', async () => {
    const { host } = makeHost();
    const s = new AIScheduler(host, new FakeBackend(), fullOpts);
    s.markDirty([R(10, 10), R(3000, 3000)]);
    expect(s.dirtyRegions).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(400);
    s.stop();
  });

  it('keeps one request in flight and re-runs once for changes made during it', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, fullOpts);
    s.markDirty([R(10, 10)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls).toHaveLength(1);

    s.markDirty([R(20, 20)]);
    s.markDirty([R(30, 30)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls).toHaveLength(1);

    await backend.finish();
    await vi.advanceTimersByTimeAsync(500);
    expect(backend.calls).toHaveLength(2);
    s.stop();
  });

  it('re-runs after a prompt or settings change with nothing drawn', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, fullOpts);
    s.nudge();
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls).toHaveLength(1);
    s.stop();
  });

  it('discards a result the canvas has moved past', async () => {
    const { host, revision, applied, emitted } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, fullOpts);
    s.markDirty([R(10, 10)]);
    await vi.advanceTimersByTimeAsync(400);

    // a newer generation was accepted while this one was running
    revision.value = 99;
    s.markDirty([R(40, 40)]);
    await vi.advanceTimersByTimeAsync(400);
    await backend.finish();
    await vi.advanceTimersByTimeAsync(10);
    expect(applied.length).toBe(1);

    await backend.finish();
    await vi.advanceTimersByTimeAsync(600);
    expect(applied.length).toBe(2);
    expect(applied[1]!.forRevision).toBe(99);
    expect(emitted.filter((m) => m.t === 'ai_result')).toHaveLength(2);
    s.stop();
  });

  it('retries after an error', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    backend.failNext = true;
    const s = new AIScheduler(host, backend, fullOpts);
    s.markDirty([R(10, 10)]);
    await vi.advanceTimersByTimeAsync(400);
    await vi.advanceTimersByTimeAsync(2100);
    expect(backend.calls.length).toBeGreaterThan(1);
    s.stop();
  });
});

/** Scheduling deadlines: nothing may shorten a debounce or a backoff. */
describe('scheduling deadlines', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('does not let an edit near the end of a run bypass the debounce', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls).toHaveLength(1);

    // an edit arrives, then the run finishes immediately afterwards
    s.markDirty([R(2100, 2100)]);
    await backend.finish();
    await vi.advanceTimersByTimeAsync(100);
    // still inside the debounce window: no second request yet
    expect(backend.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls).toHaveLength(2);
    s.stop();
  });

  it('does not let a new edit shorten the error backoff', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    backend.failNext = true;
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(200);
    s.markDirty([R(2000, 2000)]); // would previously re-run after 400 ms
    await vi.advanceTimersByTimeAsync(900);
    expect(backend.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1200);
    expect(backend.calls).toHaveLength(2);
    s.stop();
  });
});

/** Generation resolution is decoupled from the canvas size in full mode. */
describe('AIScheduler generation resolution', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const fullOpts = { ...opts, mode: 'full' as const, canvasSize: 1024, window: 768, apply: 1024 };

  it('renders and generates at AI_WINDOW while applying to the whole canvas', async () => {
    const { host, masks, applied, renderSizes } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, fullOpts);
    s.markDirty([R(10, 10)]);
    await vi.advanceTimersByTimeAsync(400);

    expect(backend.calls[0]?.size).toBe(768);
    expect(masks[0]?.crop).toEqual({ x: 0, y: 0, width: 768, height: 768 });
    expect(renderSizes[0]).toEqual({ rect: { x: 0, y: 0, width: 1024, height: 1024 }, size: 768 });

    await backend.finish();
    // the result is applied to the whole canvas, not to a 768 square
    expect(applied[0]?.crop).toEqual({ x: 0, y: 0, width: 1024, height: 1024 });
    s.stop();
  });

  it('prefers the room setting over the server default', async () => {
    const { host, renderSizes, settings } = makeHost();
    settings.resolution = 512;
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, fullOpts);
    s.markDirty([R(10, 10)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(backend.calls[0]?.size).toBe(512);
    expect(renderSizes[0]?.size).toBe(512);
    s.stop();
  });

  it('can generate above the canvas size too', async () => {
    const { host, renderSizes } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, { ...fullOpts, canvasSize: 512, window: 1024, apply: 512 });
    s.markDirty([R(10, 10)]);
    await vi.advanceTimersByTimeAsync(400);
    expect(renderSizes[0]).toEqual({ rect: { x: 0, y: 0, width: 512, height: 512 }, size: 1024 });
    await backend.finish();
    s.stop();
  });
});

/** The profile is chosen per room, so it travels with the job, not the config. */
describe('profile reaches the backend', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('passes the job profile and the matching step count', async () => {
    const { host, settings } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, { ...opts, steps: 14, fastSteps: 4 });

    settings.profile = 'fast';
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(500);
    await backend.finish();

    settings.profile = 'quality';
    s.markDirty([R(2100, 2100)]);
    await vi.advanceTimersByTimeAsync(500);
    await backend.finish();

    expect(backend.calls.map((c) => ({ profile: c.profile, steps: c.steps }))).toEqual([
      { profile: 'fast', steps: 4 },
      { profile: 'quality', steps: 14 },
    ]);
    s.stop();
  });

  it('defaults to quality when a job does not say', async () => {
    const { host } = makeHost();
    const backend = new FakeBackend();
    const s = new AIScheduler(host, backend, { ...opts, fastSteps: 4 });
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(500);
    expect(backend.calls[0]).toMatchObject({ profile: 'quality', steps: 14 });
    s.stop();
  });

  it('uses the plain step count for a fast job when fastSteps is not configured', async () => {
    const { host, settings } = makeHost();
    const backend = new FakeBackend();
    settings.profile = 'fast';
    const s = new AIScheduler(host, backend, opts);
    s.markDirty([R(2000, 2000)]);
    await vi.advanceTimersByTimeAsync(500);
    expect(backend.calls[0]).toMatchObject({ profile: 'fast', steps: 14 });
    s.stop();
  });
});
