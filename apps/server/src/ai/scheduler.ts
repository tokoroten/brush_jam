import {
  CANVAS_SIZE,
  DEFAULT_NEGATIVE_PROMPT,
  applyRect,
  chooseCrop,
  mergeDirtyAll,
  shouldAcceptResult,
  subtractRect,
  type AIState,
  type Rect,
  type ServerMessage,
} from '@brushjam/shared';
import { AbortedError, type AIBackend } from './backends/index.js';

export interface MaskHandle {
  png: Buffer;
  /** Opaque handle passed straight back to `applyResult`. */
  alpha: unknown;
  empty: boolean;
}

export interface SchedulerHost {
  getPrompt(): string;
  getRevision(): number;
  renderInput(crop: Rect, size: number): Promise<Buffer>;
  buildMask(dirty: Rect[], crop: Rect, size: number, applySize: number): MaskHandle;
  /** Composite the patch and publish it; returns the URL clients should fetch. */
  applyResult(patch: Buffer, crop: Rect, mask: unknown, forRevision: number): Promise<{ rect: Rect; url: string }>;
  emit(msg: ServerMessage): void;
}

export interface SchedulerOptions {
  window: number;
  apply: number;
  steps: number;
  denoise: number;
  debounceMs: number;
  canvasSize?: number;
  errorBackoffMs?: number;
  tag?: string;
  seed?: () => number;
}

/**
 * Debounces drawing activity into at most one in-flight generation per room,
 * picks the crop, drives the backend, and applies only results that are still
 * useful. Human drawing never waits on any of this.
 */
export class AIScheduler {
  private dirty: Rect[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private inFlight = false;
  private pending = false;
  private lastAccepted = 0;
  private controller: AbortController | null = null;
  private stopped = false;
  private stateValue: AIState = 'idle';
  /** Regions that were already dirty when the in-flight request was built. */
  private snapshotDirty = new Set<Rect>();

  constructor(
    private readonly host: SchedulerHost,
    private readonly backend: AIBackend,
    private readonly opts: SchedulerOptions,
  ) {}

  get state(): AIState {
    return this.stateValue;
  }

  get dirtyRegions(): readonly Rect[] {
    return this.dirty;
  }

  markDirty(rects: readonly Rect[]): void {
    if (this.stopped || rects.length === 0) return;
    this.dirty = mergeDirtyAll(this.dirty, rects);
    this.setState('queued');
    this.schedule(this.opts.debounceMs);
  }

  /** Prompt changes re-run the last dirty area without adding new geometry. */
  nudge(): void {
    if (this.stopped || this.dirty.length === 0) return;
    this.setState('queued');
    this.schedule(this.opts.debounceMs);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.controller?.abort();
  }

  private setState(state: AIState, message?: string, latencyMs?: number): void {
    this.stateValue = state;
    const msg: ServerMessage = { t: 'ai_status', state, forRevision: this.host.getRevision() };
    if (message !== undefined) msg.message = message;
    if (latencyMs !== undefined) msg.latencyMs = latencyMs;
    this.host.emit(msg);
  }

  private schedule(ms: number): void {
    if (this.stopped) return;
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.tick();
    }, ms);
  }

  private async tick(): Promise<void> {
    if (this.stopped) return;
    if (this.inFlight) {
      this.pending = true;
      return;
    }
    await this.run();
  }

  private async run(): Promise<void> {
    if (this.dirty.length === 0) {
      this.setState('idle');
      return;
    }
    const region = this.dirty[this.dirty.length - 1]!;
    const crop = chooseCrop(region, this.opts.window, this.opts.canvasSize ?? CANVAS_SIZE);
    const mask = this.host.buildMask([...this.dirty], crop, this.opts.window, this.opts.apply);
    if (mask.empty) {
      this.dirty = this.dirty.filter((r) => r !== region);
      this.setState('idle');
      if (this.dirty.length > 0) this.schedule(0);
      return;
    }

    const forRevision = this.host.getRevision();
    this.inFlight = true;
    this.pending = false;
    this.controller = new AbortController();
    this.snapshotDirty = new Set(this.dirty);
    this.setState('generating');
    const startedAt = Date.now();

    try {
      const imagePng = await this.host.renderInput(crop, this.opts.window);
      const patch = await this.backend.generate(
        {
          prompt: this.host.getPrompt(),
          negativePrompt: DEFAULT_NEGATIVE_PROMPT,
          imagePng,
          maskPng: mask.png,
          size: this.opts.window,
          denoise: this.opts.denoise,
          steps: this.opts.steps,
          seed: (this.opts.seed ?? defaultSeed)(),
          tag: `${this.opts.tag ?? 'room'}_r${forRevision}`,
        },
        this.controller.signal,
      );

      if (!shouldAcceptResult(forRevision, this.lastAccepted)) {
        this.setState('idle');
      } else {
        const applied = await this.host.applyResult(patch, crop, mask.alpha, forRevision);
        this.lastAccepted = forRevision;
        const latencyMs = Date.now() - startedAt;
        this.host.emit({ t: 'ai_result', rect: applied.rect, url: applied.url, aiRevision: forRevision, crop, latencyMs });
        this.consumeDirty(crop);
        this.setState('idle', undefined, latencyMs);
      }
      this.afterRun(0);
    } catch (err) {
      if (err instanceof AbortedError || this.stopped) {
        this.inFlight = false;
        this.controller = null;
        return;
      }
      this.setState('error', err instanceof Error ? err.message : String(err));
      this.afterRun(this.opts.errorBackoffMs ?? 2000);
    }
  }

  private afterRun(delayMs: number): void {
    this.inFlight = false;
    this.controller = null;
    if (this.stopped) return;
    if (this.pending || this.dirty.length > 0) this.schedule(delayMs);
  }

  /**
   * Drop the parts of each dirty region the AI actually repainted. Regions that
   * appeared *during* the request are kept whole - the result predates them.
   */
  private consumeDirty(crop: Rect): void {
    const applied = applyRect(crop, this.opts.apply);
    const next: Rect[] = [];
    for (const r of this.dirty) {
      if (!this.snapshotDirty.has(r)) {
        next.push(r);
        continue;
      }
      for (const part of subtractRect(r, applied)) {
        if (part.width >= 1 && part.height >= 1) next.push(part);
      }
    }
    this.dirty = next;
    this.snapshotDirty.clear();
  }
}

const defaultSeed = (): number => Math.floor(Math.random() * 2 ** 31);
