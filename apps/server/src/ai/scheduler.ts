import {
  CANVAS_SIZE,
  DEFAULT_NEGATIVE_PROMPT,
  applyRectFor,
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

/** An immutable render job captured synchronously from the room. */
export interface RenderJob {
  revision: number;
  prompt: string;
  render(crop: Rect, size: number): Promise<Buffer>;
}

const rectKey = (r: Rect): string => `${r.x},${r.y},${r.width},${r.height}`;

export interface SchedulerHost {
  /** MUST capture room state synchronously - no awaits before the copy. */
  beginJob(): RenderJob;
  getRevision(): number;
  buildMask(dirty: Rect[], crop: Rect, size: number, apply: Rect): MaskHandle;
  applyResult(patch: Buffer, crop: Rect, apply: Rect, mask: unknown, forRevision: number): Promise<{ rect: Rect; url: string }>;
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
  /** Hard cap on a single generation before it is abandoned. */
  watchdogMs?: number;
  tag?: string;
  seed?: () => number;
  log?: (message: string) => void;
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
  /** The last area the AI actually repainted, re-used when the prompt changes. */
  private lastAppliedRect: Rect | null = null;
  /** Bumped on every prompt change so an in-flight run can notice it is stale. */
  private promptEpoch = 0;
  /** crop+region signature of the last run that repainted none of its region. */
  private lastNoProgress: string | null = null;

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

  /**
   * Prompt changes re-run the AI. If nothing is dirty (the usual case - the last
   * generation consumed its region), the last applied area is re-dirtied so the
   * new prompt still produces a visible result.
   */
  nudge(): void {
    if (this.stopped) return;
    // Bumped even while a run is in flight: that run captured the old prompt and
    // must be re-done, otherwise the new prompt is silently dropped.
    this.promptEpoch += 1;
    if (this.dirty.length === 0) {
      if (!this.lastAppliedRect) return;
      this.dirty = [this.lastAppliedRect];
    }
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
    // Centred on the region, not on the crop: a region against a canvas edge must
    // still fall inside the repainted area or it would never be consumed.
    const apply = applyRectFor(crop, this.opts.apply, region);
    const mask = this.host.buildMask([...this.dirty], crop, this.opts.window, apply);
    if (mask.empty) {
      this.dirty = this.dirty.filter((r) => r !== region);
      this.setState('idle');
      if (this.dirty.length > 0) this.schedule(0);
      return;
    }

    const job = this.host.beginJob();
    const forRevision = job.revision;
    const promptEpoch = this.promptEpoch;
    this.inFlight = true;
    this.pending = false;
    this.controller = new AbortController();
    this.snapshotDirty = new Set(this.dirty);
    this.setState('generating');
    const startedAt = Date.now();
    let timedOut = false;
    const watchdog = setTimeout(() => {
      timedOut = true;
      this.controller?.abort();
    }, this.opts.watchdogMs ?? 180_000);

    try {
      const imagePng = await job.render(crop, this.opts.window);
      const patch = await this.backend.generate(
        {
          prompt: job.prompt,
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
        const applied = await this.host.applyResult(patch, crop, apply, mask.alpha, forRevision);
        this.lastAccepted = forRevision;
        this.lastAppliedRect = apply;
        const latencyMs = Date.now() - startedAt;
        this.host.emit({ t: 'ai_result', rect: applied.rect, url: applied.url, aiRevision: forRevision, crop, apply, latencyMs });
        this.consumeDirty(apply, crop, region);
        if (promptEpoch !== this.promptEpoch) {
          // The prompt changed while this was generating: redo the same area so
          // the visible result matches what people actually typed.
          this.dirty = mergeDirtyAll(this.dirty, [apply]);
        }
        this.setState('idle', undefined, latencyMs);
      }
      this.afterRun(0);
    } catch (err) {
      if (this.stopped) {
        this.inFlight = false;
        this.controller = null;
        return;
      }
      if (err instanceof AbortedError && !timedOut) {
        this.inFlight = false;
        this.controller = null;
        return;
      }
      const message = timedOut ? 'generation timed out' : err instanceof Error ? err.message : String(err);
      this.setState('error', message);
      this.afterRun(this.opts.errorBackoffMs ?? 2000);
    } finally {
      clearTimeout(watchdog);
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
   *
   * Only the region this run was selected for can be judged "stuck", and only
   * when the identical crop+region pair already failed to make progress once
   * before. Other regions that merely overlap the crop are still owed their own
   * generation: with a small AI_APPLY inside a large AI_WINDOW that is the
   * normal case, and discarding them would silently lose people's drawing.
   */
  private consumeDirty(apply: Rect, crop: Rect, selected: Rect): void {
    const signature = `${rectKey(crop)}|${rectKey(selected)}`;
    const repeated = this.lastNoProgress === signature;
    const next: Rect[] = [];
    let selectedStuck = false;

    for (const r of this.dirty) {
      if (!this.snapshotDirty.has(r)) {
        next.push(r);
        continue;
      }
      const parts = subtractRect(r, apply).filter((p) => p.width >= 1 && p.height >= 1);
      if (r === selected && area(parts) >= area([r])) {
        selectedStuck = true;
        if (repeated) continue; // second identical no-op run: give up on it
      }
      next.push(...parts);
    }

    this.dirty = next;
    this.snapshotDirty.clear();
    this.lastNoProgress = selectedStuck && !repeated ? signature : null;
    if (selectedStuck && repeated) {
      (this.opts.log ?? console.warn)(
        `[ai] no-progress guard: dropped dirty region ${rectKey(selected)} after two identical runs in crop ${rectKey(crop)}`,
      );
    }
  }
}

const area = (rects: readonly Rect[]): number => rects.reduce((sum, r) => sum + r.width * r.height, 0);
const defaultSeed = (): number => Math.floor(Math.random() * 2 ** 31);
