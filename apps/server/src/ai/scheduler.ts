import {
  CANVAS_SIZE,
  DEFAULT_NEGATIVE_PROMPT,
  applyRectFor,
  chooseCrop,
  mergeDirtyAll,
  shouldAcceptResult,
  subtractRect,
  type AIProfileName,
  type AIState,
  type Rect,
  type ServerMessage,
} from '@brushjam/shared';
import { AbortedError, BackendHttpError, type AIBackend } from './backends/index.js';

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
  /** Room-level overrides; absent means "use the server defaults". */
  denoise?: number;
  negativePrompt?: string;
  /** Full mode: generation size, independent of the canvas size. */
  resolution?: number;
  /** Which workflow to run; picks the step count too. */
  profile?: AIProfileName;
  render(crop: Rect, size: number): Promise<Buffer>;
}

/**
 * A refusal, not a failure: the backend understood the request and said no, so
 * repeating it unchanged cannot work. Everything else - a refused connection, a
 * timeout, a 5xx, a worker still loading - is transient by definition and must
 * stay retryable, or a room that was drawn in before the worker was up would
 * owe its work forever.
 */
export function isPermanentError(err: unknown): boolean {
  // A real status beats reading tea leaves in the message: "size 512 out of
  // range" contains something that looks like a 5xx, and "400" can appear in a
  // pixel count. Backends attach the status they actually got.
  if (err instanceof BackendHttpError) return err.status >= 400 && err.status < 500;
  const message = err instanceof Error ? err.message : String(err);
  if (/timed out|timeout|abort|econnrefused|econnreset|enotfound|socket hang up|fetch failed|not answering|no answer|not warm/i.test(message)) {
    return false;
  }
  // Only a status in a place a status is written, not any three digits.
  if (/(?:^|\s)(?:status|failed:?|code)\s*5\d\d\b/i.test(message)) return false;
  if (/(?:^|\s)(?:status|failed:?|code)\s*4\d\d\b/i.test(message)) return true;
  return /out of range|too large|too small|max_size|not supported|unsupported|malformed|invalid/i.test(message);
}

const rectKey = (r: Rect): string => `${r.x},${r.y},${r.width},${r.height}`;

export interface SchedulerHost {
  /** MUST capture room state synchronously - no awaits before the copy. */
  beginJob(): RenderJob;
  getRevision(): number;
  buildMask(dirty: Rect[], crop: Rect, size: number, apply: Rect): MaskHandle;
  /** Fully opaque mask: full-canvas mode regenerates everything. */
  buildFullMask(size: number): MaskHandle;
  /** Called on every failed generation, with the run of identical failures. */
  onError?(message: string, repeated: number): void;
  applyResult(
    patch: Buffer,
    crop: Rect,
    apply: Rect,
    mask: unknown,
    forRevision: number,
  ): Promise<{ rect: Rect; url: string; aiGeneration: number }>;
  emit(msg: ServerMessage): void;
}

export interface SchedulerOptions {
  /**
   * 'full' regenerates the whole canvas on every change (the playtest default,
   * small canvases only); 'patch' keeps the dirty-region + crop pipeline.
   */
  mode?: 'full' | 'patch';
  window: number;
  apply: number;
  /** Steps for the quality profile. */
  steps: number;
  /**
   * Identical consecutive failures after which the scheduler stops retrying by
   * itself. A 400 "size out of range" never becomes true by waiting, and
   * retrying it every two seconds hides the message behind an endless
   * generating/error flicker.
   */
  maxRepeatedErrors?: number;
  /** Steps for the fast profile; falls back to `steps` when unset. */
  fastSteps?: number;
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
  /** Full-canvas mode: anything at all changed since the last generation. */
  private changed = false;
  /** Absolute debounce deadline; scheduling can move it later, never earlier. */
  private notBefore = 0;
  /** Absolute end of an error backoff. */
  private backoffUntil = 0;
  private timerAt = 0;
  /** The error that made the scheduler stop retrying, if any. */
  private stuckOn: string | null = null;
  private lastError: string | null = null;
  private repeatedError = 0;

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

  private get full(): boolean {
    return this.opts.mode === 'full';
  }

  markDirty(rects: readonly Rect[]): void {
    if (this.stopped || rects.length === 0) return;
    // A new edit is a different request; give it a chance.
    this.clearStuck();
    if (this.full) {
      // No regions, no crop selection: the whole canvas is the unit of work.
      this.changed = true;
      this.setState('queued');
      this.schedule(this.opts.debounceMs);
      return;
    }
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
    this.clearStuck();
    // Bumped even while a run is in flight: that run captured the old prompt and
    // must be re-done, otherwise the new prompt is silently dropped.
    this.promptEpoch += 1;
    if (this.full) {
      this.changed = true;
      this.setState('queued');
      this.schedule(this.opts.debounceMs);
      return;
    }
    if (this.dirty.length === 0) {
      if (!this.lastAppliedRect) return;
      this.dirty = [this.lastAppliedRect];
    }
    this.setState('queued');
    this.schedule(this.opts.debounceMs);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer !== null) clearTimeout(this.timer);
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

  /**
   * Scheduling never *shortens* a wait: the debounce deadline and the error
   * backoff are both absolute, so an edit that lands just as a run finishes
   * still waits out the quiet period, and a new edit cannot cut a backoff short.
   */
  private schedule(ms: number): void {
    if (this.stopped) return;
    const now = Date.now();
    const target = Math.max(now + ms, this.notBefore, this.backoffUntil);
    if (ms > 0) this.notBefore = Math.max(this.notBefore, now + ms);
    // Always re-aimed at the deadline: a later edit pushes the debounce out,
    // and neither can pull it in before notBefore/backoffUntil.
    if (this.timer !== null) clearTimeout(this.timer);
    this.timerAt = target;
    this.timer = setTimeout(
      () => {
        this.timer = null;
        void this.tick();
      },
      Math.max(0, target - now),
    );
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
    if (this.full) return this.runFull();
    if (this.dirty.length === 0) {
      this.setState('idle');
      return;
    }
    const region = this.dirty[this.dirty.length - 1]!;
    // Captured before the crop so the whole request - crop, mask, render and
    // size - uses one window (beginJob is synchronous and side-effect free).
    // The window has to come from the job, not from the constructor: when a
    // worker restarts smaller the room is clamped, and a scheduler still asking
    // for the old size would 400 on every retry forever.
    const job = this.host.beginJob();
    const window = job.resolution ?? this.opts.window;
    const crop = chooseCrop(region, window, this.opts.canvasSize ?? CANVAS_SIZE);
    // Centred on the region, not on the crop: a region against a canvas edge must
    // still fall inside the repainted area or it would never be consumed.
    const apply = applyRectFor(crop, Math.min(this.opts.apply, window), region);
    const mask = this.host.buildMask([...this.dirty], crop, window, apply);
    if (mask.empty) {
      this.dirty = this.dirty.filter((r) => r !== region);
      this.setState('idle');
      if (this.dirty.length > 0) this.schedule(0);
      return;
    }

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
      const imagePng = await job.render(crop, window);
      const patch = await this.backend.generate(
        {
          prompt: job.prompt,
          // An empty room setting means "keep the built-in list".
          negativePrompt: job.negativePrompt?.trim() ? job.negativePrompt : DEFAULT_NEGATIVE_PROMPT,
          imagePng,
          maskPng: mask.png,
          size: window,
          denoise: job.denoise ?? this.opts.denoise,
          steps: this.stepsFor(job.profile),
          profile: job.profile ?? 'quality',
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
        this.host.emit({
          t: 'ai_result',
          rect: applied.rect,
          url: applied.url,
          aiRevision: forRevision,
          aiGeneration: applied.aiGeneration,
          crop,
          apply,
          latencyMs,
          // The profile this run used, which may no longer be the room's.
          profile: job.profile ?? 'quality',
        });
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
      this.noteError(message, timedOut ? undefined : err);
      if (this.stuckOn !== null) {
        // A request the backend refuses outright will be refused again.
        this.setState('error', `${message} - not retrying until something changes`);
        this.inFlight = false;
        this.controller = null;
        return;
      }
      this.setState('error', message);
      this.backoffUntil = Date.now() + (this.opts.errorBackoffMs ?? 2000);
      this.afterRun(this.opts.errorBackoffMs ?? 2000);
    } finally {
      clearTimeout(watchdog);
    }
  }

  /**
   * Full-canvas mode: render everything, regenerate everything, replace
   * everything. One request in flight, latest revision wins, and any change
   * that arrives mid-flight simply queues another whole-canvas run.
   */
  /** The fast profile is only fast because it runs fewer steps. */
  private stepsFor(profile: AIProfileName | undefined): number {
    return profile === 'fast' ? this.opts.fastSteps ?? this.opts.steps : this.opts.steps;
  }

  private async runFull(): Promise<void> {
    if (!this.changed) {
      this.setState('idle');
      return;
    }
    const size = this.opts.canvasSize ?? CANVAS_SIZE;
    const rect: Rect = { x: 0, y: 0, width: size, height: size };
    const job = this.host.beginJob();
    const forRevision = job.revision;
    // The whole canvas is rendered, then resampled to the generation size; the
    // result is scaled back to the canvas when it is composited.
    const resolution = job.resolution ?? this.opts.window;
    const mask = this.host.buildFullMask(resolution);

    this.changed = false;
    this.inFlight = true;
    this.pending = false;
    this.controller = new AbortController();
    this.setState('generating');
    const startedAt = Date.now();
    let timedOut = false;
    const watchdog = setTimeout(() => {
      timedOut = true;
      this.controller?.abort();
    }, this.opts.watchdogMs ?? 180_000);

    try {
      const imagePng = await job.render(rect, resolution);
      const patch = await this.backend.generate(
        {
          prompt: job.prompt,
          negativePrompt: job.negativePrompt?.trim() ? job.negativePrompt : DEFAULT_NEGATIVE_PROMPT,
          imagePng,
          maskPng: mask.png,
          size: resolution,
          denoise: job.denoise ?? this.opts.denoise,
          steps: this.stepsFor(job.profile),
          profile: job.profile ?? 'quality',
          seed: (this.opts.seed ?? defaultSeed)(),
          tag: `${this.opts.tag ?? 'room'}_r${forRevision}`,
        },
        this.controller.signal,
      );

      if (!shouldAcceptResult(forRevision, this.lastAccepted)) {
        this.setState('idle');
      } else {
        const applied = await this.host.applyResult(patch, rect, rect, mask.alpha, forRevision);
        this.lastAccepted = forRevision;
        this.lastAppliedRect = rect;
        const latencyMs = Date.now() - startedAt;
        this.host.emit({
          t: 'ai_result',
          rect: applied.rect,
          url: applied.url,
          aiRevision: forRevision,
          aiGeneration: applied.aiGeneration,
          crop: rect,
          apply: rect,
          latencyMs,
          profile: job.profile ?? 'quality',
        });
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
      // the work was not done, so it is still owed
      this.changed = true;
      const message = timedOut ? 'generation timed out' : err instanceof Error ? err.message : String(err);
      this.noteError(message, timedOut ? undefined : err);
      if (this.stuckOn !== null) {
        this.setState('error', `${message} - not retrying until something changes`);
        this.inFlight = false;
        this.controller = null;
        clearTimeout(watchdog);
        return;
      }
      this.setState('error', message);
      this.backoffUntil = Date.now() + (this.opts.errorBackoffMs ?? 2000);
      this.afterRun(this.opts.errorBackoffMs ?? 2000);
    } finally {
      clearTimeout(watchdog);
    }
  }

  /**
   * Repeated identical errors mean the request itself is wrong, not that the
   * backend is busy. `host.onError` lets the owner re-probe the backend, which
   * is how a worker that came back with different limits gets noticed.
   */
  private noteError(message: string, err?: unknown): void {
    this.repeatedError = this.lastError === message ? this.repeatedError + 1 : 1;
    this.lastError = message;
    // Only a request the backend REFUSED is worth giving up on. A worker that
    // is down answers with the same connection error every time, and treating
    // that as permanent left work owed until someone drew again.
    if (this.repeatedError >= (this.opts.maxRepeatedErrors ?? 2) && isPermanentError(err ?? message)) {
      this.stuckOn = message;
    }
    this.host.onError?.(message, this.repeatedError);
  }

  /**
   * The backend became usable again (a worker finished loading, or came back).
   * Forget the error state and run if anything is still owed.
   */
  retryNow(): void {
    this.clearStuck();
    // Both floors have to go: the failed run pushed the debounce out too, and
    // the point of this call is that the wait is over.
    this.backoffUntil = 0;
    this.notBefore = 0;
    if (this.full ? this.changed || this.pending : this.dirty.length > 0) this.schedule(0);
  }

  /** Any real change (an edit, a settings change, new limits) unsticks it. */
  private clearStuck(): void {
    this.stuckOn = null;
    this.repeatedError = 0;
    this.lastError = null;
  }

  private afterRun(delayMs: number): void {
    this.inFlight = false;
    this.controller = null;
    if (this.stopped) return;
    if (this.full) {
      if (this.pending || this.changed) this.schedule(delayMs);
      return;
    }
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
