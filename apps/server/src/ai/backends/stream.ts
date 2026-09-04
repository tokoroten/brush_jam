import { randomUUID } from 'node:crypto';
import { QUALITY_SUFFIX } from '@brushjam/shared';

/** Used when the worker does not report its own limits. */
export const DEFAULT_STREAM_MAX_RESOLUTION = 1024;
export const DEFAULT_STREAM_MAX_DENOISE = 0.9;
import { AbortedError, type AIBackend, type BackendCapabilities, type GenerateRequest, BackendHttpError } from './types.js';

export interface StreamOptions {
  /** Base URL of apps/stream-worker, e.g. http://127.0.0.1:8790 */
  url: string;
  /** Overall deadline for one generation. */
  timeoutMs?: number;
  /** Deadline for the cheap /healthz probe. */
  probeTimeoutMs?: number;
  /**
   * Wait for the GPU when the worker is busy instead of being refused with 409.
   * On by default: the scheduler already allows one in-flight request per room,
   * so a 409 here would only mean "another room is generating", which is worth
   * waiting for rather than failing.
   */
  queue?: boolean;
  /** How long to wait for a cancelled job to actually stop. */
  settleTimeoutMs?: number;
}

/** Sampling parameters the worker reports, recorded by the experiment scripts. */
export interface StreamSampling {
  steps?: number;
  guidance?: number;
  vae?: string;
  model?: string;
  lora?: string;
}

export interface StreamHealth {
  ok: boolean;
  /** The model is loaded and has run at least once. */
  warm: boolean;
  /** Largest square the worker will accept, 0 when it does not say. */
  maxSize: number;
  /** True while a generation is running. */
  busy: boolean;
  backend: string;
  /** Id of the job currently running, when the worker reports one. */
  currentRequestId?: string;
  /** Largest denoise the worker accepts; absent means it did not say. */
  maxDenoise?: number;
  /** Whether its guidance setting makes the negative prompt do anything. */
  negativePromptActive?: boolean;
  /** Whatever it reports about how it samples: steps, guidance, vae, model. */
  sampling: StreamSampling;
  /** Why the worker is not usable, for the log line. */
  reason?: string;
}

interface StreamResponse {
  image_b64?: string;
  width?: number;
  height?: number;
  timings?: Record<string, number>;
}

/**
 * HTTP client for the model-resident Python worker in `apps/stream-worker`.
 *
 * Unlike ComfyUI there is no queue to poll and nothing to clean up: the worker
 * holds the model in VRAM and answers one request in a single round trip, so a
 * dropped connection leaves nothing behind. The only failure this has to handle
 * carefully is a stalled socket, which would otherwise pin the scheduler's
 * single in-flight slot forever.
 *
 * The worker appends its own quality suffix, so it is stripped here to avoid
 * sending it twice.
 */
export class StreamBackend implements AIBackend {
  readonly name = 'stream';

  /**
   * Resolves once a cancelled generation has really stopped. Cancelling only
   * asks the worker to stop at its next diffusion step, so a retry issued
   * immediately would queue behind the job we just abandoned and pay for it
   * twice. Every generate waits for this first.
   */
  private settling: Promise<void> | null = null;

  constructor(private readonly opts: StreamOptions) {}

  private get base(): string {
    return this.opts.url.replace(/\/+$/, '');
  }

  async generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer> {
    if (this.settling) await this.settling;
    if (signal.aborted) throw new AbortedError();
    const timeout = AbortSignal.timeout(this.opts.timeoutMs ?? 120_000);
    // Dropping the HTTP request does NOT stop the GPU: the worker is already
    // inside a diffusion loop on a background thread and would run to
    // completion, holding the GPU while the next request queues behind it. The
    // id lets us tell it to stop at its next step.
    const requestId = randomUUID();
    let res: Response;
    try {
      res = await fetch(`${this.base}/generate`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          image_b64: req.imagePng.toString('base64'),
          mask_b64: req.maskPng.toString('base64'),
          prompt: stripQualitySuffix(req.prompt),
          negative_prompt: req.negativePrompt,
          denoise: req.denoise,
          steps: req.steps,
          seed: req.seed,
          width: req.size,
          height: req.size,
          request_id: requestId,
          queue: this.opts.queue ?? true,
        }),
        signal: AbortSignal.any([signal, timeout]),
      });
    } catch (err) {
      // Abandoned or timed out: free the GPU rather than leaving it working on
      // a result nobody will read.
      this.settling = this.cancelAndSettle(requestId);
      if (signal.aborted) throw new AbortedError();
      const name = (err as { name?: string })?.name;
      if (name === 'TimeoutError' || name === 'AbortError') throw new Error('stream worker request timed out');
      throw err;
    }

    // 499 is the worker acknowledging our own cancellation.
    if (res.status === 499) throw new AbortedError();
    if (res.status === 409) {
      throw new BackendHttpError(`stream worker is busy: ${await safeText(res)}`, res.status);
    }
    if (!res.ok) throw new BackendHttpError(`stream worker /generate failed: ${res.status} ${await safeText(res)}`, res.status);

    let body: StreamResponse;
    try {
      body = (await res.json()) as StreamResponse;
    } catch (err) {
      if (signal.aborted) throw new AbortedError();
      throw new Error(`stream worker returned invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
    return decodeImage(body, req.size);
  }

  /**
   * Cancel, then wait for `busy` to clear. Bounded: if the worker never reports
   * itself idle we give up and let the next request queue, which is bad but not
   * worse than blocking the room forever.
   */
  private async cancelAndSettle(requestId: string): Promise<void> {
    try {
      await this.cancel(requestId);
      const deadline = Date.now() + (this.opts.settleTimeoutMs ?? 15_000);
      while (Date.now() < deadline) {
        const health = await this.health();
        // Not reachable, idle, or already working on someone else's job.
        if (!health.ok || !health.busy) return;
        if (health.currentRequestId && health.currentRequestId !== requestId) return;
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    } finally {
      this.settling = null;
    }
  }

  /**
   * Best effort "stop working on this". Fire-and-forget by design: the caller
   * has already given up, and a failed cancel must not turn into a second
   * error on a path that is already unwinding.
   */
  private async cancel(requestId: string): Promise<void> {
    try {
      await fetch(`${this.base}/cancel`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ request_id: requestId }),
        signal: AbortSignal.timeout(5000),
      });
    } catch {
      /* the worker will finish the job; nothing better to do from here */
    }
  }

  /**
   * Structured probe. Reachability alone is not enough to auto-select a worker:
   * an unloaded or dry-run worker answers `ok` and then echoes the input, and a
   * worker capped below the configured window answers `ok` and then 400s every
   * request - both of which reach the scheduler as an endless retry loop.
   */
  async health(): Promise<StreamHealth> {
    const dead = (reason: string): StreamHealth => ({
      ok: false,
      warm: false,
      maxSize: 0,
      busy: false,
      backend: '',
      sampling: {},
      reason,
    });
    try {
      const res = await fetch(`${this.base}/healthz`, {
        signal: AbortSignal.timeout(this.opts.probeTimeoutMs ?? 1500),
      });
      if (!res.ok) return dead(`/healthz answered ${res.status}`);
      const body = (await res.json()) as {
        ok?: boolean;
        warm?: boolean;
        max_size?: number;
        max_denoise?: number;
        negative_prompt_active?: boolean;
        busy?: boolean;
        backend?: string;
        current_request_id?: string;
        steps?: number;
        guidance?: number;
        vae?: string;
        model?: string;
        lora?: string;
      };
      if (body.ok !== true) return dead('/healthz reported not ok');
      const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
      const str = (v: unknown): string | undefined => (typeof v === 'string' && v !== '' ? v : undefined);
      return {
        ok: true,
        warm: body.warm === true,
        maxSize: num(body.max_size) ?? 0,
        maxDenoise: num(body.max_denoise),
        negativePromptActive: typeof body.negative_prompt_active === 'boolean' ? body.negative_prompt_active : undefined,
        busy: body.busy === true,
        backend: str(body.backend) ?? 'stream',
        currentRequestId: str(body.current_request_id),
        sampling: {
          steps: num(body.steps),
          guidance: num(body.guidance),
          vae: str(body.vae),
          model: str(body.model),
          lora: str(body.lora),
        },
      };
    } catch (err) {
      return dead(err instanceof Error ? err.message : String(err));
    }
  }

  /**
   * One fused LCM LoRA, so there is no quality profile to offer: asking for 14
   * steps would silently run 4. Limits come from the worker where it reports
   * them, and from conservative defaults where it does not.
   */
  async capabilities(): Promise<BackendCapabilities> {
    const health = await this.health();
    // The worker knows whether its own guidance setting evaluates the negative
    // branch; assume it does when it does not say.
    const negative = health.negativePromptActive ?? true;
    return {
      profiles: ['fast'],
      maxResolution: health.maxSize > 0 ? health.maxSize : DEFAULT_STREAM_MAX_RESOLUTION,
      // Above ~0.9 an LCM worker tends to ignore the drawing entirely.
      maxDenoise: health.maxDenoise ?? DEFAULT_STREAM_MAX_DENOISE,
      negativePromptActive: { fast: negative, quality: negative },
    };
  }

  /** Cheap reachability probe, mirroring `comfyReachable`. */
  async healthy(): Promise<boolean> {
    return (await this.health()).ok;
  }
}

/**
 * The worker's own output, validated. `Buffer.from(x, 'base64')` silently
 * accepts nearly anything, so a truncated or wrong-sized image would otherwise
 * surface much later as a decode failure or as a stretched composite.
 */
function decodeImage(body: StreamResponse, size: number): Buffer {
  if (typeof body.image_b64 !== 'string' || body.image_b64.length === 0) {
    throw new Error('stream worker returned no image');
  }
  const payload = stripDataUrl(body.image_b64).trim();
  if (payload.length === 0 || !BASE64.test(payload)) {
    throw new Error('stream worker returned a malformed base64 image');
  }
  const bytes = Buffer.from(payload, 'base64');
  if (bytes.length === 0) throw new Error('stream worker returned an empty image');
  const dims = pngSize(bytes);
  if (!dims) throw new Error('stream worker returned something that is not a PNG');
  if (dims.width !== size || dims.height !== size) {
    throw new Error(`stream worker returned ${dims.width}x${dims.height}, expected ${size}x${size}`);
  }
  return bytes;
}

const BASE64 = /^[A-Za-z0-9+/\s]+={0,2}$/;
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** Width/height straight out of the IHDR chunk; no decode needed. */
export function pngSize(bytes: Buffer): { width: number; height: number } | null {
  if (bytes.length < 24 || !bytes.subarray(0, 8).equals(PNG_MAGIC)) return null;
  if (bytes.subarray(12, 16).toString('latin1') !== 'IHDR') return null;
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

/** The worker adds QUALITY_SUFFIX itself; do not send it twice. */
function stripQualitySuffix(prompt: string): string {
  return prompt.endsWith(QUALITY_SUFFIX) ? prompt.slice(0, -QUALITY_SUFFIX.length) : prompt;
}

function stripDataUrl(value: string): string {
  const comma = value.startsWith('data:') ? value.indexOf(',') : -1;
  return comma >= 0 ? value.slice(comma + 1) : value;
}

/** FastAPI errors are a JSON `detail` field; show that, not the raw envelope. */
async function safeText(res: Response): Promise<string> {
  try {
    const text = (await res.text()).slice(0, 1000);
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      if (typeof parsed.detail === 'string') return parsed.detail.slice(0, 300);
      if (parsed.detail !== undefined) return JSON.stringify(parsed.detail).slice(0, 300);
    } catch {
      /* not JSON: fall through to the raw text */
    }
    return text.slice(0, 300);
  } catch {
    return '';
  }
}

/** Reachability probe used when picking a backend at boot. */
export async function streamReachable(url: string, timeoutMs = 1500): Promise<boolean> {
  return new StreamBackend({ url, probeTimeoutMs: timeoutMs }).healthy();
}

/** Full capability probe used by auto-selection and the startup warning. */
export async function streamHealth(url: string, timeoutMs = 1500): Promise<StreamHealth> {
  return new StreamBackend({ url, probeTimeoutMs: timeoutMs }).health();
}
