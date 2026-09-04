import { randomUUID } from 'node:crypto';
import { QUALITY_SUFFIX } from '@brushjam/shared';
import { AbortedError, type AIBackend, type GenerateRequest } from './types.js';

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

  constructor(private readonly opts: StreamOptions) {}

  private get base(): string {
    return this.opts.url.replace(/\/+$/, '');
  }

  async generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer> {
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
      void this.cancel(requestId);
      if (signal.aborted) throw new AbortedError();
      const name = (err as { name?: string })?.name;
      if (name === 'TimeoutError' || name === 'AbortError') throw new Error('stream worker request timed out');
      throw err;
    }

    // 499 is the worker acknowledging our own cancellation.
    if (res.status === 499) throw new AbortedError();
    if (res.status === 409) {
      throw new Error(`stream worker is busy: ${await safeText(res)}`);
    }
    if (!res.ok) throw new Error(`stream worker /generate failed: ${res.status} ${await safeText(res)}`);

    let body: StreamResponse;
    try {
      body = (await res.json()) as StreamResponse;
    } catch (err) {
      if (signal.aborted) throw new AbortedError();
      throw new Error(`stream worker returned invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (typeof body.image_b64 !== 'string' || body.image_b64.length === 0) {
      throw new Error('stream worker returned no image');
    }
    return Buffer.from(stripDataUrl(body.image_b64), 'base64');
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

  /** Cheap reachability + warmth probe, mirroring `comfyReachable`. */
  async healthy(): Promise<boolean> {
    try {
      const res = await fetch(`${this.base}/healthz`, {
        signal: AbortSignal.timeout(this.opts.probeTimeoutMs ?? 1500),
      });
      if (!res.ok) return false;
      const body = (await res.json()) as { ok?: boolean };
      return body.ok === true;
    } catch {
      return false;
    }
  }
}

/** The worker adds QUALITY_SUFFIX itself; do not send it twice. */
function stripQualitySuffix(prompt: string): string {
  return prompt.endsWith(QUALITY_SUFFIX) ? prompt.slice(0, -QUALITY_SUFFIX.length) : prompt;
}

function stripDataUrl(value: string): string {
  const comma = value.startsWith('data:') ? value.indexOf(',') : -1;
  return comma >= 0 ? value.slice(comma + 1) : value;
}

async function safeText(res: Response): Promise<string> {
  try {
    return (await res.text()).slice(0, 300);
  } catch {
    return '';
  }
}

/** Reachability probe used when picking a backend at boot. */
export async function streamReachable(url: string, timeoutMs = 1500): Promise<boolean> {
  return new StreamBackend({ url, probeTimeoutMs: timeoutMs }).healthy();
}
