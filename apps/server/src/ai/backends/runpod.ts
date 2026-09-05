import { MAX_AI_RESOLUTION, MAX_DENOISE } from '@brushjam/shared';
import { buildWorkflow, fastProfile, negativePromptActive } from './comfyui.js';
import { AbortedError, BackendHttpError, delay, type AIBackend, type GenerateRequest, type BackendCapabilities } from './types.js';

export interface RunpodOptions {
  endpointId: string;
  apiKey: string;
  checkpoint: string;
  cfg?: number;
  vaeTile?: number;
  fastLora?: string;
  baseUrl?: string;
  /**
   * Overall deadline for one generation, counted from the first byte sent.
   * A cold worker on this endpoint spends ~25 s waiting for a machine and then
   * ~30 s loading a 7 GB checkpoint off the network volume before it samples
   * anything, so anything under ~180 s would abandon perfectly healthy jobs.
   */
  timeoutMs?: number;
  /** Gap between /status polls once the job has gone asynchronous. */
  pollIntervalMs?: number;
}

/** What `/runsync` and `/status/{id}` both answer with. */
interface RunpodJob {
  id?: string;
  status?: string;
  output?: { images?: RunpodImage[]; errors?: string[] };
  error?: string;
  delayTime?: number;
  executionTime?: number;
}

interface RunpodImage {
  filename?: string;
  /** `base64` by default; `s3_url` when the worker is configured to upload. */
  type?: string;
  data?: string;
}

/** Terminal job states. Anything else means "still going". */
const DONE = new Set(['COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT']);

/**
 * The same workflow JSON as the local ComfyUI backend, posted to RunPod's
 * `worker-comfyui` serverless image - transport-only difference, which is the
 * whole point of building the workflow in TS.
 *
 * Verified against a live endpoint (`runpod/worker-comfyui:5.10.0-base` with the
 * checkpoint and LoRA on a network volume) - see docs/RUNPOD.md.
 *
 * Two things about the contract are worth stating, because both were guesses
 * before and only one was right:
 *
 * - Output images arrive as `output.images[].data`, base64, no data-URI prefix.
 * - `/runsync` is not synchronous for long jobs. RunPod gives it about 90 s and
 *   then answers `{id, status: "IN_PROGRESS"}`, leaving the caller to poll
 *   `/status/{id}`. A cold start here is 80-120 s, so that path is the normal
 *   one for the first generation, not an edge case.
 */
export class RunpodBackend implements AIBackend {
  readonly name = 'runpod';

  constructor(private readonly opts: RunpodOptions) {}

  async capabilities(): Promise<BackendCapabilities> {
    // Same workflow builder as ComfyUI, so the same two profiles.
    const cfg = this.opts.cfg ?? 5.5;
    return {
      profiles: this.opts.fastLora ? ['fast', 'quality'] : ['quality'],
      maxResolution: MAX_AI_RESOLUTION,
      maxDenoise: MAX_DENOISE,
      negativePromptActive: {
        fast: negativePromptActive(fastProfile(this.opts.fastLora), cfg),
        quality: negativePromptActive(null, cfg),
      },
    };
  }

  private get base(): string {
    return (this.opts.baseUrl ?? 'https://api.runpod.ai/v2').replace(/\/+$/, '');
  }

  private get url(): string {
    return `${this.base}/${this.opts.endpointId}`;
  }

  private get headers(): Record<string, string> {
    return { 'content-type': 'application/json', authorization: `Bearer ${this.opts.apiKey}` };
  }

  async generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer> {
    if (signal.aborted) throw new AbortedError();
    const stamp = `${req.tag}_${Date.now()}`;
    const imageName = `brushjam_${stamp}_img.png`;
    const maskName = `brushjam_${stamp}_mask.png`;
    const workflow = buildWorkflow({
      checkpoint: this.opts.checkpoint,
      prompt: req.prompt,
      negativePrompt: req.negativePrompt,
      imageName,
      maskName,
      seed: req.seed,
      steps: req.steps,
      cfg: this.opts.cfg ?? 5.5,
      vaeTile: this.opts.vaeTile ?? 512,
      fastLora: req.profile === 'fast' ? this.opts.fastLora || undefined : undefined,
      denoise: req.denoise,
      filenamePrefix: `brushjam/${req.tag}`,
    });

    const deadline = Date.now() + (this.opts.timeoutMs ?? 300_000);
    const res = await this.post(
      '/runsync',
      {
        input: {
          workflow,
          images: [
            { name: imageName, image: req.imagePng.toString('base64') },
            { name: maskName, image: req.maskPng.toString('base64') },
          ],
        },
      },
      signal,
      deadline,
    );
    if (!res.ok) {
      throw new BackendHttpError(`RunPod /runsync failed: ${res.status} ${(await res.text()).slice(0, 300)}`, res.status);
    }
    let job = (await res.json()) as RunpodJob;

    // /runsync only waits ~90 s. Past that the job is still running on RunPod's
    // side and has to be polled - and abandoning it without a /cancel would
    // leave the single worker busy while the scheduler retries behind it.
    if (!DONE.has(job.status ?? '')) {
      const id = job.id;
      if (!id) throw new Error(`RunPod returned status ${job.status ?? 'unknown'} with no job id`);
      try {
        job = await this.poll(id, signal, deadline);
      } catch (err) {
        await this.cancel(id);
        throw err;
      }
    }
    return decodeImage(job);
  }

  /** One POST, bounded by both the caller's signal and the overall deadline. */
  private async post(path: string, body: unknown, signal: AbortSignal, deadline: number): Promise<Response> {
    return this.call(path, { method: 'POST', headers: this.headers, body: JSON.stringify(body) }, signal, deadline);
  }

  private async call(path: string, init: RequestInit, signal: AbortSignal, deadline: number): Promise<Response> {
    const left = Math.max(1000, deadline - Date.now());
    const timeout = AbortSignal.timeout(left);
    try {
      return await fetch(`${this.url}${path}`, { ...init, signal: AbortSignal.any([signal, timeout]) });
    } catch (err) {
      if (signal.aborted) throw new AbortedError();
      const name = (err as { name?: string })?.name;
      if (name === 'TimeoutError' || name === 'AbortError') throw new Error(`RunPod request timed out: ${path}`);
      throw err;
    }
  }

  private async poll(id: string, signal: AbortSignal, deadline: number): Promise<RunpodJob> {
    const interval = this.opts.pollIntervalMs ?? 1000;
    let lastPollError: string | null = null;
    for (;;) {
      if (signal.aborted) throw new AbortedError();
      try {
        const res = await this.call(`/status/${encodeURIComponent(id)}`, { headers: this.headers }, signal, deadline);
        if (res.ok) {
          const job = (await res.json()) as RunpodJob;
          if (DONE.has(job.status ?? '')) return job;
          lastPollError = null;
        } else if (res.status >= 500 || res.status === 429) {
          // The job outlives a blip on the control plane; keep polling.
          lastPollError = `status ${res.status}`;
        } else {
          throw new BackendHttpError(`RunPod /status failed: ${res.status}`, res.status);
        }
      } catch (err) {
        if (err instanceof AbortedError || err instanceof BackendHttpError) throw err;
        lastPollError = err instanceof Error ? err.message : String(err);
      }
      if (Date.now() > deadline) {
        throw new Error(lastPollError ? `RunPod generation timed out (last poll: ${lastPollError})` : 'RunPod generation timed out');
      }
      await delay(interval, signal);
    }
  }

  /** Best effort: a job left running would block the endpoint's only worker. */
  private async cancel(id: string): Promise<void> {
    try {
      await fetch(`${this.url}/cancel/${encodeURIComponent(id)}`, {
        method: 'POST',
        headers: this.headers,
        signal: AbortSignal.timeout(5000),
      });
    } catch {
      /* the result is dropped either way */
    }
  }
}

/**
 * Pull the PNG out of a finished job. Kept separate so the smoke script and the
 * tests can reuse the exact error messages the server would print.
 */
export function decodeImage(job: RunpodJob): Buffer {
  if (job.status && job.status !== 'COMPLETED') {
    const detail = job.error ?? job.output?.errors?.join('; ') ?? '';
    throw new Error(`RunPod job ${job.status}${detail ? `: ${detail}` : ''}`);
  }
  if (job.error) throw new Error(`RunPod error: ${job.error}`);
  const first = job.output?.images?.[0];
  if (!first) {
    const detail = job.output?.errors?.join('; ');
    throw new Error(detail ? `RunPod returned no image: ${detail}` : 'RunPod returned no image');
  }
  if (first.type === 's3_url') {
    throw new Error('RunPod worker is configured for S3 upload; this backend expects base64 images');
  }
  if (!first.data) throw new Error('RunPod returned an image entry with no data');
  return Buffer.from(first.data.replace(/^data:image\/\w+;base64,/, ''), 'base64');
}
