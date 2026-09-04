import { randomUUID } from 'node:crypto';
import { QUALITY_SUFFIX } from '@brushjam/shared';
import { AbortedError, delay, type AIBackend, type GenerateRequest } from './types.js';

export interface ComfyOptions {
  url: string;
  checkpoint: string;
  cfg?: number;
  pollIntervalMs?: number;
  /** Overall deadline for one generation. */
  timeoutMs?: number;
  /** Deadline for a single HTTP call to ComfyUI. */
  requestTimeoutMs?: number;
}

export interface WorkflowInput {
  checkpoint: string;
  prompt: string;
  negativePrompt: string;
  imageName: string;
  maskName: string;
  seed: number;
  steps: number;
  cfg: number;
  denoise: number;
  filenamePrefix: string;
}

/**
 * ComfyUI API-format workflow. Node ids are fixed so tests can assert on them.
 * SetLatentNoiseMask (not VAEEncodeForInpaint) keeps the human drawing as the
 * img2img base, so the model reinterprets the strokes instead of filling holes.
 */
export function buildWorkflow(i: WorkflowInput): Record<string, unknown> {
  return {
    '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: i.checkpoint } },
    '2': { class_type: 'CLIPTextEncode', inputs: { text: i.prompt + QUALITY_SUFFIX, clip: ['1', 1] } },
    '3': { class_type: 'CLIPTextEncode', inputs: { text: i.negativePrompt, clip: ['1', 1] } },
    '4': { class_type: 'LoadImage', inputs: { image: i.imageName, upload: 'image' } },
    '5': { class_type: 'LoadImage', inputs: { image: i.maskName, upload: 'image' } },
    '6': { class_type: 'ImageToMask', inputs: { image: ['5', 0], channel: 'red' } },
    '7': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
    '8': { class_type: 'SetLatentNoiseMask', inputs: { samples: ['7', 0], mask: ['6', 0] } },
    '9': {
      class_type: 'KSampler',
      inputs: {
        model: ['1', 0],
        positive: ['2', 0],
        negative: ['3', 0],
        latent_image: ['8', 0],
        seed: i.seed,
        steps: i.steps,
        cfg: i.cfg,
        sampler_name: 'euler_ancestral',
        scheduler: 'normal',
        denoise: i.denoise,
      },
    },
    '10': { class_type: 'VAEDecode', inputs: { samples: ['9', 0], vae: ['1', 2] } },
    '11': { class_type: 'SaveImage', inputs: { images: ['10', 0], filename_prefix: i.filenamePrefix } },
  };
}

interface UploadResult { name: string; subfolder: string; type: string }
interface HistoryImage { filename: string; subfolder: string; type: string }

export class ComfyUIBackend implements AIBackend {
  readonly name = 'comfyui';
  private readonly clientId = randomUUID();

  constructor(private readonly opts: ComfyOptions) {}

  private get base(): string {
    return this.opts.url.replace(/\/+$/, '');
  }

  /**
   * Every call to ComfyUI carries both the caller's abort signal and its own
   * timeout. Without the timeout a stalled socket would leave the scheduler
   * `inFlight` forever and the room would never generate again.
   */
  private async fetch(path: string, init: RequestInit, signal: AbortSignal): Promise<Response> {
    const timeout = AbortSignal.timeout(this.opts.requestTimeoutMs ?? 30_000);
    try {
      return await fetch(`${this.base}${path}`, { ...init, signal: AbortSignal.any([signal, timeout]) });
    } catch (err) {
      if (signal.aborted) throw new AbortedError();
      if ((err as { name?: string })?.name === 'TimeoutError' || (err as { name?: string })?.name === 'AbortError') {
        throw new Error(`ComfyUI request timed out: ${path.split('?')[0]}`);
      }
      throw err;
    }
  }

  private async upload(bytes: Buffer, filename: string, signal: AbortSignal): Promise<UploadResult> {
    const form = new FormData();
    form.append('image', new Blob([new Uint8Array(bytes)], { type: 'image/png' }), filename);
    form.append('overwrite', 'true');
    form.append('type', 'input');
    const res = await this.fetch('/upload/image', { method: 'POST', body: form }, signal);
    if (!res.ok) throw new Error(`ComfyUI upload failed: ${res.status} ${await res.text()}`);
    return (await res.json()) as UploadResult;
  }

  async generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer> {
    const stamp = `${req.tag}_${Date.now()}`;
    const image = await this.upload(req.imagePng, `brushjam_${stamp}_img.png`, signal);
    const mask = await this.upload(req.maskPng, `brushjam_${stamp}_mask.png`, signal);

    const workflow = buildWorkflow({
      checkpoint: this.opts.checkpoint,
      prompt: req.prompt,
      negativePrompt: req.negativePrompt,
      imageName: image.subfolder ? `${image.subfolder}/${image.name}` : image.name,
      maskName: mask.subfolder ? `${mask.subfolder}/${mask.name}` : mask.name,
      seed: req.seed,
      steps: req.steps,
      cfg: this.opts.cfg ?? 5.5,
      denoise: req.denoise,
      filenamePrefix: `brushjam/${req.tag}`,
    });

    const queued = await this.fetch(
      '/prompt',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ prompt: workflow, client_id: this.clientId }),
      },
      signal,
    );
    if (!queued.ok) throw new Error(`ComfyUI /prompt failed: ${queued.status} ${await queued.text()}`);
    const { prompt_id: promptId } = (await queued.json()) as { prompt_id: string };
    if (typeof promptId !== 'string' || promptId.length === 0) throw new Error('ComfyUI returned no prompt_id');

    const out = await this.waitForOutput(promptId, signal);
    const query = new URLSearchParams({ filename: out.filename, subfolder: out.subfolder ?? '', type: out.type ?? 'output' });
    const view = await this.fetch(`/view?${query.toString()}`, {}, signal);
    if (!view.ok) throw new Error(`ComfyUI /view failed: ${view.status}`);
    return Buffer.from(await view.arrayBuffer());
  }

  /**
   * Only interrupt when the job ComfyUI is currently running is ours - a blind
   * POST /interrupt would kill another room's (or another app's) generation.
   */
  private async interrupt(promptId: string): Promise<never> {
    try {
      const res = await fetch(`${this.base}/queue`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const queue = (await res.json()) as { queue_running?: unknown[][] };
        const running = (queue.queue_running ?? []).some((entry) => entry.some((v) => v === promptId));
        if (running) await fetch(`${this.base}/interrupt`, { method: 'POST', signal: AbortSignal.timeout(5000) });
      }
    } catch {
      /* best effort: dropping the result is enough for correctness */
    }
    throw new AbortedError();
  }

  private async waitForOutput(promptId: string, signal: AbortSignal): Promise<HistoryImage> {
    const interval = this.opts.pollIntervalMs ?? 250;
    const deadline = Date.now() + (this.opts.timeoutMs ?? 180_000);
    for (;;) {
      if (signal.aborted) await this.interrupt(promptId);
      const res = await this.fetch(`/history/${encodeURIComponent(promptId)}`, {}, signal);
      if (res.ok) {
        const history = (await res.json()) as Record<string, { status?: { status_str?: string }; outputs?: Record<string, { images?: HistoryImage[] }> }>;
        const entry = history[promptId];
        if (entry) {
          if (entry.status?.status_str === 'error') throw new Error('ComfyUI reported an execution error');
          for (const node of Object.values(entry.outputs ?? {})) {
            const first = node.images?.[0];
            if (first) return first;
          }
        }
      }
      if (Date.now() > deadline) throw new Error('ComfyUI generation timed out');
      try {
        await delay(interval, signal);
      } catch (err) {
        if (err instanceof AbortedError) await this.interrupt(promptId);
        throw err;
      }
    }
  }
}

/** Cheap reachability probe used to pick the default backend at boot. */
export async function comfyReachable(url: string, timeoutMs = 1500): Promise<boolean> {
  try {
    const res = await fetch(`${url.replace(/\/+$/, '')}/system_stats`, { signal: AbortSignal.timeout(timeoutMs) });
    return res.ok;
  } catch {
    return false;
  }
}
