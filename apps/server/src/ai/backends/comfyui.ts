import { randomUUID } from 'node:crypto';
import { MAX_AI_RESOLUTION, MAX_DENOISE, QUALITY_SUFFIX } from '@brushjam/shared';
import { AbortedError, delay, type AIBackend, type BackendCapabilities, type GenerateRequest } from './types.js';

export interface ComfyOptions {
  url: string;
  checkpoint: string;
  cfg?: number;
  /**
   * Name of the 4-step LoRA. Requests with `profile: 'fast'` load it; empty or
   * undefined means the fast profile is unavailable and every request runs the
   * quality workflow.
   */
  fastLora?: string;
  /** VAE decode tile size; 0 falls back to a plain VAEDecode. */
  vaeTile?: number;
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
  /**
   * 4-step LCM mode: loads a LoRA between the checkpoint and its consumers and
   * switches the sampler to the low-step settings that actually work with it.
   * Empty/undefined keeps the ordinary many-step workflow.
   */
  fastLora?: string;
  /**
   * Tile size for VAEDecodeTiled. Plain VAEDecode of a 1024x1024 latent takes
   * 1-4 minutes on an 8 GB card once VRAM is contended (sampling itself is
   * ~22 s), so tiling is the default. 0 restores the plain node.
   */
  vaeTile?: number;
}

/**
 * ComfyUI API-format workflow. Node ids are fixed so tests can assert on them.
 * SetLatentNoiseMask (not VAEEncodeForInpaint) keeps the human drawing as the
 * img2img base, so the model reinterprets the strokes instead of filling holes.
 */
export function buildWorkflow(i: WorkflowInput): Record<string, unknown> {
  const profile = fastProfile(i.fastLora);
  const fast = profile !== null;
  // With the LoRA loaded, MODEL and CLIP come from node 12 instead of the
  // checkpoint. VAE still comes from the checkpoint: LoraLoader has no VAE out.
  const model: [string, number] = fast ? ['12', 0] : ['1', 0];
  const clip: [string, number] = fast ? ['12', 1] : ['1', 1];

  const workflow: Record<string, unknown> = {
    '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: i.checkpoint } },
    '2': { class_type: 'CLIPTextEncode', inputs: { text: i.prompt + QUALITY_SUFFIX, clip } },
    '3': { class_type: 'CLIPTextEncode', inputs: { text: i.negativePrompt, clip } },
    '4': { class_type: 'LoadImage', inputs: { image: i.imageName, upload: 'image' } },
    '5': { class_type: 'LoadImage', inputs: { image: i.maskName, upload: 'image' } },
    '6': { class_type: 'ImageToMask', inputs: { image: ['5', 0], channel: 'red' } },
    '7': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
    '8': { class_type: 'SetLatentNoiseMask', inputs: { samples: ['7', 0], mask: ['6', 0] } },
    '9': {
      class_type: 'KSampler',
      inputs: {
        model,
        positive: ['2', 0],
        negative: ['3', 0],
        latent_image: ['8', 0],
        seed: i.seed,
        // `steps` is the real sampler-step count at any denoise: ComfyUI builds
        // the longer schedule and then keeps the last steps+1 sigmas, so denoise
        // only picks the starting noise level (comfy/samplers.py calculate_sigmas).
        // Dividing by denoise here made "4-step" mode run 8 steps at 0.55.
        steps: i.steps,
        cfg: profile ? profile.cfg : i.cfg,
        sampler_name: profile ? profile.sampler : 'euler_ancestral',
        scheduler: profile ? profile.scheduler : 'normal',
        denoise: i.denoise,
      },
    },
    '10': decodeNode(i.vaeTile),
    '11': { class_type: 'SaveImage', inputs: { images: ['10', 0], filename_prefix: i.filenamePrefix } },
  };

  if (fast) {
    workflow['12'] = {
      class_type: 'LoraLoader',
      inputs: { model: ['1', 0], clip: ['1', 1], lora_name: i.fastLora, strength_model: 1, strength_clip: 1 },
    };
  }
  return workflow;
}

/** LCM wants cfg 1.0-2.0; the normal 5.5 destroys a 4-step result. */
export const FAST_CFG = 1.5;
export const DEFAULT_FAST_LORA = 'lcm-lora-sdxl.safetensors';

/**
 * Few-step LoRAs are not interchangeable: DMD2 is a distilled model that wants
 * cfg 1.0 (no guidance at all), while LCM wants a little. Forcing 1.5 on DMD2
 * was quietly wrong, so fast mode is a named profile keyed off the LoRA name.
 */
export interface FastProfile {
  name: string;
  cfg: number;
  sampler: string;
  scheduler: string;
}

const LCM_PROFILE: FastProfile = { name: 'lcm', cfg: FAST_CFG, sampler: 'lcm', scheduler: 'sgm_uniform' };
const DMD2_PROFILE: FastProfile = { name: 'dmd2', cfg: 1.0, sampler: 'lcm', scheduler: 'sgm_uniform' };

export function fastProfile(lora: string | undefined): FastProfile | null {
  if (!lora) return null;
  return /dmd2/i.test(lora) ? DMD2_PROFILE : LCM_PROFILE;
}

/** VAEDecodeTiled on ComfyUI 0.28 requires all four size inputs. */
function decodeNode(tile: number | undefined): Record<string, unknown> {
  const samples: [string, number] = ['9', 0];
  const vae: [string, number] = ['1', 2];
  if (!tile || tile <= 0) return { class_type: 'VAEDecode', inputs: { samples, vae } };
  return {
    class_type: 'VAEDecodeTiled',
    inputs: { samples, vae, tile_size: tile, overlap: 64, temporal_size: 64, temporal_overlap: 8 },
  };
}

interface UploadResult { name: string; subfolder: string; type: string }
interface HistoryImage { filename: string; subfolder: string; type: string }

export class ComfyUIBackend implements AIBackend {
  readonly name = 'comfyui';
  private readonly clientId = randomUUID();
  /** Undefined until the first request; then the LoRA the graph last used. */
  private lastLora: string | undefined | null = null;

  constructor(private readonly opts: ComfyOptions) {}

  private get base(): string {
    return this.opts.url.replace(/\/+$/, '');
  }

  async capabilities(): Promise<BackendCapabilities> {
    // Both profiles, unless there is no LoRA to build the fast graph from.
    return {
      profiles: this.opts.fastLora ? ['fast', 'quality'] : ['quality'],
      maxResolution: MAX_AI_RESOLUTION,
      maxDenoise: MAX_DENOISE,
    };
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

  /**
   * The LoRA for this request's profile, or undefined for the quality
   * workflow. ComfyUI keeps one model in VRAM, so alternating profiles makes
   * it load or unload the LoRA - a few seconds on the first request after a
   * switch. Worth saying out loud when it happens rather than looking like a
   * random slow generation.
   */
  private loraFor(req: GenerateRequest): string | undefined {
    const wanted = req.profile === 'fast' ? this.opts.fastLora || undefined : undefined;
    if (this.lastLora !== undefined && this.lastLora !== wanted) {
      const to = wanted ? `fast (${wanted})` : 'quality';
      console.log(`[comfyui] switching profile to ${to}; the first generation after a switch reloads the model`);
    }
    this.lastLora = wanted;
    return wanted;
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
      vaeTile: this.opts.vaeTile ?? 512,
      fastLora: this.loraFor(req),
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

    // From here on the job exists on the ComfyUI side: any failure must cancel
    // it, or the next retry would queue a duplicate expensive generation behind
    // an orphan that still runs (and blocks every other room on that instance).
    try {
      const out = await this.waitForOutput(promptId, signal);
      const query = new URLSearchParams({ filename: out.filename, subfolder: out.subfolder ?? '', type: out.type ?? 'output' });
      const view = await this.fetch(`/view?${query.toString()}`, {}, signal);
      if (!view.ok) throw new Error(`ComfyUI /view failed: ${view.status}`);
      return Buffer.from(await view.arrayBuffer());
    } catch (err) {
      await this.cancelPrompt(promptId);
      throw err;
    }
  }

  /**
   * Remove one prompt from ComfyUI: interrupt it if it is the running job,
   * delete it from the queue if it is only waiting. Never a blind POST
   * /interrupt - that would kill another room's (or another app's) generation.
   */
  private async cancelPrompt(promptId: string): Promise<void> {
    try {
      const res = await fetch(`${this.base}/queue`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) return;
      const queue = (await res.json()) as { queue_running?: unknown[][]; queue_pending?: unknown[][] };
      const mentions = (entries: unknown[][] | undefined): boolean =>
        (entries ?? []).some((entry) => Array.isArray(entry) && entry.some((v) => v === promptId));

      if (mentions(queue.queue_running)) {
        await fetch(`${this.base}/interrupt`, { method: 'POST', signal: AbortSignal.timeout(5000) });
      } else if (mentions(queue.queue_pending)) {
        await fetch(`${this.base}/queue`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ delete: [promptId] }),
          signal: AbortSignal.timeout(5000),
        });
      }
    } catch {
      /* best effort: the result is dropped either way */
    }
  }

  private async waitForOutput(promptId: string, signal: AbortSignal): Promise<HistoryImage> {
    const interval = this.opts.pollIntervalMs ?? 250;
    const deadline = Date.now() + (this.opts.timeoutMs ?? 180_000);
    let lastPollError: string | null = null;
    for (;;) {
      // Cleanup happens once, in generate()'s catch, for every failure path.
      if (signal.aborted) throw new AbortedError();
      // A single stalled or failed /history poll is transient: the job is still
      // running on the other side, so keep polling until the overall deadline.
      try {
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
      } catch (err) {
        if (err instanceof AbortedError) throw err;
        if (!isTransient(err)) throw err;
        lastPollError = err instanceof Error ? err.message : String(err);
      }
      if (Date.now() > deadline) {
        throw new Error(lastPollError ? `ComfyUI generation timed out (last poll: ${lastPollError})` : 'ComfyUI generation timed out');
      }
      await delay(interval, signal);
    }
  }
}

/** A timed-out or dropped poll is worth retrying; a protocol error is not. */
function isTransient(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return /timed out|fetch failed|ECONNRESET|socket hang up|network/i.test(message);
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
