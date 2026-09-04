import type { AIProfileName } from '@brushjam/shared';

export interface GenerateRequest {
  /**
   * Which workflow to run. Chosen per room and therefore per request, so one
   * backend instance serves both: the ComfyUI backend swaps the LoRA in and
   * out of the graph, mock and stream ignore it.
   */
  profile: AIProfileName;
  prompt: string;
  negativePrompt: string;
  /** size x size PNG, opaque. */
  imagePng: Buffer;
  /** size x size PNG, white = regenerate. */
  maskPng: Buffer;
  size: number;
  denoise: number;
  steps: number;
  seed: number;
  /** Free-form tag used for upload filenames / logging. */
  tag: string;
}

/**
 * What a backend can actually do. Rooms are clamped to this and the UI hides
 * what is unavailable, so a user cannot pick a profile the running backend does
 * not have (the stream worker holds one fused LCM LoRA: it has no quality mode
 * at all, and asking for 14 steps would silently give 4).
 */
export interface BackendCapabilities {
  profiles: AIProfileName[];
  /** Largest square the backend will generate. */
  maxResolution: number;
  maxDenoise: number;
  /**
   * Whether the negative prompt does anything, per profile. A distilled
   * few-step model running at CFG 1.0 never evaluates the negative branch, so
   * the box is inert and the UI says so rather than pretending.
   */
  negativePromptActive: Record<AIProfileName, boolean>;
}

export interface AIBackend {
  readonly name: string;
  /** Probed once at startup; may hit the network. */
  capabilities(): Promise<BackendCapabilities>;
  /** Resolves to a size x size PNG. */
  generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer>;
}

export class AbortedError extends Error {
  constructor() {
    super('generation aborted');
    this.name = 'AbortedError';
  }
}

export function delay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new AbortedError());
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(new AbortedError());
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}
