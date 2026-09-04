export interface GenerateRequest {
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

export interface AIBackend {
  readonly name: string;
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
