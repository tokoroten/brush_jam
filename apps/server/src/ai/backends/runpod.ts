import { buildWorkflow } from './comfyui.js';
import { AbortedError, type AIBackend, type GenerateRequest } from './types.js';

export interface RunpodOptions {
  endpointId: string;
  apiKey: string;
  checkpoint: string;
  cfg?: number;
  baseUrl?: string;
}

/**
 * Same workflow JSON as the local backend, posted to RunPod's `worker-comfyui`
 * serverless image. Transport-only difference, which is the whole point of
 * building the workflow in TS. NOT verified against a live endpoint - nothing is
 * deployed to RunPod by this repo.
 */
export class RunpodBackend implements AIBackend {
  readonly name = 'runpod';

  constructor(private readonly opts: RunpodOptions) {}

  async generate(req: GenerateRequest, signal: AbortSignal): Promise<Buffer> {
    if (signal.aborted) throw new AbortedError();
    const imageName = `brushjam_${req.tag}_img.png`;
    const maskName = `brushjam_${req.tag}_mask.png`;
    const workflow = buildWorkflow({
      checkpoint: this.opts.checkpoint,
      prompt: req.prompt,
      negativePrompt: req.negativePrompt,
      imageName,
      maskName,
      seed: req.seed,
      steps: req.steps,
      cfg: this.opts.cfg ?? 5.5,
      denoise: req.denoise,
      filenamePrefix: `brushjam/${req.tag}`,
    });

    const base = (this.opts.baseUrl ?? 'https://api.runpod.ai/v2').replace(/\/+$/, '');
    const res = await fetch(`${base}/${this.opts.endpointId}/runsync`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${this.opts.apiKey}` },
      body: JSON.stringify({
        input: {
          workflow,
          images: [
            { name: imageName, image: req.imagePng.toString('base64') },
            { name: maskName, image: req.maskPng.toString('base64') },
          ],
        },
      }),
      signal,
    });
    if (!res.ok) throw new Error(`RunPod runsync failed: ${res.status}`);
    const body = (await res.json()) as { output?: { images?: Array<{ data?: string; image?: string }> }; error?: string };
    if (body.error) throw new Error(`RunPod error: ${body.error}`);
    const first = body.output?.images?.[0];
    const b64 = first?.data ?? first?.image;
    if (!b64) throw new Error('RunPod returned no image');
    return Buffer.from(b64.replace(/^data:image\/\w+;base64,/, ''), 'base64');
  }
}
