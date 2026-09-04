import type { Config } from '../../config.js';
import { ComfyUIBackend, comfyReachable } from './comfyui.js';
import { MockBackend } from './mock.js';
import { RunpodBackend } from './runpod.js';
import type { AIBackend } from './types.js';

export * from './types.js';
export { buildWorkflow, ComfyUIBackend, comfyReachable } from './comfyui.js';
export { MockBackend } from './mock.js';
export { RunpodBackend } from './runpod.js';

/** Pick a backend from config; `auto` uses ComfyUI when it answers at boot. */
export async function createBackend(config: Config, log: (m: string) => void = console.log): Promise<AIBackend> {
  const comfy = (): AIBackend =>
    new ComfyUIBackend({ url: config.comfyUrl, checkpoint: config.comfyCheckpoint, cfg: config.aiCfg });

  if (config.aiBackend === 'mock') {
    log('[ai] backend: mock (AI_BACKEND=mock)');
    return new MockBackend();
  }
  if (config.aiBackend === 'runpod') {
    log('[ai] backend: runpod');
    return new RunpodBackend({
      endpointId: config.runpodEndpointId,
      apiKey: config.runpodApiKey,
      checkpoint: config.comfyCheckpoint,
      cfg: config.aiCfg,
    });
  }
  if (config.aiBackend === 'comfyui') {
    log(`[ai] backend: comfyui at ${config.comfyUrl}`);
    return comfy();
  }
  if (await comfyReachable(config.comfyUrl)) {
    log(`[ai] backend: comfyui at ${config.comfyUrl} (auto-detected)`);
    return comfy();
  }
  log(`[ai] backend: mock - ComfyUI was not reachable at ${config.comfyUrl}`);
  return new MockBackend();
}
