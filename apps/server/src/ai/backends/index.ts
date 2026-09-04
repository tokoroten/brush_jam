import type { Config } from '../../config.js';
import { ComfyUIBackend, comfyReachable } from './comfyui.js';
import { MockBackend } from './mock.js';
import { RunpodBackend } from './runpod.js';
import { StreamBackend, streamReachable } from './stream.js';
import type { AIBackend } from './types.js';

export * from './types.js';
export { buildWorkflow, ComfyUIBackend, comfyReachable } from './comfyui.js';
export { MockBackend } from './mock.js';
export { RunpodBackend } from './runpod.js';
export { StreamBackend, streamReachable } from './stream.js';

/**
 * Pick a backend. An explicit AI_BACKEND always wins; `auto` prefers the
 * resident stream worker (much the fastest when it is up), then ComfyUI, then
 * the mock. Both probes are bounded so a dead endpoint cannot delay startup.
 */
export const PROBE_TIMEOUT_MS = 2000;

export async function createBackend(config: Config, log: (m: string) => void = console.log): Promise<AIBackend> {
  const comfy = (): AIBackend =>
    new ComfyUIBackend({
      url: config.comfyUrl,
      checkpoint: config.comfyCheckpoint,
      cfg: config.aiCfg,
      vaeTile: config.aiVaeTile,
      fastLora: config.aiFast ? config.comfyFastLora : undefined,
    });
  // The worker may take ~90 s to answer its first request while it loads the
  // model, so it gets a generation-sized deadline, not a probe-sized one.
  const stream = (): AIBackend => new StreamBackend({ url: config.streamUrl, timeoutMs: config.streamTimeoutMs });

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
      vaeTile: config.aiVaeTile,
      fastLora: config.aiFast ? config.comfyFastLora : undefined,
    });
  }
  if (config.aiBackend === 'stream') {
    log(`[ai] backend: stream at ${config.streamUrl} (AI_BACKEND=stream)`);
    return stream();
  }
  if (config.aiBackend === 'comfyui') {
    log(`[ai] backend: comfyui at ${config.comfyUrl} (AI_BACKEND=comfyui)`);
    return comfy();
  }

  if (await streamReachable(config.streamUrl, PROBE_TIMEOUT_MS)) {
    log(`[ai] backend: stream at ${config.streamUrl} (auto-detected: /healthz answered ok)`);
    return stream();
  }
  if (await comfyReachable(config.comfyUrl, PROBE_TIMEOUT_MS)) {
    log(`[ai] backend: comfyui at ${config.comfyUrl} (auto-detected: no stream worker at ${config.streamUrl})`);
    return comfy();
  }
  log(`[ai] backend: mock - neither the stream worker (${config.streamUrl}) nor ComfyUI (${config.comfyUrl}) answered`);
  return new MockBackend();
}
