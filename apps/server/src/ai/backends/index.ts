import type { Config } from '../../config.js';
import { ComfyUIBackend, comfyReachable } from './comfyui.js';
import { MockBackend } from './mock.js';
import { RunpodBackend } from './runpod.js';
import { StreamBackend, streamHealth } from './stream.js';
import type { AIBackend } from './types.js';

export * from './types.js';
export { buildWorkflow, ComfyUIBackend, comfyReachable, fastProfile, DEFAULT_FAST_LORA, FAST_CFG } from './comfyui.js';
export { MockBackend } from './mock.js';
export { RunpodBackend } from './runpod.js';
export { StreamBackend, streamHealth, streamReachable, pngSize } from './stream.js';

/**
 * Pick a backend. An explicit AI_BACKEND always wins; `auto` prefers the
 * resident stream worker (much the fastest when it is up), then ComfyUI, then
 * the mock. Both probes are bounded so a dead endpoint cannot delay startup.
 */
export const PROBE_TIMEOUT_MS = 2000;

/** How often to look again for a stream worker that was not up at startup. */
export const STREAM_RETRY_MS = 10_000;

/** Stoppable so a test does not leave a timer behind. */
export interface Watcher {
  stop(): void;
}

/**
 * Poll until the worker answers, then say so once and stop. The backend object
 * needs no repair - it builds its URL per request - so appearing late is enough;
 * RoomRegistry picks the real limits up on its own capability poll.
 */
export function watchForStreamWorker(
  url: string,
  opts: { log?: (m: string) => void; intervalMs?: number; onReady?: () => void } = {},
): Watcher {
  const log = opts.log ?? console.log;
  const timer = setInterval(() => {
    void streamHealth(url, PROBE_TIMEOUT_MS).then((health) => {
      if (!health.ok) return;
      stop();
      log(
        health.warm
          ? `[ai] stream worker is up at ${url}; generations will use it from now on`
          : `[ai] stream worker is up at ${url} but still loading its model; the first request will wait`,
      );
      // Anyone who drew while it was down is owed a generation.
      opts.onReady?.();
    });
  }, opts.intervalMs ?? STREAM_RETRY_MS);
  // A background probe must never be the reason the process stays alive.
  (timer as { unref?: () => void }).unref?.();
  const stop = (): void => clearInterval(timer);
  return { stop };
}

/** Lets the caller own the late-worker watcher and react when it fires. */
export interface BackendHooks {
  /** Receives the watcher so it can be stopped with the server. */
  onWatcher?: (watcher: Watcher) => void;
  /** The worker that was missing at startup is now answering. */
  onStreamReady?: () => void;
}

export async function createBackend(
  config: Config,
  log: (m: string) => void = console.log,
  hooks: BackendHooks = {},
): Promise<AIBackend> {
  const comfy = (): AIBackend =>
    new ComfyUIBackend({
      url: config.comfyUrl,
      checkpoint: config.comfyCheckpoint,
      cfg: config.aiCfg,
      vaeTile: config.aiVaeTile,
      fastLora: config.comfyFastLora || undefined,
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
      fastLora: config.comfyFastLora || undefined,
      timeoutMs: config.runpodTimeoutMs,
    });
  }
  if (config.aiBackend === 'stream') {
    log(`[ai] backend: stream at ${config.streamUrl} (AI_BACKEND=stream)`);
    // An explicit choice is honoured either way, but say so now rather than
    // letting every generation fail with a puzzling 400.
    const health = await streamHealth(config.streamUrl, PROBE_TIMEOUT_MS);
    if (!health.ok) {
      log(`[ai] warning: stream worker is not answering (${health.reason ?? 'no answer'})`);
      log(`[ai] start it with: cd apps/stream-worker && uv run stream-worker   (~40 s to warm up)`);
      // Nothing else to do here: the room will work the moment it appears, so
      // keep looking rather than making someone restart the server.
      const watcher = watchForStreamWorker(config.streamUrl, {
        log,
        onReady: hooks.onStreamReady,
      });
      hooks.onWatcher?.(watcher);
    } else if (!health.warm) log('[ai] warning: stream worker is reachable but not warm; the first request will load the model');
    else if (health.maxSize > 0 && health.maxSize < config.aiWindow) {
      log(`[ai] warning: stream worker max_size ${health.maxSize} is below AI_WINDOW ${config.aiWindow}; requests will be refused`);
    }
    return stream();
  }
  if (config.aiBackend === 'comfyui') {
    log(`[ai] backend: comfyui at ${config.comfyUrl} (AI_BACKEND=comfyui)`);
    return comfy();
  }

  // The stream worker is NOT auto-selected by default (docs/STREAM_WORKER.md
  // section 6): it answering /healthz means it is holding ~5 GB of VRAM, which
  // on an 8 GB card starves ComfyUI, and it needs a different denoise to look
  // right. Which model owns the GPU is a deployment decision, not something a
  // reachability probe should infer. AI_STREAM_AUTO=1 opts back in.
  if (config.streamAuto) {
    const health = await streamHealth(config.streamUrl, PROBE_TIMEOUT_MS);
    if (!health.ok) {
      log(`[ai] skipping stream worker: ${health.reason ?? 'no answer'}`);
    } else if (!health.warm) {
      // An unloaded or dry-run worker answers ok and then echoes the input
      // back, which the scheduler cannot tell from a real result.
      log(`[ai] skipping stream worker at ${config.streamUrl}: reachable but not warm (no model loaded)`);
    } else if (health.maxSize > 0 && health.maxSize < config.aiWindow) {
      // It would 400 every request; full mode would retry that forever.
      log(`[ai] skipping stream worker at ${config.streamUrl}: max_size ${health.maxSize} < AI_WINDOW ${config.aiWindow}`);
    } else {
      log(`[ai] backend: stream at ${config.streamUrl} (auto-detected: warm, max_size ${health.maxSize || 'unreported'})`);
      return stream();
    }
  }
  if (await comfyReachable(config.comfyUrl, PROBE_TIMEOUT_MS)) {
    const why = config.streamAuto ? 'no usable stream worker' : 'stream is explicit-only';
    log(`[ai] backend: comfyui at ${config.comfyUrl} (auto-detected: ${why})`);
    return comfy();
  }
  // Name the setting that led here: "why is it mock?" is almost always an
  // AI_BACKEND that never reached this process, and the log should say so.
  log(`[ai] backend: mock - ComfyUI (${config.comfyUrl}) did not answer (AI_BACKEND=${config.aiBackend})`);
  return new MockBackend();
}
