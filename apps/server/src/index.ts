import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createBackend, type Watcher } from './ai/backends/index.js';
import { envOrigin, loadConfig, resolveBackendConfig, type Config } from './config.js';
import { createBrushJamServer } from './server.js';

// Optional repo-root .env (RunPod credentials, AI_BACKEND, ...). Never logged.
// Node's loader fills gaps only: a variable already in the environment - from
// `pnpm dev:stream`, say - keeps its value, so .env is a default, not an
// override.
const envFile = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '.env');
const backendBefore = process.env.AI_BACKEND;
if (existsSync(envFile)) {
  try {
    process.loadEnvFile(envFile);
  } catch {
    /* malformed .env is not fatal */
  }
}
const backendOrigin = envOrigin(backendBefore, process.env.AI_BACKEND);

let config: Config;
try {
  config = loadConfig();
} catch (err) {
  console.error(`[brushjam] ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}

/**
 * The registry does not exist yet when the backend is built, so the late-worker
 * watcher talks to it through this: re-probe the limits and let every room that
 * gave up while the worker was down run what it owes.
 */
let watcher: Watcher | null = null;
let onStreamReady: (() => void) | null = null;

const backend = await createBackend(config, console.log, {
  onWatcher: (w) => {
    watcher = w;
  },
  onStreamReady: () => onStreamReady?.(),
});
// The chosen backend gets the last word on the defaults (`auto` only resolves
// here) and on what a room may ask for at all.
const capabilities = await backend.capabilities();
try {
  config = resolveBackendConfig(config, backend.name, capabilities);
} catch (err) {
  console.error(`[brushjam] ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}
const { server, registry, close } = createBrushJamServer(config, backend, {
  profiles: capabilities.profiles,
  maxDenoise: capabilities.maxDenoise,
  maxResolution: config.maxResolution,
  negativePromptActive: capabilities.negativePromptActive,
});
if (watcher) registry.watchBackend(watcher);
onStreamReady = () => void registry.refreshCapabilities();

/**
 * A watch restart can begin before the previous process has actually released
 * the port (Windows in particular), so listening is retried briefly instead of
 * dying with EADDRINUSE.
 */
const LISTEN_RETRIES = 12;
const LISTEN_RETRY_MS = 250;

/** The retry's error handler, so each attempt replaces the previous one. */
let listenErrorHandler: ((err: NodeJS.ErrnoException) => void) | null = null;

function listenWithRetry(attempt = 1): void {
  if (listenErrorHandler) server.off('error', listenErrorHandler);
  const onError = (err: NodeJS.ErrnoException): void => {
    if (err.code === 'EADDRINUSE' && attempt < LISTEN_RETRIES) {
      console.warn(`[brushjam] port ${config.port} still busy, retrying (${attempt}/${LISTEN_RETRIES})`);
      setTimeout(() => listenWithRetry(attempt + 1), LISTEN_RETRY_MS);
      return;
    }
    console.error(`[brushjam] ${err.message}`);
    process.exit(1);
  };
  listenErrorHandler = onError;
  server.once('error', onError);
  // The success callback is registered once, not per attempt: passing it to
  // listen() each retry stacked up 'listening' handlers and warned about a
  // leak after ten tries.
  server.listen(config.port, config.host);
}

function announce(): void {
  {
    if (listenErrorHandler) {
      server.off('error', listenErrorHandler);
      listenErrorHandler = null;
    }
    console.log(`[brushjam] server on http://${config.host}:${config.port} (set HOST=0.0.0.0 to expose on the LAN)`);
    // Named explicitly: a stale terminal or a second dev stack is by far the
    // most common reason a server is not the backend someone expected.
    console.log(
      backendOrigin === 'unset'
        ? `[brushjam] AI_BACKEND is not set (auto-detecting; put AI_BACKEND=stream in .env to pin it)`
        : `[brushjam] AI_BACKEND=${config.aiBackend} (from the ${backendOrigin === '.env' ? 'repo-root .env' : 'environment'})`,
    );
    console.log(`[brushjam] canvas ${config.canvasSize} / ai mode ${config.aiMode}`);
    if (config.aiMode === 'full') {
      const note = config.aiWindow === config.canvasSize ? 'same as canvas' : `resampled from/to ${config.canvasSize}`;
      console.log(`[brushjam] generation resolution ${config.aiWindow} (${note})`);
    }
    console.log(`[brushjam] ai window ${config.aiWindow} / apply ${config.aiApply} / steps ${config.aiSteps} / denoise ${config.aiDenoise}`);
    console.log(
      `[brushjam] default profile ${config.aiProfile} (fast: ${config.aiFastSteps}-step LCM ${config.comfyFastLora || 'unavailable'}, quality: ${config.aiSteps}-step euler_a)`,
    );
    console.log(
      `[brushjam] backend ${backend.name} supports ${capabilities.profiles.join('/')} up to ${capabilities.maxResolution} at denoise <= ${capabilities.maxDenoise}`,
    );
    if (capabilities.maxResolution < config.aiWindow) {
      console.warn(
        `[brushjam] AI_WINDOW ${config.aiWindow} is larger than the backend's ${capabilities.maxResolution}; requests may be refused`,
      );
    }
    if (config.fastDisabled) {
      console.warn('[brushjam] the fast profile is unavailable: COMFYUI_FAST_LORA is empty, so every room starts on quality');
    }
  }
}

server.once('listening', announce);

listenWithRetry();

let shuttingDown = false;
function shutdown(): void {
  if (shuttingDown) return;
  shuttingDown = true;
  // Hard deadline: a stuck close would keep the port bound and break the next
  // watch restart, which is far worse than an abrupt exit.
  const kill = setTimeout(() => process.exit(0), 2000);
  kill.unref();
  void close().then(() => process.exit(0));
}

for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK'] as const) {
  process.on(signal, shutdown);
}
// tsx watch (and other supervisors) may ask over IPC instead of by signal
process.on('message', (msg) => {
  if (msg === 'shutdown' || msg === 'SIGTERM' || msg === 'SIGINT') shutdown();
});
