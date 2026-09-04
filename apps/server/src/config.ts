export interface Config {
  host: string;
  port: number;
  aiBackend: 'comfyui' | 'mock' | 'runpod' | 'auto';
  comfyUrl: string;
  comfyCheckpoint: string;
  aiWindow: number;
  aiApply: number;
  aiSteps: number;
  aiDenoise: number;
  aiCfg: number;
  /** VAE decode tile size; 0 uses a plain (non-tiled) VAEDecode. */
  aiVaeTile: number;
  aiDebounceMs: number;
  aiWatchdogMs: number;
  roomIdleMs: number;
  runpodEndpointId: string;
  runpodApiKey: string;
  webDist: string | null;
}

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigError';
  }
}

interface NumberRule {
  min: number;
  max: number;
  integer?: boolean;
  multipleOf?: number;
}

/**
 * Environment values are validated once, at startup, and the process refuses to
 * boot on bad input. A fractional or negative AI window would otherwise produce
 * broken crops and a room that throws the same render error every two seconds.
 */
function num(env: NodeJS.ProcessEnv, key: string, fallback: number, rule: NumberRule, errors: string[]): number {
  const raw = env[key];
  if (raw === undefined || raw === '') return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    errors.push(`${key} must be a number (got ${JSON.stringify(raw)})`);
    return fallback;
  }
  if (rule.integer && !Number.isInteger(value)) errors.push(`${key} must be a whole number (got ${value})`);
  if (value < rule.min || value > rule.max) errors.push(`${key} must be between ${rule.min} and ${rule.max} (got ${value})`);
  if (rule.multipleOf && value % rule.multipleOf !== 0) errors.push(`${key} must be a multiple of ${rule.multipleOf} (got ${value})`);
  return value;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const errors: string[] = [];
  const backendRaw = (env.AI_BACKEND ?? 'auto').toLowerCase();
  if (!['auto', 'comfyui', 'mock', 'runpod'].includes(backendRaw)) {
    errors.push(`AI_BACKEND must be one of auto, comfyui, mock, runpod (got ${JSON.stringify(env.AI_BACKEND)})`);
  }

  const config: Config = {
    host: env.HOST ?? '127.0.0.1',
    port: num(env, 'PORT', 8787, { min: 1, max: 65535, integer: true }, errors),
    aiBackend: (['comfyui', 'mock', 'runpod'].includes(backendRaw) ? backendRaw : 'auto') as Config['aiBackend'],
    comfyUrl: (env.COMFYUI_URL ?? 'http://127.0.0.1:8188').replace(/\/+$/, ''),
    comfyCheckpoint: env.COMFYUI_CHECKPOINT ?? 'waiNSFWIllustrious_v150.safetensors',
    aiWindow: num(env, 'AI_WINDOW', 1024, { min: 256, max: 2048, integer: true, multipleOf: 64 }, errors),
    aiApply: num(env, 'AI_APPLY', 768, { min: 128, max: 2048, integer: true, multipleOf: 64 }, errors),
    aiSteps: num(env, 'AI_STEPS', 14, { min: 1, max: 150, integer: true }, errors),
    aiDenoise: num(env, 'AI_DENOISE', 0.55, { min: 0, max: 1 }, errors),
    aiCfg: num(env, 'AI_CFG', 5.5, { min: 0, max: 30 }, errors),
    aiVaeTile: num(env, 'AI_VAE_TILE', 512, { min: 0, max: 4096, integer: true }, errors),
    aiDebounceMs: num(env, 'AI_DEBOUNCE_MS', 400, { min: 0, max: 600_000, integer: true }, errors),
    aiWatchdogMs: num(env, 'AI_WATCHDOG_MS', 180_000, { min: 1000, max: 3_600_000, integer: true }, errors),
    roomIdleMs: num(env, 'ROOM_IDLE_MS', 30 * 60_000, { min: 10_000, max: 24 * 3_600_000, integer: true }, errors),
    runpodEndpointId: env.RUNPOD_ENDPOINT_ID ?? '',
    runpodApiKey: env.RUNPOD_API_KEY ?? '',
    webDist: env.WEB_DIST ?? null,
  };

  if (config.aiApply > config.aiWindow) {
    errors.push(`AI_APPLY (${config.aiApply}) must not exceed AI_WINDOW (${config.aiWindow})`);
  }
  try {
    new URL(config.comfyUrl);
  } catch {
    errors.push(`COMFYUI_URL is not a valid URL (got ${JSON.stringify(config.comfyUrl)})`);
  }
  if (config.aiBackend === 'runpod' && (!config.runpodEndpointId || !config.runpodApiKey)) {
    errors.push('AI_BACKEND=runpod requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY');
  }

  if (errors.length > 0) throw new ConfigError(`invalid configuration:\n  - ${errors.join('\n  - ')}`);
  return config;
}
