import { AI_PROFILES, PROFILE_DEFAULTS, type AIProfileName } from '@brushjam/shared';

export interface Config {
  host: string;
  port: number;
  aiBackend: 'comfyui' | 'mock' | 'runpod' | 'stream' | 'auto';
  /** Base URL of the model-resident worker in apps/stream-worker. */
  streamUrl: string;
  streamTimeoutMs: number;
  /** Let `auto` consider the stream worker at all (off: explicit-only). */
  streamAuto: boolean;
  comfyUrl: string;
  comfyCheckpoint: string;
  /** World canvas size in px (square). */
  canvasSize: number;
  /** 'full' regenerates the whole canvas; 'patch' uses crops + dirty regions. */
  aiMode: 'full' | 'patch';
  aiWindow: number;
  aiApply: number;
  aiDenoise: number;
  aiCfg: number;
  /** VAE decode tile size; 0 uses a plain (non-tiled) VAEDecode. */
  aiVaeTile: number;
  /** Default workflow for new rooms; each room can switch at runtime. */
  aiProfile: AIProfileName;
  /** Sampler steps for the quality profile. */
  aiSteps: number;
  /** Sampler steps for the fast (LCM) profile. */
  aiFastSteps: number;
  /** The fast profile was asked for but COMFYUI_FAST_LORA is empty. */
  fastDisabled: boolean;
  /** LoRA used by the fast profile; empty forces every room to quality. */
  comfyFastLora: string;
  aiDebounceMs: number;
  aiWatchdogMs: number;
  roomIdleMs: number;
  runpodEndpointId: string;
  runpodApiKey: string;
  webDist: string | null;
}

/** `1`, `true`, `yes`, `on` (case-insensitive) are true; anything else false. */
const flag = (raw: string | undefined): boolean => ['1', 'true', 'yes', 'on'].includes((raw ?? '').trim().toLowerCase());

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

/**
 * Per-backend starting values for a room, applied when the operator has not
 * chosen explicitly. The stream worker is a 4-step LCM worker: 0.7 barely
 * moves the drawing there, and it is fastest at 768.
 * See docs/experiments/2026-09-05-stream/REPORT.md.
 */
export const STREAM_DEFAULTS = { resolution: 768, denoise: 0.8 } as const;

/**
 * Apply the backend's own defaults to a config the operator did not pin.
 * Called once the backend is known, because `auto` only resolves at startup.
 */
export function applyBackendDefaults(config: Config, backendName: string, env: NodeJS.ProcessEnv = process.env): Config {
  if (backendName !== 'stream') return config;
  const next = { ...config, aiProfile: 'fast' as const };
  if (env.AI_WINDOW === undefined) next.aiWindow = Math.min(STREAM_DEFAULTS.resolution, config.canvasSize);
  if (env.AI_DENOISE === undefined) next.aiDenoise = STREAM_DEFAULTS.denoise;
  return next;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const errors: string[] = [];
  const backendRaw = (env.AI_BACKEND ?? 'auto').toLowerCase();
  const modeRaw = (env.AI_MODE ?? 'full').toLowerCase();
  if (!['full', 'patch'].includes(modeRaw)) {
    errors.push(`AI_MODE must be full or patch (got ${JSON.stringify(env.AI_MODE)})`);
  }
  if (!['auto', 'comfyui', 'mock', 'runpod', 'stream'].includes(backendRaw)) {
    errors.push(`AI_BACKEND must be one of auto, comfyui, mock, runpod, stream (got ${JSON.stringify(env.AI_BACKEND)})`);
  }

  // The fast profile is only fast if there is a LoRA to load. Asking for it
  // with COMFYUI_FAST_LORA='' must fall all the way back to quality, not leave
  // a 4-step euler_ancestral at cfg 5.5, which is neither profile.
  const fastLora = (env.COMFYUI_FAST_LORA ?? 'lcm-lora-sdxl.safetensors').trim();
  // AI_PROFILE is the setting. AI_FAST survives as an alias so existing
  // scripts keep working: AI_FAST=1 means fast, AI_FAST=0 means quality.
  const aliased = env.AI_FAST === undefined ? undefined : flag(env.AI_FAST) ? 'fast' : 'quality';
  const profileRaw = (env.AI_PROFILE ?? aliased ?? 'fast').toLowerCase();
  if (!AI_PROFILES.includes(profileRaw as AIProfileName)) {
    errors.push(`AI_PROFILE must be fast or quality (got ${JSON.stringify(env.AI_PROFILE)})`);
  }
  const requested: AIProfileName = profileRaw === 'quality' ? 'quality' : 'fast';
  const fastRequested = requested === 'fast';
  const fast = fastRequested && fastLora !== '';
  const profile: AIProfileName = fast ? 'fast' : 'quality';

  const config: Config = {
    host: env.HOST ?? '127.0.0.1',
    port: num(env, 'PORT', 8787, { min: 1, max: 65535, integer: true }, errors),
    aiBackend: (['comfyui', 'mock', 'runpod', 'stream'].includes(backendRaw) ? backendRaw : 'auto') as Config['aiBackend'],
    comfyUrl: (env.COMFYUI_URL ?? 'http://127.0.0.1:8188').replace(/\/+$/, ''),
    streamUrl: (env.STREAM_URL ?? 'http://127.0.0.1:8790').replace(/\/+$/, ''),
    streamTimeoutMs: num(env, 'STREAM_TIMEOUT_MS', 120_000, { min: 1000, max: 3_600_000, integer: true }, errors),
    streamAuto: flag(env.AI_STREAM_AUTO),
    comfyCheckpoint: env.COMFYUI_CHECKPOINT ?? 'waiNSFWIllustrious_v150.safetensors',
    canvasSize: num(env, 'CANVAS_SIZE', 1024, { min: 512, max: 4096, integer: true, multipleOf: 64 }, errors),
    aiMode: modeRaw === 'patch' ? 'patch' : 'full',
    // The profile picks the window when the operator has not: fast is only
    // worth having if it is also smaller (768 measured 3.7 s against 5.7 s).
    aiWindow: num(env, 'AI_WINDOW', PROFILE_DEFAULTS[profile].resolution, { min: 256, max: 2048, integer: true, multipleOf: 64 }, errors),
    aiApply: num(env, 'AI_APPLY', 768, { min: 128, max: 2048, integer: true, multipleOf: 64 }, errors),
    aiSteps: num(env, 'AI_STEPS', PROFILE_DEFAULTS.quality.steps, { min: 1, max: 150, integer: true }, errors),
    aiFastSteps: num(env, 'AI_FAST_STEPS', PROFILE_DEFAULTS.fast.steps, { min: 1, max: 150, integer: true }, errors),
    aiDenoise: num(env, 'AI_DENOISE', PROFILE_DEFAULTS[profile].denoise, { min: 0, max: 1 }, errors),
    aiCfg: num(env, 'AI_CFG', 5.5, { min: 0, max: 30 }, errors),
    aiVaeTile: num(env, 'AI_VAE_TILE', 512, { min: 0, max: 4096, integer: true }, errors),
    aiProfile: profile,
    fastDisabled: fastRequested && !fast,
    comfyFastLora: fastLora,
    aiDebounceMs: num(env, 'AI_DEBOUNCE_MS', 400, { min: 0, max: 600_000, integer: true }, errors),
    aiWatchdogMs: num(env, 'AI_WATCHDOG_MS', 180_000, { min: 1000, max: 3_600_000, integer: true }, errors),
    roomIdleMs: num(env, 'ROOM_IDLE_MS', 30 * 60_000, { min: 10_000, max: 24 * 3_600_000, integer: true }, errors),
    runpodEndpointId: env.RUNPOD_ENDPOINT_ID ?? '',
    runpodApiKey: env.RUNPOD_API_KEY ?? '',
    webDist: env.WEB_DIST ?? null,
  };

  // ComfyUI's VAEDecodeTiled has a minimum tile of 64; 0 means "do not tile".
  if (config.aiVaeTile !== 0 && config.aiVaeTile < 64) {
    errors.push(`AI_VAE_TILE must be 0 (no tiling) or at least 64 (got ${config.aiVaeTile})`);
  }
  // The room slider only offers 0.2..0.95 in 0.05 steps; a start value outside
  // that would be silently snapped, so say so instead.
  if (config.aiDenoise < 0.2 || config.aiDenoise > 0.95) {
    errors.push(`AI_DENOISE must be between 0.2 and 0.95 (got ${config.aiDenoise})`);
  } else if (Math.abs(config.aiDenoise * 100 - Math.round((config.aiDenoise * 100) / 5) * 5) > 1e-6) {
    errors.push(`AI_DENOISE must be a multiple of 0.05 (got ${config.aiDenoise})`);
  }

  if (config.aiMode === 'full') {
    // The whole canvas is regenerated, but not necessarily at canvas
    // resolution: AI_WINDOW is the *generation* size, so an 8 GB card can run a
    // 1024 canvas at 768 or 512 and the result is scaled back up.
    if (config.canvasSize > 2048) {
      errors.push(`AI_MODE=full needs CANVAS_SIZE <= 2048 (got ${config.canvasSize}); use AI_MODE=patch for a large canvas`);
    }
    // The profile already chose the generation size (fast 768, quality 1024);
    // only clamp it to a canvas that is smaller than that.
    if (env.AI_WINDOW === undefined) config.aiWindow = Math.min(config.aiWindow, config.canvasSize);
    if (config.aiWindow < 512) {
      errors.push(`AI_MODE=full needs AI_WINDOW >= 512 (got ${config.aiWindow})`);
    }
    // the whole canvas is always the applied area in full mode
    config.aiApply = config.canvasSize;
  }

  // In full mode the apply area is the canvas and the window is only the
  // generation resolution, so the patch-mode relation between them does not apply.
  if (config.aiMode === 'patch' && config.aiApply > config.aiWindow) {
    errors.push(`AI_APPLY (${config.aiApply}) must not exceed AI_WINDOW (${config.aiWindow})`);
  }
  try {
    new URL(config.comfyUrl);
  } catch {
    errors.push(`COMFYUI_URL is not a valid URL (got ${JSON.stringify(config.comfyUrl)})`);
  }
  try {
    new URL(config.streamUrl);
  } catch {
    errors.push(`STREAM_URL is not a valid URL (got ${JSON.stringify(config.streamUrl)})`);
  }
  if (config.aiBackend === 'runpod' && (!config.runpodEndpointId || !config.runpodApiKey)) {
    errors.push('AI_BACKEND=runpod requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY');
  }

  if (errors.length > 0) throw new ConfigError(`invalid configuration:\n  - ${errors.join('\n  - ')}`);
  return config;
}
