export interface Config {
  port: number;
  aiBackend: 'comfyui' | 'mock' | 'runpod' | 'auto';
  comfyUrl: string;
  comfyCheckpoint: string;
  aiWindow: number;
  aiApply: number;
  aiSteps: number;
  aiDenoise: number;
  aiCfg: number;
  aiDebounceMs: number;
  runpodEndpointId: string;
  runpodApiKey: string;
  webDist: string | null;
}

const num = (v: string | undefined, fallback: number): number => {
  const n = v === undefined ? NaN : Number(v);
  return Number.isFinite(n) ? n : fallback;
};

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const backend = (env.AI_BACKEND ?? 'auto').toLowerCase();
  return {
    port: num(env.PORT, 8787),
    aiBackend: backend === 'comfyui' || backend === 'mock' || backend === 'runpod' ? backend : 'auto',
    comfyUrl: (env.COMFYUI_URL ?? 'http://127.0.0.1:8188').replace(/\/+$/, ''),
    comfyCheckpoint: env.COMFYUI_CHECKPOINT ?? 'waiNSFWIllustrious_v150.safetensors',
    aiWindow: num(env.AI_WINDOW, 1024),
    aiApply: num(env.AI_APPLY, 768),
    aiSteps: num(env.AI_STEPS, 14),
    aiDenoise: num(env.AI_DENOISE, 0.55),
    aiCfg: num(env.AI_CFG, 5.5),
    aiDebounceMs: num(env.AI_DEBOUNCE_MS, 400),
    runpodEndpointId: env.RUNPOD_ENDPOINT_ID ?? '',
    runpodApiKey: env.RUNPOD_API_KEY ?? '',
    webDist: env.WEB_DIST ?? null,
  };
}
