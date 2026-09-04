import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import { ConfigError, envOrigin, loadConfig, stepsForProfile } from '../src/config.js';

const env = (extra: Record<string, string> = {}): NodeJS.ProcessEnv => ({ ...extra }) as NodeJS.ProcessEnv;

/** Finding 15: bad configuration must fail fast, not produce broken crops. */
describe('loadConfig', () => {
  it('uses the documented defaults', () => {
    // patch mode: full mode derives the window from the profile instead
    expect(loadConfig(env({ AI_MODE: 'patch', AI_PROFILE: 'quality' }))).toMatchObject({
      host: '127.0.0.1',
      port: 8787,
      aiBackend: 'auto',
      aiWindow: 1024,
      aiApply: 768,
      aiSteps: 14,
      aiDenoise: 0.7,
      aiDebounceMs: 400,
    });
  });

  it.each([
    ['AI_WINDOW', '-1'],
    ['AI_WINDOW', '1024.5'],
    ['AI_WINDOW', '1000'],
    ['AI_WINDOW', 'big'],
    ['AI_APPLY', '-20'],
    ['AI_APPLY', '100'],
    ['AI_STEPS', '0'],
    ['AI_STEPS', '3.5'],
    ['AI_DENOISE', '1.5'],
    ['AI_DENOISE', '-0.1'],
    ['PORT', '0'],
    ['PORT', '70000'],
    ['PORT', '80.5'],
    ['AI_DEBOUNCE_MS', '-5'],
    ['AI_BACKEND', 'stablediffusion'],
    ['COMFYUI_URL', 'not a url'],
  ])('rejects %s=%s', (key, value) => {
    expect(() => loadConfig(env({ [key]: value }))).toThrow(ConfigError);
  });

  it('rejects an apply area larger than the window', () => {
    expect(() => loadConfig(env({ AI_MODE: 'patch', AI_WINDOW: '512', AI_APPLY: '1024' }))).toThrow(/AI_APPLY/);
  });

  it('requires credentials for the runpod backend', () => {
    expect(() => loadConfig(env({ AI_BACKEND: 'runpod' }))).toThrow(/RUNPOD_ENDPOINT_ID/);
    expect(loadConfig(env({ AI_BACKEND: 'runpod', RUNPOD_ENDPOINT_ID: 'e', RUNPOD_API_KEY: 'k' })).aiBackend).toBe('runpod');
  });

  it('accepts valid overrides', () => {
    const config = loadConfig(
      env({ AI_MODE: 'patch', AI_WINDOW: '1536', AI_APPLY: '1024', AI_DENOISE: '0.85', HOST: '0.0.0.0', PORT: '9000' }),
    );
    expect(config).toMatchObject({ aiWindow: 1536, aiApply: 1024, aiDenoise: 0.85, host: '0.0.0.0', port: 9000 });
  });

  it('reports every problem at once', () => {
    try {
      loadConfig(env({ AI_WINDOW: '3', AI_STEPS: '0', AI_DENOISE: '9' }));
      expect.unreachable();
    } catch (err) {
      const message = (err as Error).message;
      expect(message).toContain('AI_WINDOW');
      expect(message).toContain('AI_STEPS');
      expect(message).toContain('AI_DENOISE');
    }
  });
});

describe('AI_VAE_TILE', () => {
  it('defaults to 512', () => {
    expect(loadConfig({} as NodeJS.ProcessEnv).aiVaeTile).toBe(512);
  });

  it('accepts 0 to mean "plain VAEDecode"', () => {
    expect(loadConfig({ AI_VAE_TILE: '0' } as NodeJS.ProcessEnv).aiVaeTile).toBe(0);
  });

  it('refuses a value outside the node range', () => {
    expect(() => loadConfig({ AI_VAE_TILE: '9000' } as NodeJS.ProcessEnv)).toThrow();
  });
});

describe('canvas size and AI mode', () => {
  it('defaults to a 1024 canvas in full mode, generated at the profile size', () => {
    const c = loadConfig({} as NodeJS.ProcessEnv);
    expect(c.canvasSize).toBe(1024);
    expect(c.aiMode).toBe('full');
    // fast profile: generate at 768 and scale up to the 1024 canvas
    expect(c.aiWindow).toBe(768);
    expect(c.aiApply).toBe(1024);
    expect(loadConfig({ AI_PROFILE: 'quality' } as NodeJS.ProcessEnv).aiWindow).toBe(1024);
  });

  it('locks the apply area to the canvas in full mode, but not the window', () => {
    const c = loadConfig({ CANVAS_SIZE: '768', AI_WINDOW: '1024', AI_APPLY: '512' } as NodeJS.ProcessEnv);
    // AI_WINDOW is the generation resolution and may differ from the canvas
    expect(c.aiWindow).toBe(1024);
    expect(c.aiApply).toBe(768);
  });

  it('leaves window and apply alone in patch mode', () => {
    const c = loadConfig({ AI_MODE: 'patch', CANVAS_SIZE: '4096', AI_WINDOW: '1024', AI_APPLY: '768' } as NodeJS.ProcessEnv);
    expect(c.canvasSize).toBe(4096);
    expect(c.aiWindow).toBe(1024);
    expect(c.aiApply).toBe(768);
  });

  it('refuses a canvas too large to generate in one pass', () => {
    expect(() => loadConfig({ CANVAS_SIZE: '4096' } as NodeJS.ProcessEnv)).toThrow(/AI_MODE=patch/);
  });

  it.each(['500', '1000', '5000'])('refuses CANVAS_SIZE=%s', (value) => {
    expect(() => loadConfig({ CANVAS_SIZE: value } as NodeJS.ProcessEnv)).toThrow(ConfigError);
  });

  it('refuses an unknown AI_MODE', () => {
    expect(() => loadConfig({ AI_MODE: 'sideways' } as NodeJS.ProcessEnv)).toThrow(/AI_MODE/);
  });
});

describe('fail-fast validation', () => {
  it('rejects a VAE tile below the node minimum', () => {
    expect(() => loadConfig({ AI_VAE_TILE: '32' } as NodeJS.ProcessEnv)).toThrow(/AI_VAE_TILE/);
    expect(loadConfig({ AI_VAE_TILE: '64' } as NodeJS.ProcessEnv).aiVaeTile).toBe(64);
    expect(loadConfig({ AI_VAE_TILE: '0' } as NodeJS.ProcessEnv).aiVaeTile).toBe(0);
  });

  it('rejects a denoise the room slider could not represent', () => {
    expect(() => loadConfig({ AI_DENOISE: '0.1' } as NodeJS.ProcessEnv)).toThrow(/AI_DENOISE/);
    expect(() => loadConfig({ AI_DENOISE: '0.99' } as NodeJS.ProcessEnv)).toThrow(/AI_DENOISE/);
    expect(() => loadConfig({ AI_DENOISE: '0.57' } as NodeJS.ProcessEnv)).toThrow(/multiple of 0.05/);
    expect(loadConfig({ AI_DENOISE: '0.85' } as NodeJS.ProcessEnv).aiDenoise).toBe(0.85);
  });
});

describe('generation resolution', () => {
  it('defaults to the profile size, clamped to a smaller canvas', () => {
    const c = loadConfig({ CANVAS_SIZE: '1024', AI_PROFILE: 'quality' } as NodeJS.ProcessEnv);
    expect(c.aiWindow).toBe(1024);
    expect(c.aiApply).toBe(1024);
    // a canvas smaller than the profile default wins: never generate larger
    expect(loadConfig({ CANVAS_SIZE: '512' } as NodeJS.ProcessEnv).aiWindow).toBe(512);
  });

  it('lets AI_WINDOW go below the canvas', () => {
    const c = loadConfig({ CANVAS_SIZE: '1024', AI_WINDOW: '768' } as NodeJS.ProcessEnv);
    expect(c.aiWindow).toBe(768);
    // the whole canvas is still what gets replaced
    expect(c.aiApply).toBe(1024);
  });

  it('lets AI_WINDOW go above the canvas', () => {
    expect(loadConfig({ CANVAS_SIZE: '512', AI_WINDOW: '1024' } as NodeJS.ProcessEnv).aiWindow).toBe(1024);
  });

  it('refuses a generation size below 512 in full mode', () => {
    expect(() => loadConfig({ CANVAS_SIZE: '1024', AI_WINDOW: '256' } as NodeJS.ProcessEnv)).toThrow(/AI_WINDOW/);
  });

  it('still refuses sizes off the 64 grid or out of range', () => {
    expect(() => loadConfig({ AI_WINDOW: '700' } as NodeJS.ProcessEnv)).toThrow(/multiple of 64/);
    expect(() => loadConfig({ AI_WINDOW: '4096' } as NodeJS.ProcessEnv)).toThrow(/AI_WINDOW/);
  });
});

describe('AI profile', () => {
  it('defaults to fast, because 10 s per edit is too slow to work with', () => {
    const c = loadConfig({} as NodeJS.ProcessEnv);
    expect(c.aiProfile).toBe('fast');
    // and the window follows the profile: fast is only fast when it is smaller
    expect(c.aiWindow).toBe(768);
    expect(c.aiSteps).toBe(14);
    expect(c.aiFastSteps).toBe(4);
    expect(c.aiDenoise).toBe(0.7);
  });

  it('uses 1024 for the quality profile', () => {
    const c = loadConfig({ AI_PROFILE: 'quality' } as NodeJS.ProcessEnv);
    expect(c.aiProfile).toBe('quality');
    expect(c.aiWindow).toBe(1024);
  });

  it('lets an explicit AI_WINDOW beat the profile default', () => {
    expect(loadConfig({ AI_PROFILE: 'fast', AI_WINDOW: '1024' } as NodeJS.ProcessEnv).aiWindow).toBe(1024);
    expect(loadConfig({ AI_PROFILE: 'quality', AI_WINDOW: '512' } as NodeJS.ProcessEnv).aiWindow).toBe(512);
  });

  it('keeps AI_FAST working as an alias', () => {
    expect(loadConfig({ AI_FAST: '1' } as NodeJS.ProcessEnv).aiProfile).toBe('fast');
    expect(loadConfig({ AI_FAST: '0' } as NodeJS.ProcessEnv).aiProfile).toBe('quality');
    expect(loadConfig({ AI_FAST: 'yes' } as NodeJS.ProcessEnv).aiProfile).toBe('fast');
    // AI_PROFILE wins when both are set
    expect(loadConfig({ AI_FAST: '1', AI_PROFILE: 'quality' } as NodeJS.ProcessEnv).aiProfile).toBe('quality');
  });

  it('rejects a profile that is not one of the two', () => {
    expect(() => loadConfig({ AI_PROFILE: 'turbo' } as NodeJS.ProcessEnv)).toThrow(/AI_PROFILE/);
  });

  // Review 6 finding 6, kept: fast without a LoRA is not fast, it is a broken
  // 4-step euler_ancestral. Fall all the way back instead.
  it('falls back to quality when no LoRA is configured', () => {
    const c = loadConfig({ AI_PROFILE: 'fast', COMFYUI_FAST_LORA: '' } as NodeJS.ProcessEnv);
    expect(c.aiProfile).toBe('quality');
    expect(c.fastDisabled).toBe(true);
    expect(c.aiWindow).toBe(1024);
  });

  it('treats a whitespace-only LoRA name as empty', () => {
    expect(loadConfig({ AI_PROFILE: 'fast', COMFYUI_FAST_LORA: '   ' } as NodeJS.ProcessEnv).aiProfile).toBe('quality');
  });

  it('does not flag fastDisabled when quality was chosen anyway', () => {
    expect(loadConfig({ AI_PROFILE: 'quality', COMFYUI_FAST_LORA: '' } as NodeJS.ProcessEnv).fastDisabled).toBe(false);
  });

  it('accepts a custom LoRA name and a custom fast step count', () => {
    const c = loadConfig({ COMFYUI_FAST_LORA: 'dmd2.safetensors', AI_FAST_STEPS: '8' } as NodeJS.ProcessEnv);
    expect(c.comfyFastLora).toBe('dmd2.safetensors');
    expect(c.aiFastSteps).toBe(8);
  });

  it('defaults the fast LoRA to DMD2', () => {
    expect(loadConfig({} as NodeJS.ProcessEnv).comfyFastLora).toBe('dmd2_sdxl_4step_lora_fp16.safetensors');
  });

  it('still lets LCM be selected explicitly', () => {
    expect(loadConfig({ COMFYUI_FAST_LORA: 'lcm-lora-sdxl.safetensors' } as NodeJS.ProcessEnv).comfyFastLora).toBe(
      'lcm-lora-sdxl.safetensors',
    );
  });
});

/**
 * Review 8 finding A#5, twice regressed: quality-grid recorded one step count
 * and sent another, so a grid captioned "4 steps" was actually a 14-step run.
 * One function answers the question for both.
 */
describe('stepsForProfile', () => {
  const cfg = loadConfig({ AI_STEPS: '14', AI_FAST_STEPS: '4' } as NodeJS.ProcessEnv);

  it('gives the fast profile its own step count', () => {
    expect(stepsForProfile(cfg, 'fast')).toBe(4);
  });

  it('gives quality the full count', () => {
    expect(stepsForProfile(cfg, 'quality')).toBe(14);
  });

  it('follows a custom AI_FAST_STEPS', () => {
    const custom = loadConfig({ AI_STEPS: '20', AI_FAST_STEPS: '8' } as NodeJS.ProcessEnv);
    expect(stepsForProfile(custom, 'fast')).toBe(8);
    expect(stepsForProfile(custom, 'quality')).toBe(20);
  });

  it('is what quality-grid sends AND records', () => {
    // Guarding the exact regression: the generate() call must not reach for
    // config.aiSteps behind the recorded value's back.
    const src = readFileSync(new URL('../scripts/quality-grid.ts', import.meta.url), 'utf8');
    const request = src.slice(src.indexOf('backend.generate('), src.indexOf('controller.signal'));
    expect(request).toContain('steps: gridSteps');
    expect(request).not.toContain('config.aiSteps');
    expect(src).toContain('const gridSteps = stepsForProfile(config, opts.profile)');
  });
});

/**
 * Which stack am I looking at? With several dev servers open on one repo the
 * usual cause of "it picked mock" is a process that never had AI_BACKEND, so
 * the startup log names where the value came from.
 */
describe('envOrigin', () => {
  it('reports the environment when the variable was already set', () => {
    expect(envOrigin('stream', 'stream')).toBe('environment');
  });

  it('reports .env when the file supplied it', () => {
    expect(envOrigin(undefined, 'stream')).toBe('.env');
  });

  it('reports unset when neither did', () => {
    expect(envOrigin(undefined, undefined)).toBe('unset');
  });

  it('treats an empty environment value as unset, like loadConfig does', () => {
    expect(envOrigin('', 'stream')).toBe('.env');
    expect(envOrigin('', '')).toBe('unset');
  });

  it('still says environment when .env holds a different value', () => {
    // Node's loader never overwrites, so the environment is what took effect.
    expect(envOrigin('stream', 'stream')).toBe('environment');
  });
});

/**
 * The precedence the startup log claims, pinned against Node itself: a real
 * environment variable must beat the file, or `pnpm dev:stream` would be
 * silently overridden by whatever someone left in .env.
 */
describe('.env precedence', () => {
  it('fills gaps without overwriting the environment', () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'brushjam-env-'));
    const file = path.join(dir, '.env');
    writeFileSync(file, 'BRUSHJAM_TEST_SET=from_dotenv\nBRUSHJAM_TEST_GAP=from_dotenv\n');
    process.env.BRUSHJAM_TEST_SET = 'from_environment';
    delete process.env.BRUSHJAM_TEST_GAP;
    try {
      process.loadEnvFile(file);
      expect(process.env.BRUSHJAM_TEST_SET).toBe('from_environment');
      expect(process.env.BRUSHJAM_TEST_GAP).toBe('from_dotenv');
    } finally {
      delete process.env.BRUSHJAM_TEST_SET;
      delete process.env.BRUSHJAM_TEST_GAP;
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
