import { describe, expect, it } from 'vitest';
import { ConfigError, loadConfig } from '../src/config.js';

const env = (extra: Record<string, string> = {}): NodeJS.ProcessEnv => ({ ...extra }) as NodeJS.ProcessEnv;

/** Finding 15: bad configuration must fail fast, not produce broken crops. */
describe('loadConfig', () => {
  it('uses the documented defaults', () => {
    // patch mode: full mode deliberately locks the window to the canvas
    expect(loadConfig(env({ AI_MODE: 'patch' }))).toMatchObject({
      host: '127.0.0.1',
      port: 8787,
      aiBackend: 'auto',
      aiWindow: 1024,
      aiApply: 768,
      aiSteps: 14,
      aiDenoise: 0.55,
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
  it('defaults to a 1024 canvas in full mode, with the window locked to it', () => {
    const c = loadConfig({} as NodeJS.ProcessEnv);
    expect(c.canvasSize).toBe(1024);
    expect(c.aiMode).toBe('full');
    expect(c.aiWindow).toBe(1024);
    expect(c.aiApply).toBe(1024);
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
  it('defaults to the canvas size in full mode', () => {
    const c = loadConfig({ CANVAS_SIZE: '1024' } as NodeJS.ProcessEnv);
    expect(c.aiWindow).toBe(1024);
    expect(c.aiApply).toBe(1024);
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
