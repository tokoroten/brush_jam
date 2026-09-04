import { describe, expect, it } from 'vitest';
import { ConfigError, loadConfig } from '../src/config.js';

const env = (extra: Record<string, string> = {}): NodeJS.ProcessEnv => ({ ...extra }) as NodeJS.ProcessEnv;

/** Finding 15: bad configuration must fail fast, not produce broken crops. */
describe('loadConfig', () => {
  it('uses the documented defaults', () => {
    expect(loadConfig(env())).toMatchObject({
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
    expect(() => loadConfig(env({ AI_WINDOW: '512', AI_APPLY: '1024' }))).toThrow(/AI_APPLY/);
  });

  it('requires credentials for the runpod backend', () => {
    expect(() => loadConfig(env({ AI_BACKEND: 'runpod' }))).toThrow(/RUNPOD_ENDPOINT_ID/);
    expect(loadConfig(env({ AI_BACKEND: 'runpod', RUNPOD_ENDPOINT_ID: 'e', RUNPOD_API_KEY: 'k' })).aiBackend).toBe('runpod');
  });

  it('accepts valid overrides', () => {
    const config = loadConfig(env({ AI_WINDOW: '1536', AI_APPLY: '1024', AI_DENOISE: '0.85', HOST: '0.0.0.0', PORT: '9000' }));
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
