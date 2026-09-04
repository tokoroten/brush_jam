import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createBackend } from './ai/backends/index.js';
import { loadConfig } from './config.js';
import { createBrushJamServer } from './server.js';

// Optional repo-root .env (RunPod credentials etc). Never logged.
const envFile = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '.env');
if (existsSync(envFile)) {
  try {
    process.loadEnvFile(envFile);
  } catch {
    /* malformed .env is not fatal */
  }
}

const config = loadConfig();
const backend = await createBackend(config);
const { server, close } = createBrushJamServer(config, backend);

server.listen(config.port, () => {
  console.log(`[brushjam] server on http://127.0.0.1:${config.port}`);
  console.log(`[brushjam] ai window ${config.aiWindow} / apply ${config.aiApply} / steps ${config.aiSteps} / denoise ${config.aiDenoise}`);
});

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    void close().then(() => process.exit(0));
  });
}
