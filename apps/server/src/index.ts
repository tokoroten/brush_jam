import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createBackend } from './ai/backends/index.js';
import { loadConfig, type Config } from './config.js';
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

let config: Config;
try {
  config = loadConfig();
} catch (err) {
  console.error(`[brushjam] ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}

const backend = await createBackend(config);
const { server, close } = createBrushJamServer(config, backend);

server.listen(config.port, config.host, () => {
  console.log(`[brushjam] server on http://${config.host}:${config.port} (set HOST=0.0.0.0 to expose on the LAN)`);
  console.log(`[brushjam] ai window ${config.aiWindow} / apply ${config.aiApply} / steps ${config.aiSteps} / denoise ${config.aiDenoise}`);
});

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    void close().then(() => process.exit(0));
  });
}
