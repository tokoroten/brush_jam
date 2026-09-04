import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const shared = fileURLToPath(new URL('../../packages/shared/src/index.ts', import.meta.url));
const target = process.env.BRUSHJAM_SERVER ?? 'http://127.0.0.1:8787';

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@brushjam/shared': shared } },
  server: {
    port: 5173,
    proxy: {
      '/api': { target },
      '/rooms': { target },
      '/ws': { target, ws: true },
      '/healthz': { target },
    },
  },
});
