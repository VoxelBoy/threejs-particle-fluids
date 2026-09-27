import { defineConfig } from 'vite';

// Builds the demo site. The library itself is built with `npm run build:lib`.
export default defineConfig({ base: './', build: { target: 'es2022', outDir: 'dist-demo' } });
