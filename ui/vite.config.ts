/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  base: './',
  plugins: [vue()],
  build: {
    // The dendry engine is a `file:` dependency junctioned into node_modules;
    // Rollup resolves the junction to its real `vendor/dendrynexus` path, which
    // falls outside the default CJS-interop scope (/node_modules/). Widen the
    // include so the engine's CommonJS `module.exports` named exports (e.g.
    // convertLine) are detected in the production build. Dev (esbuild) already
    // handles this via dep pre-bundling.
    commonjsOptions: {
      include: [/node_modules/, /vendor[\\/]dendrynexus/],
    },
  },
  test: {
    environment: 'jsdom',
  },
});
