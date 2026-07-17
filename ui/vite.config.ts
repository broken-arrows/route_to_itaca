/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { gameAssets } from './vite-plugin-game-assets';

export default defineConfig({
  base: './',
  // gameAssets(): the game's art (`img/...` paths inside game.json) lives in the
  // OLD UI's web root and the Vue app has no `img/` of its own — so every image
  // 404'd into HandCard's placeholder. Serves/copies it. See the plugin's header
  // for the phase-6 relocation this defers.
  plugins: [vue(), gameAssets()],
  server: {
    fs: {
      // `src/game-bindings.ts` imports the game's macro simulation from
      // `source/lib/index.js` — the GAME's own code, handed to the engine via
      // `setGameLib` (see that file's header). It lives outside this Vite root,
      // and `ui/package-lock.json` makes Vite's workspace-root search stop AT
      // `ui/`, so the default fs.allow does not cover it: a cold request for
      // `/@fs/…/source/lib/index.js` 403s. (It happens to succeed once the
      // module graph is warm from `/src/main.ts`, which is an implementation
      // detail of Vite's serving check — do not rely on it.) Allow the repo
      // root explicitly. Dev-server only; the production build resolves the
      // import through Rollup and needs nothing here.
      allow: ['..'],
    },
  },
  optimizeDeps: {
    // The dendry engine is CommonJS (`module.exports = {convert, convertLine}`)
    // AND a `file:` dependency junctioned into node_modules. Vite does not
    // pre-bundle LINKED packages by default — it resolves the junction to the
    // real `vendor/` path and serves the file as raw source. Raw CJS has no ESM
    // named exports, so the browser dies with:
    //   "does not provide an export named 'convertLine'"
    // Listing the engine's entrypoints here forces esbuild to pre-bundle them,
    // which is what performs the CJS -> ESM interop.
    //
    // This bites the DEV SERVER ONLY, which is why it hid for two phases: vitest
    // runs the engine through Node's native `require` (CJS is fine) and the
    // production build goes through Rollup + `build.commonjsOptions` below.
    // Neither exercises Vite's module server. Do not "simplify" this away
    // without loading the app in a real browser.
    // `source/lib/index.js` (the GAME's own code) is CommonJS for the SAME
    // reason as the engine above, but it is a RELATIVE import (`game-bindings.ts`
    // imports it by path, not as a package) — `optimizeDeps.include` only
    // redirects BARE/package specifiers to its pre-bundled cache, so listing a
    // relative path here has NO effect on what actually gets served (confirmed:
    // it does get pre-bundled into `.vite/deps`, but the import is never
    // rewritten to point at it). `game-bindings.ts` works around the dev
    // server serving it raw with a namespace import + a `window.RTI_GAME_LIB`
    // fallback instead — see that file's header. Do not "fix" this by adding a
    // relative path here again; it doesn't do anything.
    include: [
      'dendrynexus-ten/lib/engine.js',
      'dendrynexus-ten/lib/ui/content/html.js',
    ],
  },
  build: {
    // Same CJS engine, other bundler: Rollup resolves the junction to its real
    // `vendor/dendrynexus-ten` path, which falls outside the default CJS-interop
    // scope (/node_modules/). Widen the include so the named exports are
    // detected in the production build. `source/lib` (the GAME's own code —
    // game-bindings.ts) is the same story: it's CommonJS (`module.exports`,
    // dual-consumable with the old shell's plain <script> tags) and lives
    // outside node_modules, so without this Rollup treats it as ESM with no
    // exports and the build fails with `"default" is not exported by
    // "../source/lib/index.js"`.
    commonjsOptions: {
      include: [/node_modules/, /vendor[\\/]dendrynexus-ten/, /source[\\/]lib/],
    },
  },
  test: {
    environment: 'jsdom',
  },
});
