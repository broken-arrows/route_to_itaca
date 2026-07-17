/* =============================================================================
 * game-bindings.ts — the ONE Route-to-Ítaca-specific module in `ui/`.
 * =============================================================================
 *
 * The game's own code (its macro simulation) lives in `source/lib/` — in the
 * GAME, not in a UI's web root. Content reaches it as `G` (`G.engineTick(Q)`),
 * and the engine gets it from whichever UI is running:
 *
 *     engine.setGameLib(gameLib)
 *
 * Nothing here installs a browser global, and nothing in `ui/` outside this file
 * knows the game is Route to Ítaca. A different dendrynexus game swaps THIS FILE
 * and nothing else.
 *
 * HISTORY, so nobody re-invents the hole: the simulation used to be
 * `out/html/cat_engine.js`, reached by content as `window.engineTick`. The Vue
 * app has no script tags, so it was `undefined` — and dendry's `runActions`
 * SWALLOWS the resulting TypeError. The whole monthly simulation silently never
 * ran for an entire phase while the calendar kept advancing. See
 * `docs/design/LEARNINGS.md`, 2026-07-13.
 *
 * The guard that keeps it dead: `ui/tests/integration.desk-loop.test.ts` asserts
 * an engine-EXCLUSIVE Q value actually moves (`gdp_growth` / `unemployment` /
 * `welfare_index` — the three nothing else in `source/` writes). Do not weaken
 * it. `tools/audit-globals.mjs` covers the content side.
 * ========================================================================== */
import type { GameLib } from '../../source/lib/index.js';

// `source/lib/*.js` is deliberately plain, dual-consumable CommonJS — no real
// ES `export` keyword, so the OLD shell can still load it as a classic
// <script> tag (see cat_engine.js / index.js headers). That file shape gets
// interop'd into a real default export in TWO of the three places this runs:
// vitest (vite-node hands it to Node's native `require`, which resolves
// `module.exports` directly) and the production build (`vite.config.ts`'s
// `build.commonjsOptions.include` tells Rollup's commonjs plugin to convert
// it). Vite's DEV SERVER is the one place that does neither: a local relative
// import is served raw, un-transformed (`optimizeDeps` only redirects BARE/
// package specifiers to its pre-bundled cache — a relative specifier like this
// one never qualifies, confirmed by curling the dev server's `/@fs/` response
// directly and finding no `export` in it). A plain `import gameLib from
// '...'` (default import) would then fail to even LOAD in a real browser:
// "does not provide an export named 'default'" — a hard SyntaxError before any
// of our code runs, not something a runtime try/catch can catch.
//
// Fix: a NAMESPACE import (`import * as`) never statically requires a
// specific binding to exist, so it loads in all three environments; then fall
// back, at runtime, to the same `window.RTI_GAME_LIB` global `index.js`
// already publishes for the old shell's benefit — which is exactly what fires
// when this file's own `typeof module !== 'undefined'` check goes the "no
// module system" way, i.e. precisely the raw dev-server/browser case this
// works around. The side-effect import of `cat_engine.js` first is required
// only for that same raw-browser path: `index.js`'s own `require('./cat_engine.js')`
// never runs there (no `require`), so without this, `index.js`'s browser
// fallback (`window.RTI_CAT_ENGINE`) would still be unset when it looks for
// it. Harmless everywhere else — Node/vite-node dedupes a file required twice
// via its module cache, so this never double-executes the simulation module.
// Each lib module is side-effect-imported here so its window.RTI_* global is
// set before index.js's raw-dev-server branch reads it (see the long note
// above — this only matters for the un-transformed dev server; vitest and the
// Rollup build resolve index.js's own `require`s directly). A new lib file adds
// one line here. The aggregation/typing shape is deferred to phase 6 (§2.5 spec §10.1).
import '../../source/lib/cat_engine.js';
import '../../source/lib/allegiances.js';
import * as gameLibModule from '../../source/lib/index.js';

const gameLib: GameLib =
  (gameLibModule as { default?: GameLib }).default ??
  (globalThis as unknown as { RTI_GAME_LIB?: GameLib }).RTI_GAME_LIB!;

export function installGameLib(target: { setGameLib(lib: object): unknown }): void {
  target.setGameLib(gameLib);
}

export { gameLib };
