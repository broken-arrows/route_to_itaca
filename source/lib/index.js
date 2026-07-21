/* =============================================================================
 * source/lib/index.js — the GAME's own code, handed to the engine.
 * =============================================================================
 * Compiled scene code reaches everything here as `G`:
 *
 *     on-arrival: {! G.engineTick(Q); !}
 *
 * The engine receives it via `engine.setGameLib(lib)` — each UI hands it over,
 * and neither UI knows what is inside. This is the whole reason `source/` can be
 * the game and `ui/` can be game-agnostic.
 *
 * Dual-consumable on purpose: Vite/vitest `require` it; the OLD shell has no
 * bundler and loads it as a plain <script>, so it also publishes
 * `window.RTI_GAME_LIB`. That fallback dies with the old shell at phase 6.
 *
 * Everything here must be PURE and DOM-FREE. That is what makes it shareable —
 * `cat_engine.js` has zero `d3`/`document` references, which is the only reason
 * this works at all. If you need the DOM, you are writing a UI, not a game lib.
 * ========================================================================== */
(function () {
  'use strict';

  // Each lib module is dual-consumable: a bundler `require`s it; the old shell
  // (no bundler) loads it as a <script> that publishes a window global. Same
  // branch for all of them.
  var hasModule = typeof require !== 'undefined' && typeof module !== 'undefined';
  var catEngine = hasModule ? require('./cat_engine.js') : window.RTI_CAT_ENGINE;
  var allegiances = hasModule ? require('./allegiances.js') : window.RTI_ALLEGIANCES;
  var brief = hasModule ? require('./brief.js') : window.RTI_BRIEF;

  // AGGREGATION point. Object.assign forwards EVERYTHING each module exports —
  // a new sim helper only needs a line in cat_engine.js's `api`; a new lib FILE
  // needs one line here (its require + a slot in this merge) and nothing in
  // either UI. (Object.assign ignores a null/undefined source, so a failed
  // fallback degrades to a smaller lib rather than throwing on load.)
  var lib = Object.assign({}, catEngine, allegiances, brief);

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = lib;
  } else {
    window.RTI_GAME_LIB = lib;
  }
})();
