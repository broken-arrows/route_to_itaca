/* =============================================================================
 * game-bindings.ts — the ONE Route-to-Ítaca-specific module in `ui/`.
 * =============================================================================
 *
 * WHAT THIS IS
 * The compiled game content calls browser globals that the *game*, not the
 * engine, owns:
 *
 *   window.engineTick(Q)        source/scenes/post_event.scene.dry:73 — the
 *                               monthly macro simulation (GDP, unemployment,
 *                               welfare, cat–spa relations, dissent, the
 *                               parlament vote matrices). Runs on EVERY turn.
 *   window.spaSupportInject(…)  56 calls across the congreso-coalition scenes.
 *
 * Both are defined by `out/html/cat_engine.js`, a DOM-free IIFE whose last two
 * lines are `window.engineTick = monthPasses; window.spaSupportInject = …`.
 * The old UI installs it with a `<script src="cat_engine.js">` tag
 * (`out/html/index.html:18`). The Vue app has no script tags, so without this
 * side-effect import `window.engineTick` is `undefined` — and dendry's
 * `runActions` (`vendor/dendrynexus-ten/lib/engine.js`) CATCHES the resulting
 * TypeError, so the whole simulation silently never runs while the game looks
 * perfectly alive. (That is exactly how it hid for a full phase; see
 * `docs/design/LEARNINGS.md`, 2026-07-13.)
 *
 * WHY IT IS A SEPARATE FILE AND NOT A LINE IN `main.ts`
 * (a) `ui/` is meant to become a game-agnostic dendrynexus shell. This file is
 *     the only module in it that knows the game is Route to Ítaca. A different
 *     dendrynexus game swaps THIS FILE and nothing else. Keep it that way:
 *     game-specific coupling belongs here, greppable, in one place.
 * (b) `cat_engine.js` is imported VERBATIM from the old UI's tree — deliberately
 *     not copied. One file, one macro model, both UIs; there is no second copy
 *     to drift. Do not fork it, do not port it "cleanly" without deleting the
 *     original in the same change.
 * (c) BLOCKING AT PHASE 6 (the swap): phase 6 deletes `out/html/`. `cat_engine.js`
 *     must be RELOCATED first (e.g. `shared/sim/cat_engine.js`) and BOTH consumers
 *     updated — this import, and `out/html/index.html`'s script tag while it still
 *     exists. Recorded in `docs/design/desk_ui_plan.md`'s phase-6 section.
 * (d) LONG TERM: the genuinely game-agnostic end state is for a shell to load its
 *     game's runtime scripts from configuration — the app already fetches
 *     `game.json` at runtime, so a `runtimeScripts: [...]` field there (or in a
 *     sibling manifest) is the natural shape. That is a real design task with real
 *     questions (module vs classic script, sandboxing, load order, test harness),
 *     NOT something to improvise. Until it is designed, this file is the seam.
 *
 * Import for side effects only: the module has no exports; evaluating it installs
 * the globals on `window`. `main.ts` imports it once, before the app mounts.
 * Tests that drive the REAL game (`ui/tests/integration.desk-loop.test.ts`) must
 * import it too, and must assert the simulation actually RAN — not merely that
 * the calendar advanced.
 */
import '../../out/html/cat_engine.js';
