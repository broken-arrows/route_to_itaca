/**
 * Fails the build if compiled scene code reaches for a UI.
 *
 * THE RULE: content may reference `Q` (state) and `G` (the game's own code, via
 * engine.setGameLib). `window.`, `document.` and `d3.` are build errors.
 *
 * WHY THIS EXISTS: dendry's runActions wraps scene code in a try/catch that only
 * LOGS. `window.engineTick` was undefined in the Vue app for an entire phase —
 * the whole macro simulation dead, the calendar still ticking, 175 tests green.
 * Nothing but a build-time check can see this class of bug.
 *
 * Usage: node tools/audit-globals.mjs   (exit 1 + report)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { WIDGET_NAMES, DERIVE_NAMES } from '../ui/src/components/viz/widget-names.mjs';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const game = JSON.parse(readFileSync(join(root, 'out', 'game.json'), 'utf8'));

// `globalThis.` is banned alongside `window.` — it is the same escape hatch
// under a different name (and it works in Node too, so it would even pass the
// headless tests while still coupling content to an environment). Bracket
// access (window['x']) and bare `self.` are not matched: neither appears in
// content today, and `self` is a common local-variable name — add them here
// only with a check against real usage.
const BANNED = /\b(window|document|globalThis)\.|(^|[^.\w])d3\./;

// Scenes not yet migrated. THIS LIST IS THE LEDGER OF REMAINING DEBT — it only
// ever shrinks. Adding to it requires a reason in the PR.
const ALLOW = new Set([
  // phase 4/5 surfaces: d3-in-on-display charts, allowlisted until those phases build the widgets.
  'parlament_election',
  'congreso_election',
  'election_simulation',
  // root's window.location origin check — an environment check, not game logic (user, 2026-07-13).
  'root',
]);

const violations = [];
const widgetsUsed = new Set();
const derivesUsed = new Set();

const walk = (sceneId, node) => {
  if (typeof node === 'string') {
    for (const m of node.matchAll(/data-widget=["']([\w-]+)["']/g)) widgetsUsed.add(m[1]);
    // Both forms are needed: dendry's compiler HTML-escapes attribute quotes
    // in some content positions and not others. Matching only the raw form
    // silently misses half the markers. (Verified against the real compiled
    // game.json for the existing `configFrom` markers: today only the raw
    // form ever appears — 3/3 — with zero `&quot;`-escaped occurrences
    // anywhere in the file. Both patterns are kept anyway: harmless, and this
    // guards against a scene authored in whichever content position DOES
    // escape, the day one exists.)
    for (const m of node.matchAll(/&quot;deriveFrom&quot;\s*:\s*&quot;([\w-]+)&quot;/g)) derivesUsed.add(m[1]);
    for (const m of node.matchAll(/"deriveFrom"\s*:\s*"([\w-]+)"/g)) derivesUsed.add(m[1]);
    return;
  }
  if (Array.isArray(node)) return node.forEach((n) => walk(sceneId, n));
  if (!node || typeof node !== 'object') return;

  if (typeof node.$code === 'string' && BANNED.test(node.$code)) {
    const hit = node.$code.match(BANNED)[0];
    violations.push({ sceneId, hit, src: node.$code.trim().slice(0, 120) });
  }
  Object.values(node).forEach((n) => walk(sceneId, n));
};

for (const [id, scene] of Object.entries(game.scenes)) {
  const top = id.split('.')[0];
  if (ALLOW.has(id) || ALLOW.has(top)) continue;
  walk(id, scene);
}

// Second job: a data-widget nobody can render is an empty div forever. Catch the
// typo at build time.
const unknown = [...widgetsUsed].filter((w) => !WIDGET_NAMES.includes(w));
const unknownDerives = [...derivesUsed].filter((d) => !DERIVE_NAMES.includes(d));

if (violations.length === 0 && unknown.length === 0 && unknownDerives.length === 0) {
  console.log(`audit-globals: clean (${Object.keys(game.scenes).length} scenes, ` +
              `${widgetsUsed.size} widgets, ${derivesUsed.size} derivations, ` +
              `${ALLOW.size} allowlisted)`);
  process.exit(0);
}

for (const v of violations) {
  console.error(`\n✘ ${v.sceneId}: content reaches a UI via \`${v.hit}\`\n    ${v.src}`);
}
for (const w of unknown) {
  console.error(`\n✘ unknown widget "${w}" — not in ui/src/components/viz/widget-names.mjs`);
}
for (const d of unknownDerives) {
  console.error(`\n✘ unknown deriveFrom "${d}" — not in DERIVE_NAMES ` +
                `(ui/src/components/viz/widget-names.mjs) / G.brief (source/lib/brief.js)`);
}
console.error(
  `\naudit-globals FAILED: ${violations.length} global(s), ${unknown.length} unknown widget(s), ` +
  `${unknownDerives.length} unknown derivation(s).\n` +
  'Content computes and declares; content never renders. Use G.* (engine.setGameLib)\n' +
  'for game code, and a data-widget marker to declare a view.\n');
process.exit(1);
