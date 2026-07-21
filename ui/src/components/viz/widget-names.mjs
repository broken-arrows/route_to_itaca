/**
 * The widget names content may declare via `data-widget="…"`.
 *
 * Plain .mjs with NO Vue imports on purpose: `tools/audit-globals.mjs` imports
 * this from Node to check that every `data-widget` in the compiled game.json is
 * a name some UI actually knows — so a typo fails the BUILD instead of rendering
 * an empty div forever.
 *
 * - 'hemicycle' / 'achievement-gallery': Task 7/8 give these a Desk component
 *   (registry.ts, below). Not yet referenced by any compiled content.
 * - 'poll-map': declared by `source/scenes/status.scene.dry` (dual-marked
 *   alongside its legacy `#cat-polls-widget` id). No Desk component until
 *   phase 3 — WidgetHost renders the placeholder, which is correct.
 * - 'coalitions': declared by `parlament_coalition.scene.dry` and
 *   `congreso_coalition.scene.dry` (dual-marked alongside their legacy
 *   `#parlament-coalition-widget` id). No Desk component until phase 4 —
 *   same placeholder story. This is also how `window._cvParlement` dies: the
 *   marker's `data-props` names a Q key (`configFrom`) instead of a global.
 */
export const WIDGET_NAMES = ['hemicycle', 'achievement-gallery', 'poll-map', 'coalitions'];

/**
 * The derivations content may name via `data-props='{"deriveFrom":"…"}'`.
 * Must match the keys of `G.brief` in source/lib/brief.js. Checked by
 * tools/audit-globals.mjs, so a typo fails the BUILD rather than rendering an
 * empty widget forever — same contract as WIDGET_NAMES above.
 */
export const DERIVE_NAMES = [
  'benches',
  'composition',
  'cabinet',
  'control',
  'chancelleries',
  'factions',
  'street',
  'trails',
];
