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
 * - 'poll-map': rendered by the Desk's Phase-3b PollMap and by the old shell's
 *   explicit legacy handler.
 * - 'coalitions': declared by `parlament_coalition.scene.dry` and
 *   `congreso_coalition.scene.dry` (dual-marked alongside their legacy
 *   `#parlament-coalition-widget` id), rendered by Coalitions.vue.
 *   This is also how `window._cvParlement` dies: the
 *   marker's `data-props` names a Q key (`configFrom`) instead of a global.
 * - 'tension-rows', 'level-bars', 'seat-bars', 'roster-rows', 'leader-rows',
 *   'trail': declared by the Brief's six `source/scenes/status/status.scene.dry`
 *   sheets (phase 3b Task 7), each paired with a `deriveFrom` derivation below.
 *   Desk components shipped in Phase 3b Part 2.
 * - 'law-grid': old-shell-only while the frozen status.scene.dry remains live.
 *   Its model comes from G.getLawsForUI; the Desk does not render that legacy
 *   Government sheet.
 */
export const WIDGET_NAMES = [
  "hemicycle",
  "achievement-gallery",
  "poll-map",
  "coalitions",
  "tension-rows",
  "level-bars",
  "seat-bars",
  "roster-rows",
  "leader-rows",
  "trail",
  "chamber-vote",
  "law-grid",
  "roadmaps",
];

/**
 * The derivations content may name via `data-props='{"deriveFrom":"…"}'`.
 * Must match the keys of `G.brief` in source/lib/brief.js. Checked by
 * tools/audit-globals.mjs, so a typo fails the BUILD rather than rendering an
 * empty widget forever — same contract as WIDGET_NAMES above.
 */
export const DERIVE_NAMES = [
  "benches",
  "composition",
  "standing",
  "cabinet",
  "control",
  "chancelleries",
  "factions",
  "street",
  "trails",
  "crosstab",
  "seatProjection",
  "provinces",
];
