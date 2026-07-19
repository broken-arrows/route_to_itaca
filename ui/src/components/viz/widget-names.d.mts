// Co-located declaration for the plain-.mjs widget-names.mjs. This is the
// SAME "co-located sibling" trick as source/lib/index.d.ts
// (docs/design/LEARNINGS.md, 2026-07-14 #1) with one wrinkle specific to a
// `.mjs` implementation file: TypeScript's resolver wants `.d.mts` here, not
// plain `.d.ts` — a `widget-names.d.ts` sibling was tried first and left
// `TS7016: Could not find a declaration file` in place; renaming to
// `.d.mts` (this file) fixed it with zero other changes.
//
// Declared as a literal readonly TUPLE (not `readonly string[]`) so that
// `WidgetName = (typeof WIDGET_NAMES)[number]` in registry.ts collapses to
// the actual string-literal union instead of widening to `string`. The
// original `readonly string[]` here made `WidgetName` equal to `string`,
// so a typo'd key in `WIDGETS` (registry.ts) type-checked with NO error —
// the compile-time guard didn't exist (Task 6 fix round 1, finding I1).
// This tuple is a second copy of the names in widget-names.mjs and MUST be
// kept in sync with it by hand; widget-names.mjs remains the single source
// of truth read at runtime (by Node, in tools/audit-globals.mjs, and by
// ui/tests/widget.registry.test.ts against the real compiled content) —
// that runtime check is what catches an unknown/mistyped widget name in
// actual content. This literal tuple's only job is to make a bad WIDGETS
// key fail `npm run typecheck`.
export declare const WIDGET_NAMES: readonly [
  'hemicycle',
  'achievement-gallery',
  'poll-map',
  'coalitions',
];
