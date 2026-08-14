import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { WIDGET_NAMES } from '../src/components/viz/widget-names.mjs';
import { WIDGETS } from '../src/components/viz/registry';

const game = JSON.parse(readFileSync(resolve(__dirname, '../../out/game.json'), 'utf8'));

// Every data-widget name that appears anywhere in the REAL compiled content.
// Not a fixture: a fixture written from the same mental model as the registry
// cannot falsify it (LEARNINGS, 2026-07-13).
function widgetNamesInContent(): string[] {
  // This regexes ALL of game.scenes as JSON text, which also matches
  // `data-widget="…"` sitting inside `$code` comment strings (Dendry
  // preserves scene source comments in the compiled output) — not just
  // live markup. Currently harmless: the one comment hit is
  // `data-widget="coalitions"`, already a known name. A future typo'd
  // widget name written only inside a comment would spuriously fail the
  // "known name" assertion below even though no real content declares it.
  // Not rewritten here (Task 6 fix round 1, M5) — flagged for whoever next
  // touches this scanner.
  const found = new Set<string>();
  const walk = (node: unknown) => {
    if (typeof node === 'string') {
      for (const m of node.matchAll(/data-widget=["']([\w-]+)["']/g)) found.add(m[1]);
    } else if (Array.isArray(node)) node.forEach(walk);
    else if (node && typeof node === 'object') Object.values(node).forEach(walk);
  };
  walk(game.scenes);
  return [...found];
}

describe('widget registry', () => {
  // NOT "every WIDGET_NAME has a component" — law-grid remains intentionally
  // old-shell-only. Instead: the registry must never
  // register a name the guard wouldn't recognise.
  it('registers a component only for known names (WIDGETS keys ⊆ WIDGET_NAMES)', () => {
    for (const key of Object.keys(WIDGETS)) {
      expect(WIDGET_NAMES, `WIDGETS registers unknown name "${key}"`).toContain(key);
    }
  });

  it('every data-widget in the real game.json is a known name', () => {
    for (const n of widgetNamesInContent())
      expect(WIDGET_NAMES, `content declares unknown widget "${n}"`).toContain(n);
  });
});
