import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const game = JSON.parse(readFileSync(resolve(__dirname, '../../out/game.json'), 'utf8'));

describe('game.json.data.glossary', () => {
  const terms = game.data?.glossary?.terms as any[];

  it('is compiled into the game artifact', () => {
    expect(Array.isArray(terms)).toBe(true);
    expect(terms.length).toBeGreaterThan(90);
  });

  it('stores colour TOKENS, not CSS — the old UI must not leak into content', () => {
    for (const t of terms) {
      if (!t.colour) continue;
      expect(t.colour, `term ${t.id}`).not.toMatch(/^var\(/);
    }
  });

  it('carries the party terms the Brief depends on', () => {
    const ids = new Set(terms.map((t) => t.id));
    for (const id of ['ciu', 'erc', 'psc', 'cup']) expect(ids).toContain(id);
  });

  it('every match word maps to exactly one term', () => {
    const seen = new Map<string, string>();
    for (const t of terms)
      for (const w of t.match) {
        expect(seen.has(w), `"${w}" claimed by ${seen.get(w)} and ${t.id}`).toBe(false);
        seen.set(w, t.id);
      }
  });

  // Task 5 (§5 of its brief): the Desk needs a party colour token in
  // tokens.css for every TOKEN (not raw hex — those pass through
  // colourValue() verbatim, see useGlossary.ts) either the glossary OR
  // source/lib/allegiances.js can hand it. An id missing here renders
  // unstyled with zero test signal (jsdom doesn't apply CSS) — this is the
  // one guard that actually catches that class of miss.
  const tokensCss = readFileSync(resolve(__dirname, '../src/styles/tokens.css'), 'utf8');

  it('every colour token in the glossary exists in tokens.css', () => {
    for (const t of terms) {
      if (!t.colour || t.colour.startsWith('#')) continue;
      expect(tokensCss, `--${t.colour} missing from tokens.css (glossary term ${t.id})`).toContain(
        `--${t.colour}:`,
      );
    }
  });

  it('every colour token in source/lib/allegiances.js exists in tokens.css too', () => {
    const allegiancesSrc = readFileSync(
      resolve(__dirname, '../../source/lib/allegiances.js'),
      'utf8',
    );
    const tokens = new Set(
      [...allegiancesSrc.matchAll(/colour:\s*(['"])([^'"]+)\1/g)]
        .map((m) => m[2])
        .filter((c) => !c.startsWith('#')),
    );
    // Sanity on the scan itself — a change to the source's quoting style
    // Both quote styles are valid formatter output.
    expect(tokens.size).toBeGreaterThan(0);
    for (const token of tokens) {
      expect(tokensCss, `--${token} missing from tokens.css (source/lib/allegiances.js)`).toContain(
        `--${token}:`,
      );
    }
  });
});
