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
});
