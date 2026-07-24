import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

const DIR = join(__dirname, '..', '..', 'source', 'qdisplays');
const BANDED = [
  'control', 'dissent', 'social_dissent', 'independence_movement',
  'politics_trust', 'public_debt_qualifier', 'international_opinion',
  'cat_spa_relations', 'roadmap',
];

describe('qdisplay band tokens', () => {
  it('no banded qdisplay carries an inline style', () => {
    for (const name of BANDED) {
      const src = readFileSync(join(DIR, `${name}.qdisplay.dry`), 'utf8');
      expect(src, `${name} still has inline style=`).not.toMatch(/style\s*=/);
    }
  });

  it('every banded qdisplay line declares a data-band', () => {
    for (const name of BANDED) {
      const src = readFileSync(join(DIR, `${name}.qdisplay.dry`), 'utf8');
      const valueLines = src.split('\n').filter((l) => /^\(/.test(l.trim()));
      expect(valueLines.length, `${name} has no value lines`).toBeGreaterThan(0);
      for (const line of valueLines) {
        expect(line, `${name}: ${line}`).toMatch(/data-band="[a-z_]+"/);
      }
    }
  });

  it('no OTHER qdisplay was touched', () => {
    const untouched = readdirSync(DIR)
      .filter((f) => f.endsWith('.qdisplay.dry'))
      .filter((f) => !BANDED.includes(f.replace('.qdisplay.dry', '')));
    for (const f of untouched) {
      expect(readFileSync(join(DIR, f), 'utf8')).not.toMatch(/data-band=/);
    }
  });

  it('control declares its five rungs in ladder order', () => {
    const src = readFileSync(join(DIR, 'control.qdisplay.dry'), 'utf8');
    const bands = [...src.matchAll(/data-band="([a-z_]+)"/g)].map((m) => m[1]);
    expect(bands).toEqual(['none', 'limited', 'partial', 'complete', 'disputed']);
  });

  it('every banded qdisplay declares data-scale naming itself', () => {
    for (const name of BANDED) {
      const src = readFileSync(join(DIR, `${name}.qdisplay.dry`), 'utf8');
      const scales = new Set([...src.matchAll(/data-scale="([a-z_]+)"/g)].map((m) => m[1]));
      expect([...scales], `${name} data-scale`).toEqual([name]);
    }
  });
});

describe('qdisplay word-level coverage — round-trip through engine', () => {
  let a: DendryAdapter;
  beforeAll(() => {
    const GAME = join(__dirname, '..', '..', 'out', 'game.json');
    a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
  });

  it('cat_spa_relations is INVERTED — low value = very high relation', () => {
    // Regression guard for the deliberate inversion: lowest range (..15) should
    // carry "very high" while highest range (85..) should carry "very low".
    // This is opposite to every other scale and MUST NOT be accidentally undone.
    expect(a.qdisplay(10, 'cat_spa_relations')).toContain('very high');
    expect(a.qdisplay(90, 'cat_spa_relations')).toContain('very low');
  });

  it('public_debt_qualifier classifies low and high values correctly', () => {
    expect(a.qdisplay(5, 'public_debt_qualifier')).toContain('excellent');
    expect(a.qdisplay(45, 'public_debt_qualifier')).toContain('terrible');
  });

  it('roadmap classifies its three pathways distinctly', () => {
    expect(a.qdisplay(1, 'roadmap')).toContain('Vote on agreement');
    expect(a.qdisplay(3, 'roadmap')).toContain('Unilateralism');
  });
});

describe('cat_spa_relations inversion guard — token-level source check', () => {
  it('lowest range (..15) carries data-band="very_high" — NOT very_low', () => {
    const src = readFileSync(join(DIR, 'cat_spa_relations.qdisplay.dry'), 'utf8');
    const lowestRangeHasCorrectBand = /^\(\.\.\d+\)\s*<span[^>]*data-band="very_high"/m.test(src);
    expect(
      lowestRangeHasCorrectBand,
      'cat_spa_relations lowest range (..15) must have data-band="very_high" — this is the inversion guard'
    ).toBe(true);
  });

  it('highest range (85..) carries data-band="very_low" — NOT very_high', () => {
    const src = readFileSync(join(DIR, 'cat_spa_relations.qdisplay.dry'), 'utf8');
    const highestRangeHasCorrectBand = /^\(\d+\.\.\)\s*<span[^>]*data-band="very_low"/m.test(src);
    expect(
      highestRangeHasCorrectBand,
      'cat_spa_relations highest range (85..) must have data-band="very_low" — this is the inversion guard'
    ).toBe(true);
  });
});
