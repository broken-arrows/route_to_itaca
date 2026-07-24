import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';
import { parseQdisplayHtml, useBand } from '../src/components/brief/useBand';
import { useGameStore } from '../src/stores/game';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');

// The five shapes `adapter.qdisplay` was MEASURED to emit (see useBand.ts's
// header). Pinned as literals here so the parser is tested without a boot;
// the round-trip block below re-checks the same shapes against the real
// engine, so a qdisplay that changes form fails there instead of silently
// drifting away from these fixtures.
describe('parseQdisplayHtml — the shapes the engine really emits', () => {
  it('1. a <span> with a data-band token', () => {
    expect(
      parseQdisplayHtml(
        '<span class="q-band" data-scale="social_dissent" data-band="medium">medium</span>',
      ),
    ).toEqual({ band: 'medium', label: 'medium' });
  });

  it('2. a <p>, not a span (international_opinion) — the tag is not fixed', () => {
    expect(
      parseQdisplayHtml(
        '<p class="q-band" data-scale="international_opinion" data-band="watching">' +
          '"It\'s an internal matter."</p>',
      ),
    ).toEqual({ band: 'watching', label: '"It\'s an internal matter."' });
  });

  it('3. trailing text OUTSIDE the element is not part of the label', () => {
    // politics_trust's lines end in the preposition that follows the word in
    // prose. The label is "neutral" — never "neutral about".
    expect(
      parseQdisplayHtml(
        '<span class="q-band" data-scale="politics_trust" data-band="neutral">neutral</span> about',
      ),
    ).toEqual({ band: 'neutral', label: 'neutral' });
  });

  it('4. no markup at all (relationships) — the word is slugged into a token', () => {
    expect(parseQdisplayHtml('warm')).toEqual({ band: 'warm', label: 'warm' });
    expect(parseQdisplayHtml('very friendly')).toEqual({
      band: 'very_friendly',
      label: 'very friendly',
    });
  });

  it('5. no matching range — the raw value comes back stringified', () => {
    expect(parseQdisplayHtml('99')).toEqual({ band: '99', label: '99' });
  });

  it('returns markup, entities and stray whitespace to plain text', () => {
    expect(
      parseQdisplayHtml(
        '<span class="q-band  extra" data-band="unset">  None &amp;\n  nothing  </span>',
      ),
    ).toEqual({ band: 'unset', label: 'None & nothing' });
  });

  it('never returns markup in the label', () => {
    const { label } = parseQdisplayHtml(
      '<span class="q-band" data-band="x"><em>very</em> low</span>',
    );
    expect(label).toBe('very low');
    expect(label).not.toMatch(/[<>]/);
  });

  it('an empty or non-string input is EMPTY, not a throw', () => {
    expect(parseQdisplayHtml('')).toEqual({ band: '', label: '' });
    expect(parseQdisplayHtml(undefined as unknown as string)).toEqual({ band: '', label: '' });
  });
});

describe('useBand — round trip through a booted adapter', () => {
  let raw: DendryAdapter;
  beforeAll(() => {
    raw = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    raw.beginGame([1, 2, 3, 4]);
  });

  let band: ReturnType<typeof useBand>['band'];
  beforeEach(() => {
    setActivePinia(createPinia());
    const g = useGameStore();
    g.initFromText(readFileSync(GAME, 'utf8'));
    g.newGame();
    band = useBand().band;
  });

  it('bands a value through the real qdisplay thresholds', () => {
    // Values chosen inside distinct ranges of social_dissent.qdisplay.dry.
    expect(band(10, 'social_dissent')).toEqual({ band: 'very_low', label: 'very low' });
    expect(band(45, 'social_dissent')).toEqual({ band: 'medium', label: 'medium' });
    expect(band(95, 'social_dissent')).toEqual({ band: 'very_high', label: 'very high' });
  });

  it('carries cat_spa_relations’ deliberate inversion through unchanged', () => {
    expect(band(10, 'cat_spa_relations').band).toBe('very_high');
    expect(band(90, 'cat_spa_relations').band).toBe('very_low');
  });

  it('strips politics_trust’s trailing preposition (live, not a fixture)', () => {
    expect(band(50, 'politics_trust')).toEqual({ band: 'neutral', label: 'neutral' });
  });

  it('handles international_opinion’s <p> element', () => {
    const b = band(1, 'international_opinion');
    expect(b.band).toBe('watching');
    expect(b.label).toBe('"It\'s an internal matter."');
  });

  it('reads control’s ladder rungs as tokens', () => {
    expect(band(0, 'control')).toEqual({ band: 'none', label: 'None' });
    expect(band(3, 'control')).toEqual({ band: 'complete', label: 'Complete' });
  });

  it('slugs an UN-banded qdisplay (relationships) into a usable token', () => {
    // relationships.qdisplay.dry carries no data-band at all — benches rows
    // still need a stamp token, so the word supplies it.
    expect(band(60, 'relationships')).toEqual({ band: 'warm', label: 'warm' });
    expect(band(2, 'relationships')).toEqual({ band: 'hostile', label: 'hostile' });
  });

  it('null / absent qdisplay id is EMPTY — the row contract’s "no stamp"', () => {
    expect(band(null, 'social_dissent')).toEqual({ band: '', label: '' });
    expect(band(undefined, 'social_dissent')).toEqual({ band: '', label: '' });
    expect(band(42, null)).toEqual({ band: '', label: '' });
    expect(band(42, undefined)).toEqual({ band: '', label: '' });
  });

  it('an unknown qdisplay id is EMPTY, not a throw (the engine asserts)', () => {
    // Guard: prove the engine really does throw, so this test is not vacuous.
    expect(() => raw.qdisplay(42, 'no_such_qdisplay')).toThrow();
    expect(band(42, 'no_such_qdisplay')).toEqual({ band: '', label: '' });
  });
});
