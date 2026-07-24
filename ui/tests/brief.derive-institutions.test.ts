import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');
const brief = () => (gameLib as any).brief;

describe('G.brief institution derivations', () => {
  let Q: Record<string, unknown>;
  let a: DendryAdapter; // hoisted: the round-trip tests below need a.qdisplay
  beforeAll(() => {
    a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
    Q = a.qualities;
  });

  it('cabinet returns the nine portfolios in the designed order', () => {
    const rows = brief().cabinet(Q);
    expect(rows.map((r: any) => r.id)).toEqual([
      'president', 'vicepresident', 'economy', 'health', 'education',
      'interior', 'foreign', 'bnl', 'territory',
    ]);
    for (const r of rows) expect(typeof r.value).toBe('string');
  });

  it('control returns seven rungs carrying the raw 0..4 value, not a band', () => {
    const rows = brief().control(Q);
    expect(rows).toHaveLength(7);
    // Strengthened vs. the brief's sample: also pin the ids, in order, so a
    // table typo (wrong Q key, swapped row) fails here instead of only
    // showing up as an empty-string value nothing else checks.
    expect(rows.map((r: any) => r.id)).toEqual([
      'airports', 'railways', 'ports', 'borders', 'security',
      'communications', 'taxation',
    ]);
    for (const r of rows) {
      expect(typeof r.value).toBe('number');
      expect(r.value).toBeGreaterThanOrEqual(0);
      expect(r.value).toBeLessThanOrEqual(4);
      expect(r.valueDisplay).toBe('control');
    }
  });

  // Strengthened vs. the brief's sample: `typeof banded === 'string' &&
  // banded.length > 0` passes for ANY string a broken classifier might
  // return (e.g. String(value) itself), which would not prove brief.js is
  // naming a real classifying qdisplay. Assert the actual band words from
  // source/qdisplays/control.qdisplay.dry at both boundaries of the 0..4
  // range, through the real engine — not a second table in this test.
  // NB: qdisplay output is the compiled content HTML (verified by running
  // a.qdisplay(0,'control') directly — it returns
  // '<span class="q-band" data-scale="control" data-band="none">None</span>',
  // not a bare word), so these assert `toMatch`/`toContain` against that
  // markup rather than `toBe` a plain band word.
  it('the qdisplay named by a control row actually classifies its value', () => {
    const rows = brief().control(Q);
    const banded = a.qdisplay(rows[0].value, rows[0].valueDisplay);
    expect(typeof banded).toBe('string');
    expect(banded.length).toBeGreaterThan(0);
    expect(a.qdisplay(0, 'control')).toContain('None');
    expect(a.qdisplay(1, 'control')).toContain('Limited');
    expect(a.qdisplay(2, 'control')).toContain('Partial');
    expect(a.qdisplay(3, 'control')).toContain('Complete');
    expect(a.qdisplay(4, 'control')).toContain('Disputed');
  });

  it('chancelleries returns the four capitals with flags and a qdisplay id, in designed order', () => {
    const rows = brief().chancelleries(Q);
    expect(rows).toHaveLength(4);
    // Strengthened vs. the brief's sample: pin id -> flag pairing explicitly,
    // so a shuffled table (e.g. russia's flag on china's row) fails instead
    // of merely matching the generic `img/flags/*.svg` shape.
    expect(rows.map((r: any) => [r.id, r.flag])).toEqual([
      ['eu', 'img/flags/eu.svg'],
      ['usa', 'img/flags/usa.svg'],
      ['russia', 'img/flags/russia.svg'],
      ['china', 'img/flags/prc.svg'],
    ]);
    for (const r of rows) {
      expect(r.flag).toMatch(/^img\/flags\/.+\.svg$/);
      expect(r.valueDisplay).toBe('international_opinion');
      expect(typeof r.value).toBe('number');
    }
  });

  // Proves the round trip end to end, same pattern as the control test above:
  // international_opinion.qdisplay.dry bands (..-1)/(0..0)/(1..1)/(2..0)/(3..)
  // each render distinct non-empty prose, so a row naming the wrong qdisplay
  // id (or the wrong Q key entirely) would show up as either a thrown error
  // or identical text across different inputs.
  it('the qdisplay named by a chancelleries row actually classifies its value', () => {
    const low = a.qdisplay(-1, 'international_opinion');
    const high = a.qdisplay(3, 'international_opinion');
    expect(typeof low).toBe('string');
    expect(typeof high).toBe('string');
    expect(low.length).toBeGreaterThan(0);
    expect(high.length).toBeGreaterThan(0);
    expect(low).not.toBe(high);
  });

  it('cabinet rows are names, not scales — no qdisplay', () => {
    const rows = brief().cabinet(Q);
    // Strengthened vs. the brief's sample: a `for (const r of rows)` loop
    // over an unimplemented stub's `[]` never executes its body, so this
    // assertion would pass with zero checks made. Pin non-emptiness first.
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) expect(r.valueDisplay).toBeNull();
  });

  it('writes nothing to Q', () => {
    const before = JSON.stringify(Q);
    brief().cabinet(Q); brief().control(Q); brief().chancelleries(Q);
    expect(JSON.stringify(Q)).toBe(before);
  });
});
