import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');
const brief = () => (gameLib as any).brief;

describe('G.brief trend derivations', () => {
  let Q: Record<string, unknown>;
  let a: DendryAdapter; // hoisted: the round-trip tests below need a.qdisplay
  beforeAll(() => {
    a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
    Q = a.qualities;
  });

  it('factions follows the player party', () => {
    const rows = brief().factions({ ...Q, player_party: 'erc' });
    expect(rows.map((r: any) => r.id)).toEqual(['left', 'core', 'pragmatic', 'civic']);
    expect(brief().factions({ ...Q, player_party: 'cup' })).toHaveLength(3);
    expect(brief().factions({ ...Q, player_party: 'ciu' })).toHaveLength(0);
  });

  // Strengthened vs. the brief's sample: the brief only checks ids/length,
  // never that strength/dissent actually read the RIGHT stem's Q keys. The
  // real fixture leaves erc_*_strength/_dissent undefined at a bare
  // beginGame() (they're only set inside root.scene.dry's ERC party-choice
  // branch, same story as `_relations` in Task 3's LEARNINGS entry), so
  // force distinct per-faction values and confirm each lands on its own row
  // rather than a shared/shuffled one.
  it('each faction row reads its OWN stem Q keys for strength and dissent', () => {
    const Qerc = Object.assign({}, Q, {
      player_party: 'erc',
      erc_left_strength: 11, erc_left_dissent: 22,
      erc_core_strength: 33, erc_core_dissent: 44,
    });
    const rows = brief().factions(Qerc);
    const left = rows.find((r: any) => r.id === 'left');
    const core = rows.find((r: any) => r.id === 'core');
    expect(left.strength).toBe(11);
    expect(left.dissent).toBe(22);
    expect(core.strength).toBe(33);
    expect(core.dissent).toBe(44);
    for (const r of rows) {
      expect(r.strengthDisplay).toBe('dissent');
      expect(r.dissentDisplay).toBe('dissent');
    }
  });

  // Proves the round trip through the real engine and the real
  // source/qdisplays/dissent.qdisplay.dry bands, rather than trusting that
  // naming 'dissent' as the classifier is correct by construction.
  it('the dissent qdisplay actually classifies a faction strength/dissent value', () => {
    const low = a.qdisplay(2, 'dissent');
    const high = a.qdisplay(60, 'dissent');
    expect(typeof low).toBe('string');
    expect(typeof high).toBe('string');
    expect(low).not.toBe(high);
    expect(low).toContain('very low');
    expect(high).toContain('very high');
  });

  it('street returns three bars with share in 0..1 and a qdisplay per row', () => {
    const rows = brief().street(Q);
    expect(rows).toHaveLength(3);
    for (const r of rows) {
      expect(r.share).toBeGreaterThanOrEqual(0);
      expect(r.share).toBeLessThanOrEqual(1);
      expect(typeof r.valueDisplay).toBe('string');
    }
    // Each street row names its OWN qdisplay — a shared hand-rolled ladder
    // would make this assertion pass while losing every distinction.
    expect(rows.map((r: any) => r.valueDisplay)).toEqual([
      'social_dissent', 'independence_movement', 'politics_trust',
    ]);
  });

  // Proves each street row's named qdisplay really classifies ITS OWN scale
  // distinctly, through the real engine — same pattern as Task 4's
  // control/chancelleries round trip. This is what actually rules out a
  // shared ladder: three differently-named ids that all secretly resolved
  // to the same classifier would still pass every assertion above it.
  it('each street qdisplay classifies its own scale distinctly', () => {
    expect(a.qdisplay(5, 'social_dissent')).toContain('very low');
    expect(a.qdisplay(90, 'social_dissent')).toContain('very high');
    expect(a.qdisplay(5, 'independence_movement')).toContain('very low');
    expect(a.qdisplay(90, 'independence_movement')).toContain('very high');
    expect(a.qdisplay(5, 'politics_trust')).toContain('completely distrustful');
    expect(a.qdisplay(90, 'politics_trust')).toContain('blindly trusting');
  });

  it('no derivation invents a band token — classification is the qdisplay layer', () => {
    // Guards the ROW CONTRACT rule against regression: brief.js must never
    // ship its own threshold table beside source/qdisplays/*.dry.
    const all = [...brief().street(Q), ...brief().control(Q), ...brief().chancelleries(Q)];
    for (const r of all) expect(r).not.toHaveProperty('band');
  });

  it('trails returns three metrics with a series and a direction', () => {
    const rows = brief().trails(Q);
    expect(rows.map((r: any) => r.id)).toEqual(['gdp', 'unemployment', 'surplus']);
    for (const r of rows) {
      expect(Array.isArray(r.series)).toBe(true);
      expect(['up', 'down', 'flat']).toContain(r.dir);
      expect(r.value).toBe(Math.round(r.value * 10) / 10);
    }
  });

  it('direction comes from the series, never from Q.*_change HTML', () => {
    const rising = brief().trails({
      ...Q,
      gdp_growth: 2,
      economic_records: [{ gdp_growth: 1 }, { gdp_growth: 2 }],
    });
    const gdp = rising.find((r: any) => r.id === 'gdp');
    expect(gdp.dir).toBe('up');
    expect(gdp.good).toBe(true);
  });

  it('unemployment rising is BAD', () => {
    const rows = brief().trails({
      ...Q, economic_records: [{ unemployment: 10 }, { unemployment: 20 }],
    });
    const u = rows.find((r: any) => r.id === 'unemployment');
    expect(u.dir).toBe('up');
    expect(u.good).toBe(false);
  });

  it('a flat/start-of-run metric is coloured by its current scalar, not automatically good', () => {
    const rows = brief().trails({
      ...Q,
      gdp_growth: -3.1,
      unemployment: 22.5,
      generalitat_surplus: -2.3,
      economic_records: [],
    });
    expect(rows.map((r: any) => [r.id, r.dir, r.good])).toEqual([
      ['gdp', 'flat', false],
      ['unemployment', 'flat', false],
      ['surplus', 'flat', false],
    ]);
    expect(rows.map((r: any) => r.series)).toEqual([[-3.1], [22.5], [-2.3]]);
  });

  it('appends the live scalar when recorded history is one turn behind', () => {
    const rows = brief().trails({
      ...Q,
      gdp_growth: -3.1,
      economic_records: [{ gdp_growth: -2.9 }, { gdp_growth: -3.0 }],
    });
    expect(rows.find((r: any) => r.id === 'gdp').series).toEqual([-2.9, -3.0, -3.1]);
  });

  it('writes nothing to Q', () => {
    const before = JSON.stringify(Q);
    brief().factions(Q); brief().street(Q); brief().trails(Q);
    expect(JSON.stringify(Q)).toBe(before);
  });
});

// `standing` restores the player's own Parlament bar that the old OVERVIEW
// carried as four mutually-exclusive hand-styled divs. Every scenario below is
// FORCED via a spread: player_party and the coalition flags are undefined after
// a bare beginGame(), so a test leaning on boot state would pass vacuously.
describe('G.brief standing derivation', () => {
  let Q: Record<string, unknown>;
  beforeAll(() => {
    const a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
    Q = a.qualities;
  });

  it('follows the player party when in no coalition', () => {
    const [row] = brief().standing({ ...Q, player_party: 'erc' });
    expect(row.id).toBe('erc');
    expect(row.value).toBe(Q.erc_parlament_s);
    expect(row.total).toBe(Q.parlament_size);
  });

  it('covers a CiU player, who had no bar at all in the old content', () => {
    // root.scene.dry assigns player_party = "ciu", but the old OVERVIEW's four
    // hand-styled branches only ever matched erc/cup, so CiU rendered nothing.
    const [row] = brief().standing({
      ...Q, player_party: 'ciu', ciu_parlament_s: 50, parlament_size: 135,
    });
    expect(row.id).toBe('ciu');
    expect(row.label).toBe('CiU');
    expect(row.value).toBe(50);
    expect(row.share).toBeCloseTo(50 / 135, 10);
  });

  it('reports the COALITION, not the party, when the player sits inside one', () => {
    const jxsi = brief().standing({ ...Q, player_party: 'erc', erc_in_jxsi: true });
    expect(jxsi[0].id).toBe('jxsi');
    expect(jxsi[0].value).toBe(Q.jxsi_parlament_s);

    const jxcat = brief().standing({ ...Q, player_party: 'cup', cup_in_jxcat: true });
    expect(jxcat[0].id).toBe('jxcat');
    expect(jxcat[0].value).toBe(Q.jxcat_parlament_s);
  });

  it('returns exactly ONE row for a CUP player inside JxSí', () => {
    // The old content matched both the JxSí branch and an unguarded
    // `player_party == "cup"` branch, rendering two contradictory bars.
    const rows = brief().standing({ ...Q, player_party: 'cup', cup_in_jxsi: true });
    expect(rows).toHaveLength(1);
    expect(rows[0].id).toBe('jxsi');
  });

  it('share is value/total, not a raw seat count', () => {
    const [row] = brief().standing({
      ...Q, player_party: 'erc', erc_parlament_s: 27, parlament_size: 135,
    });
    expect(row.share).toBeCloseTo(0.2, 10);
    expect(row.value).toBe(27);
  });

  it('returns [] when the player has no party', () => {
    expect(brief().standing({ ...Q, player_party: undefined })).toEqual([]);
  });

  it('writes nothing to Q', () => {
    const before = JSON.stringify(Q);
    brief().standing({ ...Q, player_party: 'erc' });
    expect(JSON.stringify(Q)).toBe(before);
  });
});
