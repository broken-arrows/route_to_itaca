import { describe, expect, it } from 'vitest';
import { gameLib } from '../src/game-bindings';

const brief = () => gameLib.brief as Record<string, (q: Record<string, unknown>) => any[]>;

function pollQ(): Record<string, unknown> {
  const q: Record<string, unknown> = {
    parties: ['a', 'b', 'abstain'],
    player_party: 'a',
    parlament_demographics: ['buss', 'young'],
    parlament_seats: { barcelona: 5, tarragona: 2, lleida: 1, girona: 1 },
  };
  for (const province of ['barcelona', 'tarragona', 'lleida', 'girona']) {
    q[`parlament_${province}_buss_pop`] = 100;
    q[`parlament_${province}_young_pop`] = 50;
    q[`a_parlament_${province}_buss_support`] = province === 'girona' ? 20 : 60;
    q[`b_parlament_${province}_buss_support`] = province === 'girona' ? 70 : 30;
    q[`a_parlament_${province}_young_support`] = 40;
    q[`b_parlament_${province}_young_support`] = 50;
  }
  return q;
}

describe('G.brief poll derivations', () => {
  it('provinces weights support by demographic population and reports map metadata', () => {
    const rows = brief().provinces(pollQ());
    expect(rows.map((row) => row.id)).toEqual(['barcelona', 'tarragona', 'lleida', 'girona']);
    expect(rows.find((row) => row.id === 'barcelona')).toMatchObject({
      value: 'a', party: 'a', population: 150, seats: 5,
    });
    expect(rows.find((row) => row.id === 'girona')).toMatchObject({ value: 'b', party: 'b' });
  });

  it('crosstab returns one filed-report row per populated demographic and no abstain column', () => {
    const rows = brief().crosstab(pollQ());
    expect(rows).toHaveLength(8);
    const row = rows.find((entry) => entry.id === 'barcelona_buss');
    expect(row.label).toBe('Business');
    expect(row.value).toBe(100);
    expect(row.cells.map((cell: any) => [cell.id, cell.value])).toEqual([
      ['a', 60], ['b', 30],
    ]);
  });

  it('seatProjection applies the 3% threshold and D’Hondt for each province', () => {
    const rows = brief().seatProjection(pollQ());
    const barcelona = rows.filter((row) => row.province === 'barcelona');
    expect(barcelona.reduce((sum, row) => sum + row.value, 0)).toBe(5);
    expect(barcelona[0].value).toBeGreaterThan(barcelona[1].value);
    expect(barcelona.every((row) => row.share > 0 && row.share <= 1)).toBe(true);
  });

  it('all three views are pure', () => {
    const q = pollQ();
    const before = JSON.stringify(q);
    brief().provinces(q);
    brief().crosstab(q);
    brief().seatProjection(q);
    expect(JSON.stringify(q)).toBe(before);
  });
});
