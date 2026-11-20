import { describe, expect, it } from 'vitest';
import { gameLib } from '../src/game-bindings';
import official from './fixtures/congreso-district-magnitudes.json';

const total = (seats: Record<string, number>) => Object.values(seats).reduce((sum, n) => sum + n, 0);

describe('Congreso district magnitudes (LOREG art. 162)', () => {
  it.each([[2015, 12, '2015-12'], [2016, 6, '2016-06'], [2019, 4, '2019-04']] as const)(
    'reproduces the official %i/%i allocation for all 52 districts', (year, month, key) => {
      expect(gameLib.congresoApportionment(year, month)).toEqual(official[key]);
    });

  it('keeps November 2019 on the census of the April convocation', () => {
    expect(gameLib.congresoApportionment(2019, 11)).toEqual(official['2019-04']);
  });

  it('uses the census two years back for elections called before the December approval', () => {
    // A February 2016 poll is convoked in December 2015, on the 2014 census.
    expect(gameLib.congresoApportionment(2016, 2)).toEqual(official['2015-12']);
    expect(gameLib.congresoApportionment(2016, 3)).toEqual(official['2016-06']);
  });

  it('allocates 350 seats with the statutory minimums for any in-game date', () => {
    for (let year = 2012; year <= 2021; year++) {
      for (const month of [1, 6, 12]) {
        const seats = gameLib.congresoApportionment(year, month);
        expect(total(seats)).toBe(350);
        expect([seats.Ceuta, seats.Melilla]).toEqual([1, 1]);
        for (const [name, n] of Object.entries(seats)) if (name !== 'Ceuta' && name !== 'Melilla') expect(n).toBeGreaterThanOrEqual(2);
      }
    }
  });

  it('falls back to the 2016 allocation without a date and holds the last census afterwards', () => {
    expect(gameLib.congresoApportionment()).toEqual(official['2016-06']);
    expect(gameLib.congresoApportionment(2040, 6)).toEqual(gameLib.congresoApportionment(2022, 6));
  });
});
