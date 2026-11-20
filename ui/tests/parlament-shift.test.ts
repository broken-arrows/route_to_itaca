import { describe, expect, it } from 'vitest';
import { createRequire } from 'node:module';

// Scripted Parlament shifts (events, coalition outcomes, laws) used to be flat
// `Q[key] -= x` writes. A donor holding less than x went negative; post_event's
// clamp then zeroed it, which created support from nothing. `parlamentShift`
// is the conserving replacement: the donor can give at most what it holds.
const { parlamentShift } = createRequire(import.meta.url)('../../source/lib/cat_engine.js');
type Q = Record<string, any>;

function electorate(): Q {
  const q: Q = {
    parties: ['erc', 'ciu', 'cup', 'psc'],
    parlament_constituencies: ['barcelona', 'tarragona'],
    parlament_demographics: ['young', 'rural'],
  };
  for (const c of q.parlament_constituencies)
    for (const d of q.parlament_demographics) {
      q[`erc_parlament_${c}_${d}_support`] = 20;
      q[`ciu_parlament_${c}_${d}_support`] = 30;
      q[`cup_parlament_${c}_${d}_support`] = 5;
      q[`psc_parlament_${c}_${d}_support`] = 25;
      q[`abstain_parlament_${c}_${d}_support`] = 20;
    }
  return q;
}
const key = (p: string, c: string, d: string) => `${p}_parlament_${c}_${d}_support`;
const cellTotal = (q: Q, c: string, d: string) =>
  [...q.parties, 'abstain'].reduce((n: number, p: string) => n + q[key(p, c, d)], 0);

describe('parlamentShift', () => {
  it('moves the requested amounts in every selected cell by default', () => {
    const q = electorate();
    expect(parlamentShift(q, { from: 'erc', to: { ciu: 1, cup: 0.25 } })).toBeCloseTo(5);
    for (const c of q.parlament_constituencies)
      for (const d of q.parlament_demographics) {
        expect(q[key('erc', c, d)]).toBeCloseTo(18.75);
        expect(q[key('ciu', c, d)]).toBeCloseTo(31);
        expect(q[key('cup', c, d)]).toBeCloseTo(5.25);
        expect(cellTotal(q, c, d)).toBeCloseTo(100);
      }
  });

  it('touches only the selected constituencies and demographics', () => {
    const q = electorate();
    parlamentShift(q, { from: 'psc', to: { erc: 2 }, constituencies: ['tarragona'], demographics: ['rural'] });
    expect(q[key('erc', 'tarragona', 'rural')]).toBe(22);
    expect(q[key('psc', 'tarragona', 'rural')]).toBe(23);
    expect(q[key('erc', 'tarragona', 'young')]).toBe(20);
    expect(q[key('erc', 'barcelona', 'rural')]).toBe(20);
  });

  it('never drives the donor below zero: a short donor scales every recipient alike', () => {
    const q = electorate();
    q[key('ciu', 'tarragona', 'young')] = 0.6;
    q[key('abstain', 'tarragona', 'young')] = 49.4;
    parlamentShift(q, { from: 'ciu', to: { erc: 0.5, cup: 0.1, abstain: 0.4 }, constituencies: ['tarragona'], demographics: ['young'] });
    expect(q[key('ciu', 'tarragona', 'young')]).toBe(0);
    expect(q[key('erc', 'tarragona', 'young')]).toBeCloseTo(20.3);
    expect(q[key('cup', 'tarragona', 'young')]).toBeCloseTo(5.06);
    expect(q[key('abstain', 'tarragona', 'young')]).toBeCloseTo(49.64);
    expect(cellTotal(q, 'tarragona', 'young')).toBeCloseTo(100);
  });

  it('keeps the share of a recipient that is not on the ballot with the donor', () => {
    const q = electorate();
    parlamentShift(q, { from: 'erc', to: { ciu: 2, cs: 1 }, constituencies: ['barcelona'], demographics: ['young'] });
    expect(q[key('erc', 'barcelona', 'young')]).toBe(18);
    expect(q[key('ciu', 'barcelona', 'young')]).toBe(32);
    expect(q[key('cs', 'barcelona', 'young')]).toBeUndefined();
  });

  it('is a no-op for a donor that is not on the ballot, or for null/self recipients', () => {
    const q = electorate();
    const before = JSON.stringify(q);
    expect(parlamentShift(q, { from: 'undefined', to: { erc: 1 } })).toBe(0);
    expect(parlamentShift(q, { from: 'erc', to: { erc: 1 } })).toBe(0);
    expect(parlamentShift(q, { from: 'erc', to: { [String(null)]: 1 } })).toBe(0);
    expect(parlamentShift(q, { from: 'erc', to: { ciu: 0 } })).toBe(0);
    expect(JSON.stringify(q)).toBe(before);
  });

  it('rejects corrupt support or amounts without a partial effect', () => {
    const q = electorate();
    q[key('ciu', 'tarragona', 'rural')] = Number.NaN;
    const before = q[key('erc', 'barcelona', 'young')];
    expect(() => parlamentShift(q, { from: 'erc', to: { ciu: 1 } })).toThrow(/ciu_parlament_tarragona_rural_support/);
    expect(q[key('erc', 'barcelona', 'young')]).toBe(before);
    expect(() => parlamentShift(electorate(), { from: 'erc', to: { ciu: -1 } })).toThrow(/amount/);
    expect(() => parlamentShift(electorate(), { from: 'erc', to: { ciu: 1 }, demographics: ['elderly'] })).toThrow(/elderly/);
  });
});
