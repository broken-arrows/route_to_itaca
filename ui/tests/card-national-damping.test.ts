import { describe, expect, it } from 'vitest';
import { createRequire } from 'node:module';

const { cardSupportTransfer } = createRequire(import.meta.url)('../../source/lib/cat_engine.js');
type Q = Record<string, any>;

function electorate(): Q {
  const q: Q = {
    parties: ['erc', 'cup', 'jxsi', 'psc'],
    parlament_constituencies: ['barcelona', 'girona'],
    parlament_demographics: ['urban', 'rural'],
  };
  for (const province of q.parlament_constituencies) {
    for (const demo of q.parlament_demographics) {
      q[`parlament_${province}_${demo}_pop`] = province === 'barcelona' ? 900 : 100;
      q[`erc_parlament_${province}_${demo}_support`] = province === 'barcelona' ? 20 : 70;
      q[`cup_parlament_${province}_${demo}_support`] = 0;
      q[`jxsi_parlament_${province}_${demo}_support`] = 0;
      q[`psc_parlament_${province}_${demo}_support`] = province === 'barcelona' ? 60 : 10;
      q[`abstain_parlament_${province}_${demo}_support`] = 20;
    }
  }
  return q;
}

function transfer(q: Q, overrides: Q = {}) {
  return cardSupportTransfer(q, {
    contest: 'parlament', to: 'erc', from: 'psc', amount: 2,
    constituencies: 'all', demographics: 'all', maxDonorFraction: 0.5,
    ...overrides,
  });
}

describe('Parlament card national diminishing returns', () => {
  it('lets rural Girona exceed 65% without local damping when national support is modest', () => {
    const q = electorate();
    transfer(q, { constituencies: 'girona', demographics: 'rural' });
    // National valid share = (900*20 + 100*70) / (1000*80) = 31.25%.
    expect(q.erc_parlament_girona_rural_support).toBe(72);
    expect(q.psc_parlament_girona_rural_support).toBe(8);
    expect(q.erc_parlament_barcelona_urban_support).toBe(20);
  });

  it('weights population, excludes abstention, and uses one snapshot for every selected cell', () => {
    const q = electorate();
    for (const demo of q.parlament_demographics) {
      q[`erc_parlament_barcelona_${demo}_support`] = 30;
      q[`psc_parlament_barcelona_${demo}_support`] = 40;
      q[`abstain_parlament_barcelona_${demo}_support`] = 30;
      q[`erc_parlament_girona_${demo}_support`] = 80;
      q[`abstain_parlament_girona_${demo}_support`] = 10;
    }
    const nationalShare = (900 * 30 + 100 * 80) / (900 * 70 + 100 * 90) * 100;
    const gain = 2 / (1 + ((nationalShare - 35) / 15) ** 2);
    const before = structuredClone(q);
    expect(transfer(q)).toBeCloseTo(gain * 4, 12);
    for (const province of q.parlament_constituencies) {
      for (const demo of q.parlament_demographics) {
        const key = `erc_parlament_${province}_${demo}_support`;
        expect(q[key] - before[key]).toBeCloseTo(gain, 12);
      }
    }
  });

  it('counts a live coalition once and exempts transfers within the independence bloc', () => {
    const q = electorate();
    q.erc_in_jxsi = true;
    q.parties.push('jxsi'); // Duplicate registry entries cannot multiply its votes.
    for (const province of q.parlament_constituencies) {
      for (const demo of q.parlament_demographics) {
        q[`erc_parlament_${province}_${demo}_support`] = 0;
        q[`jxsi_parlament_${province}_${demo}_support`] = 40;
        q[`psc_parlament_${province}_${demo}_support`] = 40;
      }
    }
    expect(transfer(q)).toBeCloseTo(4, 12); // 50% national share: half strength.
    expect(transfer(q, { to: 'cup', from: 'erc' })).toBe(8);
    expect(q.cup_parlament_girona_rural_support).toBe(2);
  });

  it('rejects invalid national population before changing any selected cell', () => {
    const q = electorate();
    q.parlament_girona_rural_pop = NaN;
    const before = structuredClone(q);
    expect(() => transfer(q, { constituencies: 'barcelona' })).toThrow(/population/);
    expect(q).toEqual(before);
  });
});
