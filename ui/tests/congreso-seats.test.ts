import { it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { bindSceneRuntime } from './helpers/scene-runtime';
import official from './fixtures/congreso-official-seats.json';

const game = JSON.parse(readFileSync('../out/game.json', 'utf8'));
const actions = game.scenes['election_algorithm.esp_congreso'].onArrival
  .map((action: { $code: string }) => new Function('state', 'Q', 'G', action.$code));
const regions = ['catalunya', 'valencia', 'balears', 'navarra', 'euskadi', 'galicia', 'rest'];

// Execute the complete compiled election action, including map colouring and
// national totals. No allocator internals or library exports are exposed.
function election(votesByRegion: Record<string, Record<string, number>> = {}, year = 2016) {
  const q: Record<string, any> = { year, congreso_constituencies: regions, congreso_pop: {}, abstain_congreso_pv: 30 };
  for (const region of regions) {
    const votes = votesByRegion[region] || { pp: 100 };
    const total = Object.values(votes).reduce((sum, value) => sum + value, 0);
    q.congreso_pop[region] = total > 0 ? total / 0.7 : 100;
    q['congreso_parties_' + region] = Object.keys(votes);
    q['abstain_congreso_' + region + '_support'] = total > 0 ? 30 : 100;
    for (const [party, amount] of Object.entries(votes)) {
      q[party + '_congreso_' + region + '_support'] = amount / q.congreso_pop[region] * 100;
      q[party + '_congreso_s'] = 0;
      q[party + '_congreso_pv'] = 0;
    }
  }
  const stocks = Object.fromEntries(Object.entries(q).filter(([key]) => key.endsWith('_support')));
  const runtime = bindSceneRuntime(q);
  const log = vi.spyOn(console, 'log').mockImplementation(() => {});
  try {
    for (const action of actions) action.call(runtime.engine, runtime.state, runtime.Q, runtime.G);
  } finally { log.mockRestore(); }
  expect(Object.fromEntries(Object.entries(q).filter(([key]) => key.endsWith('_support')))).toEqual(stocks);
  return q;
}

function seats(q: Record<string, any>, region: string): Record<string, number> {
  return Object.fromEntries(q['congreso_parties_' + region].map((p: string) => [p, q[p + '_congreso_s_' + region]]));
}

it.each([2015, 2016])('allocates all 350 seats with the correct regional budgets in %s', year => {
  const votes = Object.fromEntries(regions.map(region => [region, { pp: 1200, psoe: 880, cs: 360, up: 1100, novel: 700 }]));
  const q = election(votes, year);
  expect(q.congreso_seats).toMatchObject(year === 2015 ? { valencia: 32, rest: 217 } : { valencia: 33, rest: 216 });
  let total = 0;
  for (const region of regions) {
    const allocations = Object.values(seats(q, region));
    expect(allocations.every(n => Number.isInteger(n) && n >= 0)).toBe(true);
    expect(allocations.reduce((sum, n) => sum + n, 0)).toBe(q.congreso_seats[region]);
    total += q.congreso_seats[region];
  }
  expect(total).toBe(350);
  for (const party of Object.keys(votes.rest)) {
    expect(q[party + '_congreso_s']).toBe(regions.reduce((sum, region) => sum + seats(q, region)[party], 0));
  }
});

it('preserves results when vote counts scale and gives successors the same geography', () => {
  const votes = { pp: 1200, psoe: 880, cs: 360, podemos: 1100, novel: 700 };
  const before = seats(election({ rest: votes }), 'rest');
  const scaled = Object.fromEntries(Object.entries(votes).map(([p, n]) => [p, n * 7]));
  expect(seats(election({ rest: scaled }), 'rest')).toEqual(before);
  const { podemos, ...others } = votes;
  const successor = seats(election({ rest: { ...others, up: podemos } }), 'rest');
  expect(successor.up).toBe(before.podemos);
  for (const party of Object.keys(others)) expect(successor[party]).toBe(before[party]);
});

it.each([2015, 2016] as const)('gives CC representation through concentration at %s reference support', year => {
  // Reference inputs check the home-province rule, not exact national targets.
  const votes = official[year].rest.votes;
  expect(election({ rest: votes }, year).cc_congreso_s_rest).toBe(1);
  expect(election({ rest: { ...votes, cc: 1 } }, year).cc_congreso_s_rest).toBe(0);
  expect(election({ rest: { ...votes, cc: 160000 } }, year).cc_congreso_s_rest).toBeGreaterThan(1);
});

it('gives tiny local parties no guaranteed seats and confines strong ones to their home district', () => {
  const others = { pp: 1000000, psoe: 800000 };
  const tiny = election({ rest: { ...others, cc: 1, fac: 1, prc: 1, te: 1 } });
  for (const party of ['cc', 'fac', 'prc', 'te']) expect(tiny[party + '_congreso_s_rest']).toBe(0);
  for (const [party, districtSeats] of Object.entries({ cc: 7, fac: 8, prc: 5, te: 3 })) {
    expect(election({ rest: { ...others, [party]: 100000000 } })[party + '_congreso_s_rest']).toBe(districtSeats);
  }
});

it('enforces the 3% candidacy threshold and handles an empty electorate', () => {
  // Other lists fall below 3%, isolating eligibility from district magnitude.
  for (const share of [29, 30]) {
    const votes = { small: share, ...Object.fromEntries(Array.from({ length: 40 }, (_, i) => ['other' + i, (1000 - share) / 40])) };
    expect(election({ balears: votes }).small_congreso_s_balears).toBe(share === 30 ? 8 : 0);
  }
  expect(seats(election({ balears: { pp: 0, psoe: 0 } }), 'balears')).toEqual({ pp: 0, psoe: 0 });
});

it('retains separate community-character map colouring for the rest-of-Spain electorate', () => {
  const q = election({ rest: { pp: 35, psoe: 34, cs: 10, up: 21 } });
  expect(q.congreso_prov_madrid_wp).toBe('pp');
  expect(q.congreso_prov_andalusia_wp).toBe('psoe');
});
