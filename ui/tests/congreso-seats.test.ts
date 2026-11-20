import { it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { bindSceneRuntime } from './helpers/scene-runtime';
import { gameLib } from '../src/game-bindings';
import official from './fixtures/congreso-official-seats.json';

const game = JSON.parse(readFileSync('../out/game.json', 'utf8'));
const actions = game.scenes['election_algorithm.esp_congreso'].onArrival
  .map((action: { $code: string }) => new Function('state', 'Q', 'G', action.$code));
const regions = ['catalunya', 'valencia', 'balears', 'navarra', 'euskadi', 'galicia', 'rest'];

// Execute the complete compiled election action, including map colouring and
// national totals. No allocator internals or library exports are exposed.
function election(votesByRegion: Record<string, Record<string, number>> = {}, year = 2016, previous: Record<string, any> = {}) {
  const q: Record<string, any> = { ...previous, year, congreso_constituencies: regions, congreso_pop: {}, abstain_congreso_pv: 30 };
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

// Regional budgets follow the census behind each convocation (2019: Barcelona
// gains a seat, Valencia loses one; Madrid gains Asturias's).
const regionalBudget = {
  2015: { catalunya: 47, valencia: 32, rest: 217 },
  2016: { catalunya: 47, valencia: 33, rest: 216 },
  2019: { catalunya: 48, valencia: 32, rest: 216 },
} as const;

it.each([2015, 2016, 2019] as const)('allocates all 350 seats with the correct regional budgets in %s', year => {
  const votes = Object.fromEntries(regions.map(region => [region, { pp: 1200, psoe: 880, cs: 360, up: 1100, novel: 700 }]));
  const q = election(votes, year);
  expect(q.congreso_seats).toMatchObject(regionalBudget[year]);
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

it('preserves results when vote counts scale and gives successors their family geography', () => {
  const votes = { pp: 1200, psoe: 880, cs: 360, podemos: 1100, novel: 700 };
  const before = seats(election({ rest: votes }), 'rest');
  const scaled = Object.fromEntries(Object.entries(votes).map(([p, n]) => [p, n * 7]));
  expect(seats(election({ rest: scaled }), 'rest')).toEqual(before);
  const { podemos: carried, ...others } = votes;
  const renamed = seats(election({ rest: { ...others, mpais: carried } }), 'rest');
  expect(renamed.mpais).toBe(before.podemos);
  for (const party of Object.keys(others)) expect(renamed[party]).toBe(before[party]);
});

it('gives Vox its own 2019 geography instead of PP\'s', () => {
  // At equal national votes, Vox leads its strongholds and PP the interior.
  const q = election({ rest: { pp: 100, vox: 100 } }, 2019);
  for (const id of ['murcia', 'ceuta']) expect(q['congreso_prov_' + id + '_wp']).toBe('vox');
  for (const id of ['castile_and_leon', 'la_rioja', 'canary_islands']) expect(q['congreso_prov_' + id + '_wp']).toBe('pp');
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

it('reproduces the Andalusian switch and colours the map from the provincial votes', () => {
  expect(election({ rest: official[2015].rest.votes }, 2015).congreso_prov_andalusia_wp).toBe('psoe');
  expect(election({ rest: official[2016].rest.votes }).congreso_prov_andalusia_wp).toBe('pp');
  // A national tie splits by profile: PSOE leads Andalucía, PP leads Madrid.
  const q = election({ rest: { pp: 100, psoe: 100 } });
  expect(q.congreso_prov_andalusia_wp).toBe('psoe');
  expect(q.congreso_prov_madrid_wp).toBe('pp');
  // Every pooled community is painted, including the local-list homes.
  const local = election({ rest: { pp: 100, psoe: 90, prc: 1000 } });
  expect(local.congreso_prov_cantabria_wp).toBe('prc');
  expect(local.congreso_prov_aragon_wp).toBe('pp');
});

// Province seat budgets per map community (2016+); León elects 5 in 2015.
const communityBudget: Record<string, number> = {
  madrid: 36, andalusia: 61, extremadura: 10, castile_la_mancha: 21, castile_and_leon: 31,
  murcia: 10, la_rioja: 4, aragon: 13, asturias: 8, cantabria: 5, canary_islands: 15, ceuta: 1, melilla: 1,
};
const sum = (values: Record<string, number>) => Object.values(values).reduce((total, n) => total + n, 0);

it.each([2015, 2016] as const)('records per-community provincial seats and votes for the map in %s', year => {
  const q = election({ rest: official[year].rest.votes }, year, {
    congreso_rest_communities: { madrid: { total: 99, seats: { pp: 99 }, votes: { pp: 1 } }, nowhere: { total: 3 } },
  });
  const communities = q.congreso_rest_communities;
  expect(JSON.parse(JSON.stringify(communities))).toEqual(communities);
  expect(Object.keys(communities).sort()).toEqual(Object.keys(communityBudget).sort());
  for (const [id, community] of Object.entries<any>(communities)) {
    expect(community.total).toBe(communityBudget[id] + (year === 2015 && id === 'castile_and_leon' ? 1 : 0));
    expect(sum(community.seats)).toBe(community.total);
  }
  expect(sum(Object.fromEntries(Object.entries<any>(communities).map(([id, c]) => [id, c.total])))).toBe(q.congreso_seats.rest);
  for (const party of q.congreso_parties_rest) {
    const bySeats = Object.values<any>(communities).reduce((total, c) => total + c.seats[party], 0);
    expect(bySeats).toBe(q[party + '_congreso_s_rest']);
    const byVotes = Object.values<any>(communities).reduce((total, c) => total + c.votes[party], 0);
    expect(byVotes).toBeCloseTo((official[year].rest.votes as Record<string, number>)[party], 3);
  }
  // The shared map model opens one panel per community from this record.
  const model = gameLib.congresoResultsMap(q);
  expect(model.panels.rest).toBeUndefined();
  for (const id of Object.keys(communityBudget)) {
    expect(model.regions.find(region => region.id === id)?.panel).toBe(id);
    const panel = model.panels[id];
    expect(panel.seats).toBe(communities[id].total);
    expect(panel.rows.reduce((total, row) => total + row.seats, 0)).toBe(panel.seats);
    expect(panel.rows[0].party).toBe(q['congreso_prov_' + id + '_wp']);
  }
});
