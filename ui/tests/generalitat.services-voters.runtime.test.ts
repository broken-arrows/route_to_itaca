import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');
const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
afterEach(() => {
  const actionErrors = errors.mock.calls.filter(([message]) => String(message).includes('Scene action error'));
  errors.mockClear();
  expect(actionErrors).toEqual([]);
});

function setup(overrides: Record<string, any> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  for (const party of q.parties) {
    q[`${party}_parlament_s`] = 0;
    q[`${party}_relations`] = 50;
  }
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    erc_in_jxsi: false, erc_in_jxcat: false, cup_in_jxsi: false, cup_in_jxcat: false,
    cat_coalition: ['erc'], cat_coalition_support: [], cat_coalition_abstain: [],
    president_party: 'erc', economy_minister_party: 'erc', health_minister_party: 'erc',
    education_minister_party: 'erc', vicepresident_party: 'erc', bnl_minister_party: 'erc',
    erc_parlament_s: 80, ppc_parlament_s: 55, parlament_s_majority: 68,
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    year: 2014, month: 1, cat_caretaker_gov: false, government_budget: 30, party_resources: 30,
    healthcare_spending: 0, education_spending: 0, public_salary_level: 0, social_spending_cap: 3,
    healthcare_prepared_level: null, education_prepared_level: null, public_salary_prepared_level: null,
    economy_cuts: 0, economy_atc_level: 0, economy_asset_sales: 0, vp_digital_level: 0,
    healthcare_universal_level: 0, healthcare_sexual_level: 0,
    active_mods: {}, mod_log: [], coalition_policy_commitments: [],
    education_policy_main: null, education_policy_secondary: null, education_policy_next_revision_month: null,
    ...overrides,
  });
  // Give each active alternative enough support to distinguish clipping from authored routes.
  // Inactive historical successors remain zero, just as they do in the live game.
  for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
    for (const party of q.parties) q[`${party}_parlament_${province}_${demo}_support`] = 0;
    for (const [party, support] of Object.entries({ erc: 12, cup: 5, ciu: 15, psc: 18, icv: 15, ppc: 8, cs: 7, abstain: 20 }))
      q[`${party}_parlament_${province}_${demo}_support`] = support;
  }
  return { adapter, q };
}

function support(q: Record<string, any>) {
  return Object.fromEntries(Object.entries(q).filter(([key, value]) => key.endsWith('_support') && typeof value === 'number')) as Record<string, number>;
}

function checkConservation(q: Record<string, any>, previous: Record<string, number>) {
  for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
    const parties = [...q.parties, 'abstain'];
    const suffix = `_parlament_${province}_${demo}_support`;
    expect(parties.reduce((n: number, p: string) => n + q[p + suffix], 0)).toBeCloseTo(parties.reduce((n: number, p: string) => n + previous[p + suffix], 0), 9);
  }
  for (const constituency of q.congreso_constituencies) {
    const parties = [...q[`congreso_parties_${constituency}`], 'abstain'];
    const suffix = `_congreso_${constituency}_support`;
    expect(parties.reduce((n: number, p: string) => n + q[p + suffix], 0)).toBeCloseTo(parties.reduce((n: number, p: string) => n + previous[p + suffix], 0), 9);
  }
  expect(q.parties_bcn.reduce((n: number, p: string) => n + q[`${p}_local_barcelona_support`], 0)).toBeCloseTo(q.parties_bcn.reduce((n: number, p: string) => n + previous[`${p}_local_barcelona_support`], 0), 9);
  for (const value of Object.values(support(q))) expect(value).toBeGreaterThanOrEqual(0);
}

describe('Delivered ministries target particular voters', () => {
  it('ERC health expansion takes from PSC, federal left and abstention, and reaches rural provinces more strongly', () => {
    const { adapter, q } = setup();
    const before = support(q);
    adapter.goToScene('generalitat_health_card.approve_raise');
    expect(q.healthcare_spending).toBe(1);
    for (const donor of ['psc', 'icv', 'abstain']) expect(q[`${donor}_parlament_barcelona_retired_support`]).toBeLessThan(before[`${donor}_parlament_barcelona_retired_support`]);
    for (const untouched of ['ciu', 'cs', 'ppc', 'cup']) expect(q[`${untouched}_parlament_barcelona_retired_support`]).toBe(before[`${untouched}_parlament_barcelona_retired_support`]);
    const gain = (province: string) => q[`erc_parlament_${province}_retired_support`] - before[`erc_parlament_${province}_retired_support`];
    expect(gain('girona')).toBeGreaterThan(gain('barcelona'));
    expect(q.erc_parlament_barcelona_buss_support).toBe(before.erc_parlament_barcelona_buss_support);
    checkConservation(q, before);
  });

  it('CUP healthcare draws from ERC and the federal left while PSC remains unchanged', () => {
    const { adapter, q } = setup({ player_party: 'cup', health_minister_party: 'cup', cat_coalition: ['cup'] });
    const before = support(q);
    adapter.goToScene('generalitat_health_card.approve_raise');
    expect(q.cup_parlament_barcelona_ind_support).toBeGreaterThan(before.cup_parlament_barcelona_ind_support);
    expect(q.erc_parlament_barcelona_ind_support).toBeLessThan(before.erc_parlament_barcelona_ind_support);
    expect(q.psc_parlament_barcelona_ind_support).toBe(before.psc_parlament_barcelona_ind_support);
    // CUP cannot collect rewards on an absent national ballot.
    for (const [key, value] of Object.entries(before).filter(([key]) => key.includes('_congreso_'))) expect(q[key]).toBe(value);
    checkConservation(q, before);
  });

  it('allocates an ERC-led service response 80% to ERC and 20% to a distinct CUP cabinet partner', () => {
    const { adapter, q } = setup({ cat_coalition: ['erc', 'cup', 'cup'] });
    const before = support(q);
    adapter.goToScene('generalitat_education_card.approve_raise');
    const erc = q.erc_parlament_barcelona_young_support - before.erc_parlament_barcelona_young_support;
    const cup = q.cup_parlament_barcelona_young_support - before.cup_parlament_barcelona_young_support;
    expect(erc / cup).toBeCloseTo(4, 1);
    checkConservation(q, before);
  });

  it.each(['jxsi', 'jxcat'])('credits one %s carrier without reviving member or successor ballots', (list) => {
    const { adapter, q } = setup({ player_in_jxsi: list === 'jxsi', player_in_jxcat: list === 'jxcat', erc_in_jxsi: list === 'jxsi', erc_in_jxcat: list === 'jxcat', education_minister_party: list, cat_coalition: [list, 'erc'] });
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
      q[`${list}_parlament_${province}_${demo}_support`] = q[`erc_parlament_${province}_${demo}_support`] + q[`ciu_parlament_${province}_${demo}_support`];
      q[`erc_parlament_${province}_${demo}_support`] = 0;
      q[`ciu_parlament_${province}_${demo}_support`] = 0;
    }
    const before = support(q);
    adapter.goToScene('generalitat_education_card.approve_raise');
    expect(q[`${list}_parlament_barcelona_young_support`]).toBeGreaterThan(before[`${list}_parlament_barcelona_young_support`]);
    for (const dormant of ['erc', 'ciu', 'cdc', 'pdcat', 'junts']) expect(q[`${dormant}_parlament_barcelona_young_support`]).toBe(0);
    checkConservation(q, before);
  });

  it.each([['health', 'healthcare'], ['education', 'education'], ['economy', 'public_salary']])('prepared %s spending earns zero direct votes and preserves initiative ownership', (card, policy) => {
    const { adapter, q } = setup({ social_spending_cap: 0 });
    const before = support(q);
    adapter.goToScene(`generalitat_${card}_card.${card === 'economy' ? 'raise_pay' : 'approve_raise'}`);
    expect(support(q)).toEqual(before);
    expect(q[`${policy}_prepared_level`]).toBe(1);
    const id = policy === 'public_salary' ? 'public_salary_proposal' : `${policy}_spending_proposal`;
    expect(q.active_mods[id].def.electoral_origin).toBe('erc');
    expect(q.active_mods[id].def.electoral_cabinet).toEqual(['erc']);
  });

  it('service cuts send ERC voters to CUP, federal left and abstention', () => {
    const { adapter, q } = setup();
    const before = support(q);
    adapter.goToScene('generalitat_health_card.healthcare_cuts');
    expect(q.healthcare_spending).toBe(-1);
    expect(q.erc_parlament_barcelona_retired_support).toBeLessThan(before.erc_parlament_barcelona_retired_support);
    for (const destination of ['cup', 'icv', 'abstain']) expect(q[`${destination}_parlament_barcelona_retired_support`]).toBeGreaterThan(before[`${destination}_parlament_barcelona_retired_support`]);
    expect(q.ciu_parlament_barcelona_retired_support).toBe(before.ciu_parlament_barcelona_retired_support);
    checkConservation(q, before);
  });

  it('side contests use their own donors when the Parliamentary route is unavailable', () => {
    const { adapter, q } = setup();
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
      for (const donor of ['psc', 'icv', 'abstain']) {
        q[`erc_parlament_${province}_${demo}_support`] += q[`${donor}_parlament_${province}_${demo}_support`];
        q[`${donor}_parlament_${province}_${demo}_support`] = 0;
      }
    }
    const before = support(q);
    adapter.goToScene('generalitat_health_card.approve_raise');
    expect(q.healthcare_spending).toBe(1);
    for (const [key, value] of Object.entries(before).filter(([key]) => key.includes('_parlament_'))) expect(q[key]).toBe(value);
    expect(q.erc_congreso_catalunya_support - before.erc_congreso_catalunya_support).toBeCloseTo(0.04);
    expect(q.erc_local_barcelona_support - before.erc_local_barcelona_support).toBeCloseTo(0.08 * 0.75);
    expect(q.psoe_congreso_catalunya_support).toBeLessThan(before.psoe_congreso_catalunya_support);
    expect(q.psc_local_barcelona_support).toBeLessThan(before.psc_local_barcelona_support);
    checkConservation(q, before);
  });

  it('tiny Parliamentary donors limit that contest while the side profiles remain independently authored', () => {
    const { adapter, q } = setup();
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
      for (const donor of ['psc', 'icv', 'abstain']) {
        q[`erc_parlament_${province}_${demo}_support`] += q[`${donor}_parlament_${province}_${demo}_support`] - 0.000001;
        q[`${donor}_parlament_${province}_${demo}_support`] = 0.000001;
      }
    }
    const before = support(q);
    adapter.goToScene('generalitat_health_card.approve_raise');
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
      const gain = q[`erc_parlament_${province}_${demo}_support`] - before[`erc_parlament_${province}_${demo}_support`];
      expect(gain).toBeLessThanOrEqual(0.0000015);
    }
    expect(q.erc_congreso_catalunya_support - before.erc_congreso_catalunya_support).toBeCloseTo(0.04);
    expect(q.erc_local_barcelona_support - before.erc_local_barcelona_support).toBeCloseTo(0.08 * 0.75);
    checkConservation(q, before);
  });

  it('changing province and demographic population weights leaves authored side effects unchanged', () => {
    const normal = setup();
    const skewed = setup();
    for (const province of skewed.q.parlament_constituencies) for (const demo of skewed.q.parlament_demographics)
      skewed.q[`parlament_${province}_${demo}_pop`] = province === 'lleida' && demo === 'retired' ? 1_000_000_000 : 1;
    const normalBefore = support(normal.q), skewedBefore = support(skewed.q);
    for (const state of [normal, skewed]) state.adapter.goToScene('generalitat_health_card.approve_raise');
    for (const [key, value] of Object.entries(normalBefore).filter(([key]) => key.includes('_congreso_') || key.includes('_local_')))
      expect(normal.q[key] - value).toBeCloseTo(skewed.q[key] - skewedBefore[key], 10);
    checkConservation(normal.q, normalBefore);
    checkConservation(skewed.q, skewedBefore);
  });

  it('small side-contest donors independently cap the direct profiles and conserve their pools', () => {
    const { adapter, q } = setup();
    for (const donor of ['psoe', 'iu', 'abstain']) {
      const key = `${donor}_congreso_catalunya_support`;
      q.erc_congreso_catalunya_support += q[key] - 0.000001;
      q[key] = 0.000001;
    }
    for (const donor of ['psc', 'icv']) {
      const key = `${donor}_local_barcelona_support`;
      q.erc_local_barcelona_support += q[key] - 0.000001;
      q[key] = 0.000001;
    }
    const before = support(q);
    adapter.goToScene('generalitat_health_card.approve_raise');
    const national = q.erc_congreso_catalunya_support - before.erc_congreso_catalunya_support;
    const local = q.erc_local_barcelona_support - before.erc_local_barcelona_support;
    expect(national).toBeGreaterThan(0);
    expect(national).toBeLessThanOrEqual(0.0000015 + 1e-10);
    expect(local).toBeGreaterThan(0);
    expect(local).toBeLessThanOrEqual(0.000001 + 1e-10);
    checkConservation(q, before);
  });

  it('a stale unfielded joint list does not capture native ERC local or national credit', () => {
    const { adapter, q } = setup({ player_in_jxsi: true, erc_in_jxsi: true, health_minister_party: 'jxsi', cat_coalition: ['jxsi'], jxsi_in_congreso: false, jxsi_in_local: false });
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
      q[`jxsi_parlament_${province}_${demo}_support`] = q[`erc_parlament_${province}_${demo}_support`] + q[`ciu_parlament_${province}_${demo}_support`];
      q[`erc_parlament_${province}_${demo}_support`] = 0;
      q[`ciu_parlament_${province}_${demo}_support`] = 0;
    }
    if (!q.congreso_parties_catalunya.includes('jxsi')) q.congreso_parties_catalunya.push('jxsi');
    q.jxsi_congreso_catalunya_support = 1;
    q.jxsi_local_barcelona_support = 1;
    const before = support(q);
    adapter.goToScene('generalitat_health_card.approve_raise');
    expect(q.jxsi_congreso_catalunya_support).toBe(before.jxsi_congreso_catalunya_support);
    expect(q.jxsi_local_barcelona_support).toBe(before.jxsi_local_barcelona_support);
    expect(q.erc_congreso_catalunya_support).toBeGreaterThan(before.erc_congreso_catalunya_support);
    expect(q.erc_local_barcelona_support).toBeGreaterThan(before.erc_local_barcelona_support);
    expect(q.erc_in_jxsi).toBe(true);
    checkConservation(q, before);
  });

  it('current CDC is the business-policy donor before a separately surviving PDeCAT', () => {
    const { adapter, q } = setup({ parlament_current_ciu: 'cdc', pdcat_split: true });
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
      q[`cdc_parlament_${province}_${demo}_support`] = q[`ciu_parlament_${province}_${demo}_support`];
      q[`ciu_parlament_${province}_${demo}_support`] = 0;
      q[`pdcat_parlament_${province}_${demo}_support`] = 5;
      q[`psc_parlament_${province}_${demo}_support`] -= 5;
    }
    const before = support(q);
    adapter.goToScene('generalitat_economy_card.atc1');
    expect(q.cdc_parlament_barcelona_middle_support).toBeLessThan(before.cdc_parlament_barcelona_middle_support);
    expect(q.pdcat_parlament_barcelona_middle_support).toBe(before.pdcat_parlament_barcelona_middle_support);
    checkConservation(q, before);
  });

  it.each([
    ['generalitat_economy_card.atc1', { economy_atc_level: 0 }],
    ['generalitat_economy_card.atc2', { economy_atc_level: 1 }],
    ['generalitat_economy_card.sell_assets', {}],
    ['generalitat_economy_card.raise_pay', {}],
    ['generalitat_economy_card.cut_pay', {}],
    ['generalitat_economy_card.revert_cuts', { economy_cuts: 1 }],
    ['generalitat_vp_card.digital1', {}],
    ['generalitat_vp_card.digital2', { vp_digital_level: 1, parlament_digital_passed: true, cybersecurity_agency_scope: 'generalitat', active_mods: { digital_agency_core: {active: true, def: {id: 'digital_agency_core', targets: {}}, ticks_active: 0, live_effect: {}} } }],
    ['generalitat_health_card.universal_access1', {}],
    ['generalitat_health_card.reproductive_care', {}],
    ['generalitat_health_card.trans_care', { healthcare_sexual_level: 1, parlament_lgbt_passed: true }],
  ])('%s conserves all pools and gives Parlament the largest direct response', (scene, state) => {
    const { adapter, q } = setup(state);
    const before = support(q);
    adapter.goToScene(scene);
    const parliament = Math.max(...Object.entries(before).filter(([key]) => key.startsWith('erc_parlament_')).map(([key, value]) => Math.abs(q[key] - value)));
    expect(parliament).toBeGreaterThan(0);
    for (const [key, value] of Object.entries(before).filter(([key]) => key.startsWith('erc_congreso_') || key === 'erc_local_barcelona_support')) expect(Math.abs(q[key] - value)).toBeLessThan(parliament);
    for (const [key, value] of Object.entries(before).filter(([key]) => key.includes('_congreso_') && !key.includes('_catalunya_'))) expect(q[key]).toBe(value);
    checkConservation(q, before);
  });
});

describe('Provider ownership changes require actual parliamentary passage', () => {
  it.each(['healthcare', 'education'].flatMap(service => [
    [service, 'de_public', 3, 2, 'de_public'],
    [service, 'privatize', -2, -3, 'privatization'],
    [service, 'de_privatize', -3, -2, 'de_privatization'],
  ] as const))('%s %s passage transfers votes once and conserves each pool', (service, action, old, next, token) => {
    const { adapter, q } = setup({ [`${service}_spending`]: old });
    const before = support(q);
    q[`${service}_${token}_introduced`] = true;
    adapter.goToScene(`parlament_exec_${service}_${action}`);
    expect(q[`${service}_${token}_introduced`]).toBe(true);
    expect(q).not.toHaveProperty(`parlament_${service}_${token}_open`);
    adapter.goToScene(`parlament_exec_${service}_${action}.vote_yes`);
    expect(q[`${service}_${token}_introduced`]).toBe(false);
    expect(q[`${service}_spending`]).toBe(next);
    const delta = q.erc_parlament_barcelona_ind_support - before.erc_parlament_barcelona_ind_support;
    expect(action === 'de_privatize' ? delta > 0 : delta < 0).toBe(true);
    const delivered = support(q), budget = q.government_budget;
    adapter.goToScene(`parlament_exec_${service}_${action}.on_yes`);
    expect(support(q)).toEqual(delivered);
    expect(q.government_budget).toBe(budget);
    checkConservation(q, before);
  });

  it.each(['healthcare', 'education'])('%s ownership proposals earn no votes, successful passage does, repeated arrivals do not', (service) => {
    const { adapter, q } = setup({ [`${service}_spending`]: 2 });
    const before = support(q);
    adapter.goToScene(`generalitat_${service === 'healthcare' ? 'health' : 'education'}_card.${service}_public`);
    expect(support(q)).toEqual(before);
    adapter.goToScene(`parlament_exec_${service}_public`);
    expect(q[`${service}_public_introduced`]).toBe(true);
    expect(q).not.toHaveProperty(`parlament_${service}_public_open`);
    adapter.goToScene(`parlament_exec_${service}_public.vote_yes`);
    expect(q[`${service}_public_introduced`]).toBe(false);
    expect(q[`${service}_spending`]).toBe(3);
    expect(q.erc_parlament_barcelona_ind_support).toBeGreaterThan(before.erc_parlament_barcelona_ind_support);
    const delivered = support(q), budget = q.government_budget;
    adapter.goToScene(`parlament_exec_${service}_public.on_yes`);
    expect(support(q)).toEqual(delivered);
    expect(q.government_budget).toBe(budget);
    checkConservation(q, before);
  });

  it.each(['healthcare', 'education'])('direct/unopened %s passage cannot award votes or spend resources', (service) => {
    const { adapter, q } = setup({ [`${service}_spending`]: 2, [`parlament_${service}_public_passed`]: true });
    const before = support(q), budget = q.government_budget;
    adapter.goToScene(`parlament_exec_${service}_public.on_yes`);
    expect(q[`${service}_spending`]).toBe(2);
    expect(q.government_budget).toBe(budget);
    expect(support(q)).toEqual(before);
  });

  it.each(['healthcare', 'education'].flatMap(service => [
    [service, 'public', 2, 'public'],
    [service, 'de_public', 3, 'de_public'],
    [service, 'privatize', -2, 'privatization'],
    [service, 'de_privatize', -3, 'de_privatization'],
  ] as const))('rejected %s %s bill clears the pending proposal without an electoral reward', (service, action, old, token) => {
    const { adapter, q } = setup({ [`${service}_spending`]: old });
    const before = support(q);
    q[`${service}_${token}_introduced`] = true;
    adapter.goToScene(`parlament_exec_${service}_${action}`);
    expect(q[`${service}_${token}_introduced`]).toBe(true);
    adapter.goToScene(`parlament_exec_${service}_${action}.vote_no`);
    expect(q[`${service}_${token}_introduced`]).toBe(false);
    expect(q[`${service}_spending`]).toBe(old);
    expect(support(q)).toEqual(before);
    if (action === 'public') {
      adapter.goToScene(`generalitat_${service === 'healthcare' ? 'health' : 'education'}_card.${service}_public`);
      expect(q[`${service}_public_introduced`]).toBe(true);
      expect(support(q)).toEqual(before);
    }
  });

  it.each(['healthcare', 'education'])('%s bill also clears its pending flag when enactment loses ministerial authority', (service) => {
    const { adapter, q } = setup({ [`${service}_spending`]: 2, [`${service}_public_introduced`]: true });
    const before = support(q), budget = q.government_budget;
    adapter.goToScene(`parlament_exec_${service}_public`);
    q[`${service === 'healthcare' ? 'health' : 'education'}_minister_party`] = 'psc';
    adapter.goToScene(`parlament_exec_${service}_public.vote_yes`);
    expect(q[`parlament_${service}_public_passed`]).toBe(true);
    expect(q[`${service}_public_introduced`]).toBe(false);
    expect(q[`${service}_spending`]).toBe(2);
    expect(q.government_budget).toBe(budget);
    expect(support(q)).toEqual(before);
  });
});

describe('Curriculum response replaces the old weighted package', () => {
  function revise(adapter: DendryAdapter, q: Record<string, any>, main: string, secondary: string) {
    Object.assign(q, { education_policy_draft_main: main, education_policy_draft_secondary: secondary });
    adapter.goToScene('generalitat_education_card.policy_apply');
    expect(q.education_policy_applied).toBe(true);
  }

  it('moves voters along the six distinct focus routes and permits only a funded, changed package', () => {
    const { adapter, q } = setup({ education_spending: 2 });
    const before = support(q);
    revise(adapter, q, 'language', 'humanities');
    expect(q.ciu_parlament_girona_rural_support).toBeLessThan(before.ciu_parlament_girona_rural_support);
    expect(q.cup_parlament_girona_rural_support).toBeLessThan(before.cup_parlament_girona_rural_support);
    const after = support(q), budget = q.government_budget;
    adapter.goToScene('generalitat_education_card.policy_apply');
    expect(q.education_policy_applied).toBe(false);
    expect(support(q)).toEqual(after);
    expect(q.government_budget).toBe(budget);
    checkConservation(q, before);
  });

  it('A to B to A restores the same current-policy electorate without accumulating gains', () => {
    const { adapter, q } = setup({ education_spending: 2 });
    revise(adapter, q, 'science', 'humanities');
    const first = support(q);
    const records = q.active_mods.education_school_policy.def.electoral_transfers;
    expect(records.length).toBeGreaterThan(0);
    expect(JSON.parse(JSON.stringify(records))).toEqual(records);
    q.year++;
    revise(adapter, q, 'segregation', 'democratic');
    q.year++;
    revise(adapter, q, 'science', 'humanities');
    for (const [key, value] of Object.entries(first)) expect(q[key]).toBeCloseTo(value, 8);
    expect(Object.keys(q.active_mods).filter(id => id === 'education_school_policy')).toHaveLength(1);
  });

  it('swapping the main/secondary focus changes responses rather than awarding both again', () => {
    const { adapter, q } = setup({ education_spending: 2 });
    revise(adapter, q, 'science', 'humanities');
    const main = q.erc_parlament_barcelona_young_support;
    q.year++;
    revise(adapter, q, 'humanities', 'science');
    expect(q.erc_parlament_barcelona_young_support).toBeLessThan(main);
  });
});
