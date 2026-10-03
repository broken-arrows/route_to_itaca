import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');
const provinces = ['barcelona', 'girona', 'lleida', 'tarragona'];
const groups = ['buss', 'ind', 'middle', 'young', 'retired', 'rural', 'unemployed'];

function setup(player = 'erc', cabinet = [player], overrides: Record<string, any> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  for (const p of q.parties) {
    q[p + '_parlament_s'] = 0;
    q[p + '_relations'] = 80;
    q[p + '_roadmap'] = 0;
  }
  Object.assign(q, {
    player_party: player, player_in_jxsi: false, player_in_jxcat: false,
    erc_in_jxsi: false, cup_in_jxsi: false, erc_in_jxcat: false, cup_in_jxcat: false,
    cat_coalition: cabinet, cat_coalition_support: [], cat_coalition_abstain: [],
    president_party: player, economy_minister_party: player,
    health_minister_party: player, education_minister_party: player, vicepresident_party: player,
    bnl_minister_party: player, territory_minister_party: player,
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    [player + '_parlament_s']: 100, ppc_parlament_s: 35, parlament_s_majority: 68,
    cat_caretaker_gov: false, party_resources: 40, government_budget: 40,
    year: 2014, budget_year: 2013, generalitat_surplus: 12,
    tax_policy_level: 1, social_spending_cap: 0,
    education_spending: 0, healthcare_spending: 0, public_salary_level: 0,
    education_prepared_level: null, healthcare_prepared_level: null, public_salary_prepared_level: null,
    coalition_policy_commitments: [], active_mods: {}, mod_log: [], economy_cuts: 0,
    ...overrides,
  });
  for (const c of provinces) for (const d of groups) {
    for (const p of q.parties) q[`${p}_parlament_${c}_${d}_support`] = 0;
    const stocks: Record<string, number> = { erc: 8, cup: 5, ciu: 10, psc: 16, icv: 10, ppc: 10, cs: 7, abstain: 34 };
    for (const [p, value] of Object.entries(stocks)) q[`${p}_parlament_${c}_${d}_support`] = value;
  }
  return { adapter, q };
}

function support(q: Record<string, any>) {
  return Object.fromEntries(Object.entries(q).filter(([key]) => /_(parlament_.*|congreso_.*|local_barcelona)_support$/.test(key)));
}

function visit(adapter: DendryAdapter, scene: string) {
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  try {
    adapter.goToScene(scene);
    expect(errors.mock.calls.filter(([message]) => !String(message).startsWith('Page loaded from unexpected origin'))).toEqual([]);
  } finally { errors.mockRestore(); }
}

function submit(adapter: DendryAdapter, tax = 1, cap = 0, projects: Record<string, boolean> = {}) {
  const q = adapter.qualities as Record<string, any>;
  const included = Object.fromEntries(['county', 'b40', 'metro1', 'metro2', 'water2', 'water3', 'tmobilitat2', 'rail1', 'rail2', 'icf', 'digital', 'universal'].map(k => [k, !!projects[k]]));
  q.budget_draft_origin = {
    author: 'player', proposer_party: q.player_party, year: q.year,
    tax_policy_level: tax, social_spending_cap: cap,
    enacted_tax_policy_level: q.tax_policy_level, enacted_social_spending_cap: q.social_spending_cap,
    surplus_at_draft: q.generalitat_surplus, projects: { ...included },
  };
  q.budget_draft = { tax_policy_level: tax, social_spending_cap: cap, projects: included };
  visit(adapter, 'parlament_budget');
  if (q.parlament_budget_player_default === 'no' && q.parlament_budget_player_can_yes) {
    visit(adapter, 'parlament_budget.vote_yes');
    visit(adapter, 'parlament_budget.review');
  }
  expect(q.parlament_budget_can_pass).toBe(true);
}

function conserve(q: Record<string, any>, before: Record<string, any>) {
  for (const c of provinces) for (const d of groups) {
    const keys = [...q.parties, 'abstain'].map(p => `${p}_parlament_${c}_${d}_support`);
    expect(keys.reduce((n, k) => n + q[k], 0)).toBeCloseTo(keys.reduce((n, k) => n + before[k], 0), 9);
    for (const k of keys) expect(q[k]).toBeGreaterThanOrEqual(0);
  }
  for (const c of q.congreso_constituencies) {
    const keys = [...q['congreso_parties_' + c], 'abstain'].map(p => `${p}_congreso_${c}_support`);
    expect(keys.reduce((n, k) => n + q[k], 0)).toBeCloseTo(keys.reduce((n, k) => n + before[k], 0), 9);
  }
  const keys = q.parties_bcn.map((p: string) => p + '_local_barcelona_support');
  expect(keys.reduce((n: number, k: string) => n + q[k], 0)).toBeCloseTo(keys.reduce((n: number, k: string) => n + before[k], 0), 9);
}

describe('Budget policy-specific voter transfers', () => {
  it.each(['erc', 'cup'])('%s tax increases lose the named taxpayer groups rather than every demographic', player => {
    const { adapter, q } = setup(player);
    submit(adapter, 2);
    const before = support(q);
    visit(adapter, 'parlament_budget.pass');
    expect(q[`${player}_parlament_barcelona_buss_support`]).toBeLessThan(before[`${player}_parlament_barcelona_buss_support`]);
    expect(q.ciu_parlament_barcelona_buss_support).toBeGreaterThan(before.ciu_parlament_barcelona_buss_support);
    expect(q[`${player}_parlament_barcelona_rural_support`]).toBe(before[`${player}_parlament_barcelona_rural_support`]);
    expect(q.icv_parlament_barcelona_buss_support).toBe(before.icv_parlament_barcelona_buss_support);
    conserve(q, before);
  });

  it('unchanged renewal awards no votes and an attempted repeated passage is inert', () => {
    const { adapter, q } = setup();
    submit(adapter);
    const before = support(q);
    visit(adapter, 'parlament_budget.pass');
    expect(support(q)).toEqual(before);
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    try { adapter.goToScene('parlament_budget.pass'); } finally { errors.mockRestore(); }
    expect(support(q)).toEqual(before);
  });

  it('a pending Health raise has no preparation gain and delivers minister-owned credit once', () => {
    const agreement = [{ kind: 'social_spending_cap', target: 1, scope: 'cabinet', signatories: ['erc', 'ciu'], status: 'pending_budget' }];
    const pending = setup('erc', ['erc', 'ciu'], { ciu_parlament_s: 10, coalition_policy_commitments: structuredClone(agreement) });
    const control = setup('erc', ['erc', 'ciu'], { ciu_parlament_s: 10, coalition_policy_commitments: structuredClone(agreement) });
    const before = support(pending.q);
    visit(pending.adapter, 'generalitat_health_card.approve_raise');
    expect(pending.q.healthcare_spending).toBe(0);
    expect(support(pending.q)).toEqual(before);
    submit(pending.adapter, 1, 1);
    submit(control.adapter, 1, 1);
    visit(pending.adapter, 'parlament_budget.pass');
    visit(control.adapter, 'parlament_budget.pass');
    const ercGain = pending.q.erc_parlament_barcelona_retired_support - control.q.erc_parlament_barcelona_retired_support;
    const ciuGain = pending.q.ciu_parlament_barcelona_retired_support - control.q.ciu_parlament_barcelona_retired_support;
    expect(ercGain).toBeCloseTo(0.7 * 0.8);
    expect(ciuGain).toBeCloseTo(0.7 * 0.2);
    expect(pending.q.psc_parlament_barcelona_retired_support).toBeLessThan(control.q.psc_parlament_barcelona_retired_support);
    expect(pending.q.icv_parlament_barcelona_retired_support).toBeLessThan(control.q.icv_parlament_barcelona_retired_support);
    expect(pending.q.ppc_parlament_barcelona_retired_support).toBe(control.q.ppc_parlament_barcelona_retired_support);
    conserve(pending.q, before);
  });

  it('budget delivery distinguishes Health users from Education voters', () => {
    const health = setup('erc', ['erc'], { healthcare_prepared_level: 1 });
    const education = setup('erc', ['erc'], { education_prepared_level: 1 });
    submit(health.adapter, 1, 1);
    submit(education.adapter, 1, 1);
    visit(health.adapter, 'parlament_budget.pass');
    visit(education.adapter, 'parlament_budget.pass');
    expect(health.q.erc_parlament_barcelona_retired_support).toBeGreaterThan(education.q.erc_parlament_barcelona_retired_support);
    expect(education.q.erc_parlament_barcelona_young_support).toBeGreaterThan(health.q.erc_parlament_barcelona_young_support);
    expect(health.q.ppc_parlament_barcelona_retired_support).toBe(10);
    expect(education.q.ppc_parlament_barcelona_young_support).toBe(10);
  });

  it('local and Congreso effects are smaller, conserve their pools and stay within Catalunya', () => {
    const { adapter, q } = setup();
    submit(adapter, 2);
    const before = support(q);
    visit(adapter, 'parlament_budget.pass');
    const provincialLoss = before.erc_parlament_barcelona_buss_support - q.erc_parlament_barcelona_buss_support;
    const nationalLoss = before.erc_congreso_catalunya_support - q.erc_congreso_catalunya_support;
    const localLoss = before.erc_local_barcelona_support - q.erc_local_barcelona_support;
    expect(nationalLoss).toBeGreaterThan(0);
    expect(nationalLoss).toBeLessThan(provincialLoss * 0.15);
    expect(localLoss).toBeGreaterThan(0);
    expect(localLoss).toBeLessThan(provincialLoss * 0.3);
    for (const [key, value] of Object.entries(before)) if (key.includes('_congreso_') && !key.includes('_congreso_catalunya_')) expect(q[key]).toBe(value);
    conserve(q, before);
  });

  it('a failed budget charges once per year and merely editing its cap adds no invented disappointment', () => {
    const first = setup();
    const second = setup();
    submit(first.adapter, 1, 0);
    submit(second.adapter, 1, 1);
    const before = support(first.q);
    visit(first.adapter, 'parlament_budget.walk_away');
    visit(second.adapter, 'parlament_budget.walk_away');
    expect(support(first.q)).toEqual(support(second.q));
    expect(first.q.erc_parlament_barcelona_middle_support).toBeLessThan(before.erc_parlament_barcelona_middle_support);
    const after = support(first.q);
    visit(first.adapter, 'parlament_budget.walk_away');
    expect(support(first.q)).toEqual(after);
    conserve(first.q, before);
  });
});

describe('Delivered coalition concessions use the same voter rules', () => {
  it('a CiU spending concession changes the cut service and transfers voters collectively', () => {
    const { adapter, q } = setup('erc', ['erc', 'ciu'], {
      generalitat_surplus: -1, education_spending: 1,
      coalition_affairs_concession_target: '',
    });
    const before = support(q);
    visit(adapter, 'generalitat_coalition_card.concessions_ciu');
    expect(q.education_spending).toBe(0);
    expect(q.healthcare_spending).toBe(0);
    expect(q.erc_parlament_barcelona_young_support).toBeLessThan(before.erc_parlament_barcelona_young_support);
    expect(q.ciu_parlament_barcelona_young_support).toBeLessThan(before.ciu_parlament_barcelona_young_support);
    expect(q.icv_parlament_barcelona_young_support).toBeGreaterThan(before.icv_parlament_barcelona_young_support);
    expect(q.ppc_parlament_barcelona_young_support).toBe(before.ppc_parlament_barcelona_young_support);
    conserve(q, before);
  });

  it('a CiU tax concession reaches taxpayers while untouched demographics receive nothing', () => {
    const { adapter, q } = setup('erc', ['erc', 'ciu'], { coalition_affairs_concession_target: '' });
    const before = support(q);
    visit(adapter, 'generalitat_coalition_card.concessions_ciu');
    expect(q.tax_policy_level).toBe(0);
    expect(q.erc_parlament_barcelona_buss_support).toBeGreaterThan(before.erc_parlament_barcelona_buss_support);
    expect(q.cs_parlament_barcelona_buss_support).toBeLessThan(before.cs_parlament_barcelona_buss_support);
    expect(q.erc_parlament_barcelona_rural_support).toBe(before.erc_parlament_barcelona_rural_support);
    conserve(q, before);
  });

  it('a left-partner spending concession delivers a service response and leaves the right untouched', () => {
    const { adapter, q } = setup('erc', ['erc', 'icv'], {
      education_spending: 0, healthcare_spending: 1, public_salary_level: 2,
      social_spending_cap: 2, tax_policy_level: 2, coalition_affairs_concession_target: '',
    });
    const before = support(q);
    visit(adapter, 'generalitat_coalition_card.concessions_icv');
    expect(q.education_spending).toBe(1);
    expect(q.erc_parlament_barcelona_young_support).toBeGreaterThan(before.erc_parlament_barcelona_young_support);
    expect(q.icv_parlament_barcelona_young_support).toBeGreaterThan(before.icv_parlament_barcelona_young_support);
    expect(q.psc_parlament_barcelona_young_support).toBeLessThan(before.psc_parlament_barcelona_young_support);
    expect(q.ppc_parlament_barcelona_young_support).toBe(before.ppc_parlament_barcelona_young_support);
    conserve(q, before);
  });

  it('a left-partner tax concession loses named taxpayer votes and a capped repeat awards nothing', () => {
    const { adapter, q } = setup('erc', ['erc', 'icv'], {
      education_spending: 2, healthcare_spending: 2, public_salary_level: 2,
      social_spending_cap: 2, tax_policy_level: 1, coalition_affairs_concession_target: '',
    });
    const before = support(q);
    visit(adapter, 'generalitat_coalition_card.concessions_icv');
    expect(q.tax_policy_level).toBe(2);
    expect(q.erc_parlament_barcelona_buss_support).toBeLessThan(before.erc_parlament_barcelona_buss_support);
    expect(q.ciu_parlament_barcelona_buss_support).toBeGreaterThan(before.ciu_parlament_barcelona_buss_support);
    conserve(q, before);
    const after = support(q);
    visit(adapter, 'generalitat_coalition_card.concessions_icv');
    expect(support(q)).toEqual(after);
  });

  it('extraordinary austerity loses social voters and its reversal earns a delivered response once', () => {
    const cuts = setup('erc', ['erc', 'ciu'], {
      generalitat_surplus: -1, education_spending: -2, healthcare_spending: -2,
      public_salary_level: -2, coalition_affairs_concession_target: '',
    });
    const beforeCuts = support(cuts.q);
    visit(cuts.adapter, 'generalitat_coalition_card.concessions_ciu');
    expect(cuts.q.economy_cuts).toBe(1);
    expect(cuts.q.erc_parlament_barcelona_unemployed_support).toBeLessThan(beforeCuts.erc_parlament_barcelona_unemployed_support);
    expect(cuts.q.icv_parlament_barcelona_unemployed_support).toBeGreaterThan(beforeCuts.icv_parlament_barcelona_unemployed_support);
    conserve(cuts.q, beforeCuts);

    const reversal = setup('erc', ['erc', 'icv'], {
      education_spending: 2, healthcare_spending: 2, public_salary_level: 2,
      tax_policy_level: 2, social_spending_cap: 2, economy_cuts: 1,
      coalition_affairs_concession_target: '',
    });
    const beforeReversal = support(reversal.q);
    visit(reversal.adapter, 'generalitat_coalition_card.concessions_icv');
    expect(reversal.q.economy_cuts).toBe(0);
    expect(reversal.q.erc_parlament_barcelona_unemployed_support).toBeGreaterThan(beforeReversal.erc_parlament_barcelona_unemployed_support);
    conserve(reversal.q, beforeReversal);
    const after = support(reversal.q);
    visit(reversal.adapter, 'generalitat_coalition_card.concessions_icv');
    expect(support(reversal.q)).toEqual(after);
  });

  it('caretaker concessions award no electoral response', () => {
    const { adapter, q } = setup('erc', ['erc', 'ciu'], {
      cat_caretaker_gov: true, coalition_affairs_concession_target: '',
    });
    const before = support(q);
    visit(adapter, 'generalitat_coalition_card.concessions_ciu');
    expect(support(q)).toEqual(before);
  });

  it('concessions deduplicate joint-list cabinet carriers and keep inactive components at zero', () => {
    const fixtures = [['jxsi'], ['jxsi', 'erc']].map(cabinet => {
      const state = setup('erc', cabinet, {
        player_in_jxsi: true, erc_in_jxsi: true, ciu_in_jxsi: true,
        generalitat_surplus: -1, education_spending: 1, coalition_affairs_concession_target: '',
      });
      for (const c of provinces) for (const d of groups) {
        state.q[`jxsi_parlament_${c}_${d}_support`] = 18;
        state.q[`erc_parlament_${c}_${d}_support`] = 0;
        state.q[`ciu_parlament_${c}_${d}_support`] = 0;
      }
      return state;
    });
    for (const { adapter, q } of fixtures) {
      const before = support(q);
      visit(adapter, 'generalitat_coalition_card.concessions_ciu');
      expect(q.jxsi_parlament_barcelona_young_support).toBeLessThan(before.jxsi_parlament_barcelona_young_support);
      expect(q.erc_parlament_barcelona_young_support).toBe(0);
      expect(q.ciu_parlament_barcelona_young_support).toBe(0);
      conserve(q, before);
    }
    expect(support(fixtures[0].q)).toEqual(support(fixtures[1].q));
  });
});
