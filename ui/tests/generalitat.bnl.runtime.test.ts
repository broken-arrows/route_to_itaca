import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');
const lib = gameLib as any;
const ercFactions = ['left', 'core', 'pragmatic', 'civic'];
const cupFactions = ['endavant', 'poblelliure', 'thirdsector'];

function setup(overrides: Record<string, any> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  for (const party of q.parties) {
    q[`${party}_parlament_s`] = 0;
    q[`${party}_relations`] = 50;
    q[`${party}_roadmap`] = 0;
  }
  for (const faction of ercFactions) q[`erc_${faction}_dissent`] = 10;
  for (const faction of cupFactions) q[`cup_${faction}_dissent`] = 10;
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    cat_coalition: ['erc'], cat_coalition_support: [], cat_coalition_abstain: [],
    president_party: 'erc', economy_minister_party: 'erc', bnl_minister_party: 'erc',
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    erc_parlament_s: 70, ppc_parlament_s: 65, parlament_size: 135, parlament_s_majority: 68,
    year: 2014, budget_year: 2013, cat_caretaker_gov: false,
    party_resources: 10, government_budget: 30, generalitat_surplus: 0,
    tax_policy_level: 1, social_spending_cap: 0,
    education_spending: 0, healthcare_spending: 0, public_salary_level: 0,
    public_salary_prepared_level: null, education_prepared_level: null, healthcare_prepared_level: null,
    coalition_policy_commitments: [], economy_cuts: 0, active_mods: {}, mod_log: [],
    generalitat_bnl_timer: 0, generalitat_economy_timer: 0,
    ...overrides,
  });
  return { adapter, q };
}

function action(adapter: DendryAdapter, name: string) {
  return adapter.goToScene(`generalitat_bnl_card.${name}`);
}

function maturity(q: Record<string, any>, ticks: number) {
  for (let i = 0; i < ticks; i++) lib.engineTick(q);
}

function contribution(q: Record<string, any>, target: string) {
  return Object.values(q.active_mods).reduce((total: number, law: any) => total + (law.live_effect[target] || 0), 0);
}

function submit(adapter: DendryAdapter, include = true) {
  const q = adapter.qualities as Record<string, any>;
  adapter.goToScene('generalitat_economy_card.prepare_budget');
  expect(q.economy_budget_icf_available).toBe(true);
  if (include) adapter.goToScene('generalitat_economy_card.budget_icf');
  adapter.goToScene('generalitat_economy_card.submit_budget');
  return q;
}

describe('Economy modifier scaling', () => {
  it('preserves the ATC baseline ramp and reaches the stronger combined revenue', () => {
    const { adapter, q } = setup();
    adapter.goToScene('generalitat_economy_card.atc1');
    maturity(q, 2);
    const baseline = q.active_mods.economy_atc1.live_effect.generalitat_surplus;
    adapter.goToScene('generalitat_economy_card.atc2');
    expect(q.active_mods.economy_atc1.ticks_active).toBe(2);
    expect(q.active_mods.economy_atc1.hidden).toBe(true);
    maturity(q, 1);
    expect(contribution(q, 'generalitat_surplus')).toBeGreaterThan(baseline);
    maturity(q, 11);
    // The shared setter also synchronizes taxes; measure the ATC investments themselves.
    expect(q.active_mods.economy_atc1.live_effect.generalitat_surplus +
      q.active_mods.economy_atc2.live_effect.generalitat_surplus).toBeCloseTo(0.03);
  });

  it('stacks rental obligations and improved pay without changing opening pay cuts', () => {
    const { adapter, q } = setup();
    adapter.goToScene('generalitat_economy_card.sell_assets');
    adapter.goToScene('generalitat_economy_card.sell_assets');
    expect(q.active_mods.economy_asset_sale_1.def.targets.generalitat_surplus).toBe(-0.01);
    expect(q.active_mods.economy_asset_sale_2.def.targets.generalitat_surplus).toBe(-0.01);
    q.active_mods = {};
    q.public_salary_level = 2;
    adapter.goToScene('generalitat_setpolicy');
    maturity(q, 1);
    expect(contribution(q, 'welfare_index_growth')).toBeCloseTo(0.14);
    expect(contribution(q, 'generalitat_surplus')).toBeCloseTo(-0.03); // High taxes +0.02, pay -0.05.
    q.public_salary_level = -1;
    adapter.goToScene('generalitat_setpolicy');
    expect(q.active_mods.public_salary_cuts.def.targets.welfare_index_growth).toBe(-0.02);
    expect(q.active_mods.public_salary_improvement.active).toBe(false);
  });

  it('enacts the revised extraordinary austerity package and removes it through Economy', () => {
    const { adapter, q } = setup({ education_spending: -2, healthcare_spending: -2,
      public_salary_level: -2, generalitat_surplus: -2, cat_coalition: ['erc', 'ciu'],
      coalition_affairs_includes_ciu: true, coalition_affairs_concession_target: '' });
    adapter.goToScene('generalitat_coalition_card.concessions_ciu');
    expect(q.active_mods.general_cuts_1.def.targets).toEqual({
      welfare_index_growth: -0.2, social_dissent_eq: 1, generalitat_surplus: 0.05,
      unemployment_recovery: -0.1, gdp_growth: -0.1,
    });
    adapter.goToScene('generalitat_economy_card.revert_cuts');
    expect(q.economy_cuts).toBe(0);
    expect(q.active_mods.general_cuts_1.active).toBe(false);
  });
});

describe('Business and Labour policies', () => {
  it('starts all tracks neutral and shows the Economy requirement for an ICF branch', () => {
    const { adapter, q } = setup({ economy_minister_party: 'psc' });
    expect([q.bnl_icf_level, q.bnl_inspection_level, q.bnl_tourism_level, q.bnl_union_level]).toEqual([0, 0, 0, 0]);
    action(adapter, 'icf1');
    const frame = adapter.goToScene('generalitat_bnl_card');
    const choice = frame.choices.find(c => c.id.endsWith('.icf_cooperative'))!;
    expect(choice).toBeDefined();
    expect(choice.canChoose).toBe(false);
    expect(choice.subtitle).toContain('Economy');
    action(adapter, 'icf_cooperative');
    expect(q.bnl_icf_prepared_branch).toBe(null);
  });

  it.each(['icf1', 'inspection1', 'tourism_more', 'unions_sovereignty'])('%s rejects wrong authority, caretaker and insufficient resources', name => {
    const cost = name === 'unions_sovereignty' ? { party_resources: 0 } : { government_budget: 0 };
    for (const overrides of [{ bnl_minister_party: 'psc' }, { cat_caretaker_gov: true }, cost]) {
      const { adapter, q } = setup(overrides);
      const before = [q.government_budget, q.party_resources, q.psc_relations, q.erc_left_dissent];
      action(adapter, name);
      expect([q.bnl_icf_level, q.bnl_inspection_level, q.bnl_tourism_level, q.bnl_union_level]).toEqual([0, 0, 0, 0]);
      expect([q.government_budget, q.party_resources, q.psc_relations, q.erc_left_dissent]).toEqual(before);
    }
  });

  it.each(['jxsi', 'jxcat'])('recognises %s control and doubles reactions for its other cabinet constituents', list => {
    const { adapter, q } = setup({ [`player_in_${list}`]: true, [`erc_in_${list}`]: true,
      [`cup_in_${list}`]: true, cat_coalition: [list], bnl_minister_party: list, economy_minister_party: list });
    action(adapter, 'inspection1');
    expect(q.bnl_inspection_level).toBe(1);
    expect(q.cup_relations).toBe(54);
    expect(q.erc_relations).toBe(50);
    action(adapter, 'icf1');
    action(adapter, 'icf_guarantees');
    expect(q.bnl_icf_prepared_branch).toBe('guarantees');
  });

  it('adds inspection capacity without resetting the first ramp or farming reactions', () => {
    const { adapter, q } = setup({ player_party: 'cup', bnl_minister_party: 'cup' });
    action(adapter, 'inspection1');
    maturity(q, 2);
    action(adapter, 'inspection2');
    expect(q.active_mods.bnl_inspection1.ticks_active).toBe(2);
    expect(q.cup_endavant_dissent).toBe(1);
    const resources = q.government_budget;
    action(adapter, 'inspection2');
    expect(q.government_budget).toBe(resources);
    expect(q.cup_endavant_dissent).toBe(1);
    maturity(q, 6);
    expect(contribution(q, 'welfare_index_growth')).toBeCloseTo(0.12);
    expect(contribution(q, 'social_dissent_eq')).toBe(-2);
    expect(lib.getLawsForUI(q).filter((law: any) => law.id.startsWith('bnl_inspection')).map((law: any) => law.id))
      .toEqual(['bnl_inspection2']);
  });

  it('applies the nonlinear tourism curve, its bounds and neutral political round trips', () => {
    const { adapter, q } = setup({ player_party: 'cup', bnl_minister_party: 'cup' });
    const targets: Record<string, any> = {
      '-2': { gdp_growth: -0.20, welfare_index_growth: 0.18, social_dissent_eq: -3 },
      '-1': { gdp_growth: -0.08, welfare_index_growth: 0.10, social_dissent_eq: -1.5 },
      '1': { gdp_growth: 0.15, welfare_index_growth: -0.15, social_dissent_eq: 2 },
      '2': { gdp_growth: 0.18, welfare_index_growth: -0.35, social_dissent_eq: 4 },
    };
    for (const name of ['more', 'more', 'less', 'less', 'less', 'less', 'more', 'more']) {
      action(adapter, `tourism_${name}`);
      const level = q.bnl_tourism_level;
      if (level) {
        expect(q.active_mods.bnl_tourism.def.targets).toEqual(targets[level]);
        expect(q.active_mods.bnl_tourism.def.icon).toBe('img/tourism_icon.svg');
      } else expect(q.active_mods.bnl_tourism.active).toBe(false);
      if (Math.abs(level) === 2) {
        const resources = q.government_budget;
        action(adapter, `tourism_${level > 0 ? 'more' : 'less'}`);
        expect(q.government_budget).toBe(resources);
      }
    }
    expect(q.cup_endavant_dissent).toBe(10);
    expect(q.cup_thirdsector_dissent).toBe(10);
    expect(q.ciu_relations).toBe(50);
    expect(q.icv_relations).toBe(50);
  });

  it('keeps union preference hidden, inert, bounded and politically reversible', () => {
    const { adapter, q } = setup({ player_party: 'cup', bnl_minister_party: 'cup' });
    for (let i = 0; i < 2; i++) action(adapter, 'unions_sovereignty');
    expect(q.bnl_union_level).toBe(2);
    expect(q.cup_endavant_dissent).toBe(4);
    expect(q.cup_poblelliure_dissent).toBe(2);
    expect(q.active_mods.bnl_union_policy.def.targets).toEqual({});
    expect(lib.getLawsForUI(q).some((law: any) => law.id === 'bnl_union_policy')).toBe(false);
    const resources = q.government_budget;
    action(adapter, 'unions_sovereignty');
    expect(q.government_budget).toBe(resources);
    for (let i = 0; i < 4; i++) action(adapter, 'unions_established');
    expect(q.bnl_union_level).toBe(-2);
    const lowerResources = q.government_budget;
    action(adapter, 'unions_established');
    expect(q.government_budget).toBe(lowerResources);
    for (let i = 0; i < 2; i++) action(adapter, 'unions_sovereignty');
    expect(q.bnl_union_level).toBe(0);
    expect(q.active_mods.bnl_union_policy.active).toBe(false);
    expect(q.cup_poblelliure_dissent).toBe(10);
    expect(q.psc_relations).toBe(50);
  });

  it('reacts to current party successors and preserves policy lifecycle on repeated setter calls', () => {
    const { adapter, q } = setup({ parlament_current_ciu: 'junts', parlament_current_icv: 'ecp' });
    action(adapter, 'tourism_less');
    expect(q.junts_relations).toBe(48);
    expect(q.ecp_relations).toBe(52);
    expect(q.ciu_relations).toBe(50);
    expect(q.icv_relations).toBe(50);
    q.active_mods.bnl_tourism.ticks_active = 7;
    const logLength = q.mod_log.length;
    adapter.goToScene('generalitat_setpolicy');
    expect(q.active_mods.bnl_tourism.ticks_active).toBe(7);
    expect(q.mod_log.length).toBe(logLength);
  });
});

describe('ICF budget approval', () => {
  it.each(['guarantees', 'cooperative'])('funds only the selected %s branch with two fiscal points and no repeated reactions', branch => {
    const { adapter, q } = setup();
    action(adapter, 'icf1');
    maturity(q, 2);
    action(adapter, `icf_${branch}`);
    const dissent = q.erc_left_dissent;
    const relations = q.ciu_relations;
    const resources = q.government_budget;
    action(adapter, `icf_${branch === 'guarantees' ? 'cooperative' : 'guarantees'}`);
    expect(q.government_budget).toBe(resources);
    expect(q.active_mods.bnl_icf_proposal.def.targets).toEqual({});
    expect(q.bnl_icf_level).toBe(1);
    submit(adapter, false);
    const margin = q.parlament_budget_margin;
    adapter.goToScene('parlament_budget.icf');
    expect(q.parlament_budget_margin).toBeCloseTo(margin - 2);
    expect(q.parlament_budget_can_pass).toBe(true);
    adapter.goToScene('parlament_budget.pass');
    expect(q.bnl_icf_level).toBe(2);
    expect(q.bnl_icf_branch).toBe(branch);
    expect(q.bnl_icf_prepared_branch).toBe(null);
    expect(q.active_mods.bnl_icf_proposal.active).toBe(false);
    expect(q.active_mods.bnl_icf1.hidden).toBe(true);
    expect(q.active_mods.bnl_icf1.ticks_active).toBe(2);
    expect(q.erc_left_dissent).toBe(dissent);
    expect(q.ciu_relations).toBe(relations);
    maturity(q, 6);
    expect(contribution(q, 'gdp_growth')).toBeCloseTo(branch === 'guarantees' ? 0.20 : 0.12);
    expect(contribution(q, 'unemployment_recovery')).toBeCloseTo(branch === 'guarantees' ? 0.04 : 0.03);
    expect(contribution(q, 'welfare_index_growth')).toBeCloseTo(branch === 'guarantees' ? 0 : 0.06);
    expect(contribution(q, 'generalitat_surplus')).toBeCloseTo(-0.01); // Lending -0.03, High taxes +0.02.
  });

  it('retains the proposal after omission or withdrawal and rejects funding after portfolio loss', () => {
    const { adapter, q } = setup();
    action(adapter, 'icf1');
    action(adapter, 'icf_cooperative');
    submit(adapter, true);
    adapter.goToScene('parlament_budget.icf');
    adapter.goToScene('parlament_budget.pass');
    expect(q.bnl_icf_level).toBe(1);
    expect(q.active_mods.bnl_icf_proposal.active).toBe(true);
    q.year += 1;
    submit(adapter, true);
    adapter.goToScene('parlament_budget.walk_away');
    expect(q.bnl_icf_prepared_branch).toBe('cooperative');
    expect(q.active_mods.bnl_icf_proposal.active).toBe(true);
    q.economy_minister_party = 'psc';
    adapter.goToScene('parlament_budget.review');
    expect(q.parlament_budget_icf_available).toBe(false);
    expect(q.parlament_budget_projection.valid).toBe(false);
    expect(q.parlament_budget_can_pass).toBe(false);
    adapter.goToScene('parlament_budget.icf');
    expect(q.budget_draft.projects.icf).toBe(false);
  });

  it('gives CUP the stronger cooperative approval and carries funding through an NPC proposal', () => {
    const { adapter, q } = setup({ player_party: 'cup', bnl_minister_party: 'cup', economy_minister_party: 'cup',
      president_party: 'cup', cat_coalition: ['cup'], erc_parlament_s: 0, cup_parlament_s: 70,
      tax_policy_level: 2, social_spending_cap: 1 });
    action(adapter, 'icf1');
    action(adapter, 'icf_cooperative');
    expect(q.cup_endavant_dissent).toBe(4);
    expect(q.cup_poblelliure_dissent).toBe(7);
    expect(q.cup_thirdsector_dissent).toBe(4);
    adapter.goToScene('budget_trigger');
    expect(q.budget_draft.projects.icf).toBe(true);
    expect(q.parlament_budget_icf_available).toBe(true);
    expect(q.parlament_budget_margin).toBe(-3); // +1 base, -2 cap expansion, -2 ICF.
  });

  it('keeps an unfunded branch inert when parliamentary support is insufficient', () => {
    const { adapter, q } = setup({ erc_parlament_s: 10, ppc_parlament_s: 125 });
    action(adapter, 'icf1');
    action(adapter, 'icf_guarantees');
    submit(adapter, true);
    expect(q.parlament_budget_can_pass).toBe(false);
    const choice = adapter.goToScene('parlament_budget.review').choices.find(c => c.id.endsWith('.pass'))!;
    expect(choice.canChoose).toBe(false);
    adapter.goToScene('parlament_budget.walk_away');
    expect(q.bnl_icf_level).toBe(1);
    expect(q.bnl_icf_prepared_branch).toBe('guarantees');
    expect(q.active_mods.bnl_icf_proposal.def.targets).toEqual({});
    expect(q.active_mods.bnl_icf2).toBeUndefined();
    expect(q.budget_year).toBe(2013);
  });

  it('rechecks ICF authority before a stale approval can charge resources or enact the budget', () => {
    const { adapter, q } = setup();
    action(adapter, 'icf1');
    action(adapter, 'icf_guarantees');
    submit(adapter, true);
    expect(q.parlament_budget_can_pass).toBe(true);
    q.bnl_minister_party = 'psc';
    const resources = [q.government_budget, q.party_resources];
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      adapter.goToScene('parlament_budget.pass');
      expect(errors.mock.calls.flat().join(' ')).toContain('ICF budget funding conditions changed');
    } finally {
      errors.mockRestore();
    }
    expect([q.government_budget, q.party_resources]).toEqual(resources);
    expect(q.bnl_icf_level).toBe(1);
    expect(q.budget_year).toBe(2013);
  });
});
