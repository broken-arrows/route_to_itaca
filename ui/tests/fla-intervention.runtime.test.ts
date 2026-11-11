import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
const fields = ['education_spending', 'healthcare_spending', 'public_salary_level'];
const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
afterEach(() => {
  const actionErrors = errors.mock.calls.filter(([message]) => String(message).includes('Scene action error'));
  errors.mockClear();
  expect(actionErrors).toEqual([]);
});

function setup(overrides: Record<string, any> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  for (const party of q.parties) { q[party + '_parlament_s'] = 0; q[party + '_relations'] = 60; }
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    cat_coalition: ['erc'], cat_coalition_support: [], cat_coalition_abstain: [],
    president_party: 'erc', economy_minister_party: 'erc', health_minister_party: 'erc', education_minister_party: 'erc',
    cat_caretaker_gov: false, erc_parlament_s: 80, ppc_parlament_s: 55, parlament_s_majority: 68,
    year: 2014, month: 1, week: 1, rubicon: false, government_budget: 30, party_resources: 30,
    budget_year: 2013, budget_year_attempted: 2013, generalitat_surplus: 1.5,
    social_spending_cap: 2, healthcare_spending: 1, education_spending: 1, public_salary_level: 1, economy_cuts: 0,
    healthcare_public: false, healthcare_dental: false, healthcare_expansion: false,
    education_public: false, education_modernization: false, education_expansion: false,
    healthcare_prepared_level: null, education_prepared_level: null, public_salary_prepared_level: null,
    active_mods: {}, mod_log: [], coalition_policy_commitments: [],
    fla_active: true, fla_escalated: false, fla_severe: false, fla_intervention_month: null,
    public_debt: 36, spanish_pm_party: 'pp', spanish_coalition: ['pp'],
    ...overrides,
  });
  return { adapter, q };
}
function eligible(f: ReturnType<typeof setup>, id: string) {
  const scene = f.adapter.engine.game.scenes[id];
  expect(scene, id).toBeDefined();
  return f.adapter.engine._runPredicate(scene.viewIf, true);
}
function choose(f: ReturnType<typeof setup>, id: string) {
  const choices = f.adapter.engine.getCurrentChoices()!;
  const index = choices.findIndex((c: any) => c.id === id && c.canChoose);
  expect(index, id).toBeGreaterThanOrEqual(0);
  f.adapter.choose(index);
}
function canChoose(f: ReturnType<typeof setup>, id: string) {
  return f.adapter.engine.getCurrentChoices()!.some((c: any) => c.id === id && c.canChoose);
}

describe('FLA intervention funding and lifecycle', () => {
  it('reduces expansions and public-only regimes to ordinary funding, reconciling their modifiers', () => {
    const f = setup({ healthcare_spending: 3, education_spending: 3, public_salary_level: 2, social_spending_cap: 3 });
    f.adapter.goToScene('generalitat_setpolicy');
    f.adapter.engine.gameLib.engineTick(f.q);
    const surplusBefore = f.q.generalitat_surplus;
    const serviceLaws = ['healthcare_public', 'healthcare_dental', 'healthcare_expansion', 'education_public', 'education_modernization', 'education_expansion', 'public_salary_improvement', 'public_salary_expansion'];
    const appliedCost = serviceLaws.reduce((sum, id) => sum + (f.q.active_mods[id]?.fiscal_applied || 0), 0);
    f.adapter.goToScene('fla_tightening');
    expect(fields.map(field => f.q[field])).toEqual([0, 0, 0]);
    expect(f.q.social_spending_cap).toBe(0);
    expect(f.q.fla_intervention_month).toBe(2014 * 12 + 1);
    for (const id of ['healthcare_public', 'healthcare_dental', 'healthcare_expansion', 'education_public', 'education_modernization', 'education_expansion', 'public_salary_improvement', 'public_salary_expansion'])
      expect(f.q.active_mods[id]?.active, id).toBe(false);
    expect(f.q.healthcare_public).toBe(false);
    expect(f.q.education_public).toBe(false);
    expect(f.q.generalitat_surplus).toBeCloseTo(surplusBefore - appliedCost, 10);
  });

  it('does not improve deeper cuts or spend executive resources when imposing a ceiling', () => {
    const f = setup({ healthcare_spending: -3, education_spending: -2, public_salary_level: -1, social_spending_cap: -1 });
    f.adapter.goToScene('fla_tightening');
    expect(fields.map(field => f.q[field])).toEqual([-2, -3, -1]);
    expect(f.q.social_spending_cap).toBe(-1);
    expect(f.q.government_budget).toBe(30);
    expect(f.q.party_resources).toBe(30);
  });

  it.each([
    ['pp', ['pp'], true], ['pp', ['pp', 'cs'], true], ['psoe', ['psoe', 'cs'], true],
    ['cs', ['cs', 'psoe'], true], ['psoe', ['psoe', 'pp'], true],
    ['psoe', ['psoe'], false], ['psoe', ['psoe', 'podemos'], false],
    ['up', ['up', 'psoe'], false], ['podemos', ['podemos'], false], ['up', ['up'], false],
    ['sumar', ['sumar'], false], ['psoe', ['psoe', 'sumar'], false],
  ])('further tightening under %s / %j: %s', (pm, cabinet, allowed) => {
    const f = setup({ fla_escalated: true, fla_intervention_month: 2013 * 12 + 7,
      public_debt: 40, generalitat_surplus: -1.5, spanish_pm_party: pm, spanish_coalition: cabinet });
    expect(eligible(f, 'fla_further_tightening')).toBe(allowed);
  });

  it.each([
    { month: 5 }, { public_debt: 39.99 }, { generalitat_surplus: -1.49 },
    { fla_active: false }, { fla_escalated: false }, { fla_severe: true },
  ])('requires every further-tightening condition: %j', overrides => {
    const f = setup({ fla_escalated: true, fla_intervention_month: 2014 * 12,
      month: 6, public_debt: 40, generalitat_surplus: -1.5, ...overrides });
    expect(eligible(f, 'fla_further_tightening')).toBe(false);
  });

  it('enforces -1 after six calendar months and relaxes in two steps without restoring spending', () => {
    const f = setup({ month: 7, public_debt: 40, generalitat_surplus: -1.5,
      fla_escalated: true, fla_intervention_month: 2014 * 12 + 1,
      social_spending_cap: 0, healthcare_spending: 0, education_spending: 0, public_salary_level: 0 });
    expect(eligible(f, 'fla_further_tightening')).toBe(true);
    f.adapter.goToScene('fla_further_tightening');
    expect(f.q.fla_severe).toBe(true);
    expect(f.q.social_spending_cap).toBe(-1);
    expect(fields.map(field => f.q[field])).toEqual([-1, -1, -1]);
    Object.assign(f.q, { public_debt: 37, generalitat_surplus: -1.49 });
    expect(eligible(f, 'fla_further_relaxation')).toBe(true);
    expect(eligible(f, 'fla_relaxation')).toBe(false);
    f.adapter.goToScene('fla_further_relaxation');
    expect(f.q.fla_escalated).toBe(true);
    expect(f.q.fla_severe).toBe(false);
    expect(f.q.social_spending_cap).toBe(-1);
    expect(fields.map(field => f.q[field])).toEqual([-1, -1, -1]);
    f.q.public_debt = 32;
    expect(eligible(f, 'fla_relaxation')).toBe(true);
    f.adapter.goToScene('fla_relaxation');
    expect(f.q.fla_escalated).toBe(false);
    expect(f.q.social_spending_cap).toBe(-1);
  });

  it.each([{ public_debt: 37.01 }, { generalitat_surplus: -1.5 }])('does not prematurely ease the severe tier: %j', overrides => {
    const f = setup({ fla_escalated: true, fla_severe: true, public_debt: 37, generalitat_surplus: -1.49, ...overrides });
    expect(eligible(f, 'fla_further_relaxation')).toBe(false);
  });

  it('discovers and opens further tightening through the event pool, preserving its tier across saves', () => {
    const f = setup({ month: 7, public_debt: 40, generalitat_surplus: -1.5,
      fla_escalated: true, fla_intervention_month: 2014 * 12 + 1,
      social_spending_cap: 0, healthcare_spending: 0, education_spending: 0, public_salary_level: 0 });
    f.adapter.goToScene('post_event.events_choice');
    choose(f, 'fla_further_tightening');
    expect(f.q.fla_severe).toBe(true);
    const loaded = setup();
    loaded.adapter.importStateJSON(f.adapter.exportStateJSON());
    expect(loaded.adapter.qualities.fla_severe).toBe(true);
    expect(loaded.adapter.qualities.social_spending_cap).toBe(-1);
    expect(loaded.adapter.qualities.fla_intervention_month).toBe(2014 * 12 + 1);
  });
});

describe('FLA ceilings on decisions and their enactment', () => {
  it.each([false, true])('blocks ministry approval and above-cap preparation at severe=%s', severe => {
    const ceiling = severe ? -1 : 0;
    for (const [card, field, action, prepared] of [
      ['health', 'healthcare_spending', severe ? 'revert_cuts' : 'healthcare_expand', 'healthcare_prepared_level'],
      ['education', 'education_spending', severe ? 'revert_cuts' : 'education_expand', 'education_prepared_level'],
      ['economy', 'public_salary_level', 'raise_pay', 'public_salary_prepared_level'],
    ]) {
      const f = setup({ fla_escalated: true, fla_severe: severe, social_spending_cap: ceiling, [field]: ceiling });
      f.adapter.goToScene(`generalitat_${card}_card`);
      expect(canChoose(f, `generalitat_${card}_card.${action}`)).toBe(false);
      f.adapter.goToScene(`generalitat_${card}_card.${card === 'economy' ? 'raise_pay' : 'approve_raise'}`);
      expect(f.q[field]).toBe(ceiling);
      expect(f.q[prepared]).toBeNull();
      expect(f.q.government_budget).toBe(30);
    }
  });

  it('permits restoring cuts within the intervention ceiling', () => {
    const f = setup({ fla_escalated: true, social_spending_cap: 0, healthcare_spending: -1 });
    f.adapter.goToScene('generalitat_health_card');
    choose(f, 'generalitat_health_card.revert_cuts');
    expect(f.q.healthcare_spending).toBe(0);
  });

  it.each([false, true])('blocks budget authorization above the ceiling at severe=%s', severe => {
    const ceiling = severe ? -1 : 0;
    const f = setup({ fla_escalated: true, fla_severe: severe, social_spending_cap: ceiling,
      healthcare_spending: ceiling, education_spending: ceiling, public_salary_level: ceiling });
    f.adapter.goToScene('generalitat_economy_card.prepare_budget');
    expect(canChoose(f, 'generalitat_economy_card.budget_cap_higher')).toBe(false);
    f.q.economy_budget_proposal.social_spending_cap = ceiling + 1;
    f.adapter.goToScene('generalitat_economy_card.budget_review');
    expect(canChoose(f, 'generalitat_economy_card.submit_budget')).toBe(false);
  });

  it('invalidates a previously submitted higher-cap budget after intervention, including a stale passage', () => {
    const f = setup({ healthcare_spending: 0, education_spending: 0, public_salary_level: 0, social_spending_cap: 0 });
    f.adapter.goToScene('generalitat_health_card');
    choose(f, 'generalitat_health_card.healthcare_expand');
    f.adapter.goToScene('generalitat_economy_card.prepare_budget');
    choose(f, 'generalitat_economy_card.budget_cap_higher');
    choose(f, 'generalitat_economy_card.submit_budget');
    expect(f.q.parlament_budget_can_pass).toBe(true);
    f.adapter.goToScene('fla_tightening');
    f.adapter.goToScene('parlament_budget.pass');
    expect(f.q.healthcare_spending).toBe(0);
    expect(f.q.social_spending_cap).toBe(0);
    expect(f.q.budget_year).toBe(2013);
    f.adapter.goToScene('parlament_budget.review');
    expect(f.q.parlament_budget_projection.valid).toBe(false);
    expect(canChoose(f, 'parlament_budget.pass')).toBe(false);
  });

  it('restrains NPC budgets and coalition concessions', () => {
    const f = setup({ fla_escalated: true, social_spending_cap: 0, healthcare_spending: 0,
      education_spending: 0, public_salary_level: 0, tax_policy_level: 2,
      economy_minister_party: 'psc', president_party: 'psc', cat_coalition: ['psc', 'erc', 'icv'],
      psc_parlament_s: 10, icv_parlament_s: 10, icv_relations: 40 });
    f.adapter.goToScene('budget_trigger');
    expect(f.q.budget_draft.social_spending_cap).toBeLessThanOrEqual(0);
    f.adapter.goToScene('generalitat_coalition_card');
    f.adapter.goToScene('generalitat_coalition_card.concessions_icv');
    expect(fields.map(field => f.q[field])).toEqual([0, 0, 0]);
  });

  it('blocks a new coalition funding demand but allows withdrawing an old demand', () => {
    const f = setup({ fla_escalated: true, social_spending_cap: 0, education_spending: 0,
      parlament_coalition_leverage: 100, parlament_coalition_policy_spending_cost: 5,
      parlament_coalition_policy_education: false, parlament_coalition_policy_education_target: 1 });
    f.adapter.engine.state.jumpSceneId = 'generalitat_coalition_card';
    f.adapter.goToScene('parlament_coalition_fiscal_policies.education_policy');
    expect(f.q.parlament_coalition_policy_education).toBe(false);
    f.q.parlament_coalition_policy_education = true;
    f.adapter.goToScene('parlament_coalition_fiscal_policies.education_policy');
    expect(f.q.parlament_coalition_policy_education).toBe(false);
  });

  it('excludes NPC funding increases and invalidates an already selected increase', () => {
    const f = setup({ cat_coalition: ['psc', 'erc'], president_party: 'psc', health_minister_party: 'psc',
      education_minister_party: 'psc', economy_minister_party: 'psc', psc_parlament_s: 30,
      psc_ideology: 'Social Democracy, Federalism', healthcare_spending: 0, education_spending: 0,
      public_salary_level: 0, government_does_things_timer: 0, npc_government_initiative: null });
    const lib = f.adapter.engine.gameLib as any;
    const originalRandom = lib.random;
    function candidates() {
      const found = new Map<string, any>();
      try {
        for (let step = 0; step < 100; step++) {
          lib.random = () => (step + .5) / 100;
          f.q.npc_government_initiative = null;
          f.q.government_does_things_timer = 0;
          f.adapter.goToScene('government_does_things');
          const proposal = f.q.npc_government_initiative;
          if (proposal && fields.includes(proposal.field) && proposal.to > proposal.from)
            found.set(proposal.field, JSON.parse(JSON.stringify(proposal)));
        }
      } finally { lib.random = originalRandom; }
      return found;
    }
    const before = candidates();
    expect([...before.keys()].sort()).toEqual(fields.slice().sort());
    Object.assign(f.q, { fla_escalated: true, social_spending_cap: 0 });
    expect(candidates().size).toBe(0);
    f.q.npc_government_initiative = before.get('healthcare_spending');
    f.adapter.goToScene('government_does_things');
    choose(f, 'government_does_things.accept');
    expect(f.q.healthcare_spending).toBe(0);
  });

  it.each(['healthcare', 'education'])('blocks public-only %s proposals and stale votes', policy => {
    const department = policy === 'healthcare' ? 'health' : 'education';
    const f = setup({ fla_escalated: true, social_spending_cap: 3, [policy + '_spending']: 2 });
    f.adapter.goToScene('generalitat_' + department + '_card');
    expect(canChoose(f, `generalitat_${department}_card.${policy}_public`)).toBe(false);
    f.q[policy + '_public_introduced'] = true;
    f.adapter.goToScene('parlament_exec_' + policy + '_public');
    choose(f, 'parlament_exec_' + policy + '_public.vote_yes');
    expect(f.q[policy + '_spending']).toBe(2);
    f.q['parlament_' + policy + '_public_passed'] = true;
    f.adapter.goToScene('parlament_exec_' + policy + '_public.on_yes');
    expect(f.q[policy + '_spending']).toBe(2);
    expect(f.q.government_budget).toBe(30);
  });

  it.each([1, 2])('blocks delivery of a coalition promise to level %s accepted before intervention', target => {
    const f = setup({ fla_escalated: true, social_spending_cap: 0, education_spending: 0,
      parlament_coalition_proposal_president: 'erc', parlament_coalition_proposal_president_name: 'Oriol Junqueras',
      parlament_coalition_proposal_president_gender: 'm',
      parlament_coalition_policy_accepted: true, parlament_coalition_policy_education: true,
      parlament_coalition_policy_education_target: target, parlament_coalition_policy_cap: false,
      parlament_coalition_policy_health: false, parlament_coalition_policy_pay: false, parlament_coalition_policy_tax: false });
    f.adapter.goToScene('parlament_coalition_resolve');
    expect(f.q.parlament_coalition_resolution_valid).toBe(true);
    expect(f.q.education_spending).toBe(0);
    expect(f.q.coalition_policy_commitments).toEqual([]);
  });

  it('allows a compliant budget while higher promises are suspended by intervention', () => {
    const f = setup({ fla_escalated: true, social_spending_cap: 0, healthcare_spending: 0,
      education_spending: 0, public_salary_level: 0, coalition_policy_commitments: [
        {kind: 'social_spending_cap', target: 1, scope: 'cabinet', signatories: ['erc'], status: 'pending_budget'},
        {kind: 'education_spending', target: 1, scope: 'cabinet', signatories: ['erc'], status: 'applied'},
      ] });
    f.adapter.goToScene('generalitat_economy_card.prepare_budget');
    choose(f, 'generalitat_economy_card.submit_budget');
    expect(f.q.parlament_budget_can_pass).toBe(true);
    choose(f, 'parlament_budget.pass');
    expect(f.q.budget_year).toBe(2014);
    expect(f.q.social_spending_cap).toBe(0);
    expect(f.q.education_spending).toBe(0);
  });
});
