import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
const scene = 'government_does_things';
const offices = ['president_party', 'vicepresident_party', 'economy_minister_party', 'health_minister_party',
  'education_minister_party', 'bnl_minister_party', 'territory_minister_party', 'foreign_minister_party', 'interior_minister_party'];

function setup(overrides: Record<string, any> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  Object.assign(q, Object.fromEntries(offices.map(field => [field, 'erc'])), {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    erc_in_jxsi: false, erc_in_jxcat: false, cup_in_jxsi: false, cup_in_jxcat: false,
    cat_coalition: ['ciu', 'erc'], cat_coalition_support: [], cat_coalition_abstain: [],
    president_party: 'ciu', health_minister_party: 'ciu', president: 'Artur Mas',
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    ciu_ideology: 'Liberal-Conservatism, Pact-Pragmatism', ciu_relations: 60,
    year: 2013, month: 4, week: 1, time: 8, budget_year: 2013, budget_year_attempted: 2013,
    cat_caretaker_gov: false, rubicon: false, party_resources: 4, government_budget: 7,
    generalitat_surplus: -1, social_dissent: 55, welfare_index: 70, unemployment: 14,
    healthcare_spending: -1, education_spending: -1, public_salary_level: 0, social_spending_cap: 0,
    healthcare_prepared_level: null, education_prepared_level: null, public_salary_prepared_level: null,
    coalition_policy_commitments: [], coalition_indy_commitments: [], active_mods: {}, mod_log: [],
    government_does_things_timer: 0, npc_government_initiative: null, budget_draft: null,
    budget_draft_origin: null, budget_parlament_trigger: false, budget_postponed_trigger: false,
    budget_postponed_countdown: 0, economy_cuts: 0, consultation_pending: false,
    burdens_goal: 4, burdens_time: 12, burdens_done: 0, burdens_broken: false,
    ...overrides,
  });
  for (const party of q.parties) for (const province of q.parlament_constituencies)
    for (const demo of q.parlament_demographics) q[`${party}_parlament_${province}_${demo}_support`] = 4;
  for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
    q[`erc_parlament_${province}_${demo}_support`] = 20;
    q[`ciu_parlament_${province}_${demo}_support`] = 20;
    q[`cup_parlament_${province}_${demo}_support`] = 10;
    q[`abstain_parlament_${province}_${demo}_support`] = 20;
  }
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  function candidates() {
    const found = new Map<string, any>();
    const saved = q.npc_government_initiative;
    const timer = q.government_does_things_timer;
    const random = vi.spyOn(gameLib, 'random');
    for (let step = 0; step < 100; step++) {
      q.npc_government_initiative = null;
      q.government_does_things_timer = 0;
      random.mockReturnValue((step + 0.5) / 100);
      adapter.goToScene(scene);
      expect(errors.mock.calls.filter(call => String(call[0]).includes('action'))).toEqual([]);
      const item = q.npc_government_initiative;
      if (item) found.set(item.party + item.id, structuredClone(item));
    }
    random.mockRestore();
    q.npc_government_initiative = saved;
    q.government_does_things_timer = timer;
    return [...found.values()];
  }
  return { adapter, q, candidates, errors };
}

function due(f: ReturnType<typeof setup>) {
  f.q.government_does_things_timer = 0;
  f.adapter.goToScene(scene);
  expect(f.errors.mock.calls.filter(call => String(call[0]).includes('action'))).toEqual([]);
  return f.q.npc_government_initiative;
}

function propose(f: ReturnType<typeof setup>, field: string, to?: any, party?: string) {
  const candidate = f.candidates().find(item => item.field === field && (to === undefined || item.to === to) && (!party || item.party === party));
  expect(candidate, 'available ' + field + ' initiative').toBeDefined();
  f.q.npc_government_initiative = candidate;
  f.q.government_does_things_timer = 0;
  return f.adapter.goToScene(scene);
}

function answer(f: ReturnType<typeof setup>, name: string) {
  const frame = f.adapter.goToScene(scene);
  const index = frame.choices.findIndex(c => c.id === `${scene}.${name}`);
  expect(index).toBeGreaterThanOrEqual(0);
  expect(frame.choices[index].canChoose).toBe(true);
  f.adapter.choose(index);
  expect(f.errors.mock.calls.filter(call => String(call[0]).includes('action'))).toEqual([]);
}

afterEach(() => vi.restoreAllMocks());

describe('NPC cabinet scheduling', () => {
  it('selects only on arrival, preserves a proposal across save/load and uses no scheduling scratch state', () => {
    const f = setup();
    const frame = f.adapter.goToScene('post_event');
    expect(frame.choices.some(c => c.id === scene)).toBe(true);
    expect(f.q.npc_government_initiative).toBeNull();
    due(f);
    const proposal = structuredClone(f.q.npc_government_initiative);
    const loaded = DendryAdapter.fromJSONText(gameText);
    loaded.importStateJSON(f.adapter.exportStateJSON());
    loaded.goToScene(scene);
    expect((loaded.qualities as any).npc_government_initiative).toEqual(proposal);
    for (const key of ['candidates','owners','cabinet','controlled','signature','live_signature','parties'])
      expect(f.q['npc_government_' + key]).toBeUndefined();
  });
  it('uses the same event during Rubicon', () => {
    const f = setup({ rubicon:true });
    expect(due(f)?.party).toBe('ciu');
    answer(f, 'accept');
    expect(f.q.healthcare_spending).toBe(-2);
  });
  it('hides for caretakers and player-only cabinets', () => {
    for (const overrides of [{cat_caretaker_gov:true}, {cat_coalition:['erc'], president_party:'erc', health_minister_party:'erc'}]) {
      const f = setup(overrides);
      const frame = f.adapter.goToScene('post_event');
      expect(frame.choices.some(c => c.id === scene)).toBe(false);
    }
  });
  it.each(['jxsi', 'jxcat'])('allows %s initiatives while retaining player ministry access and influence', list => {
    const f = setup({cat_coalition:[list], ...Object.fromEntries(offices.map(field => [field,list])),
      ['player_in_' + list]:true, ['erc_in_' + list]:true, ['cdc_in_' + list]:true,
      [list + '_ideology']:'National Coalition, Pragmatic Independence', [list + '_relations']:65,
      cdc_ideology:'Liberal-Conservatism, Pact-Pragmatism', jxsi_formation_num:40, jxcat_leader_party:'cdc'});
    const newspaper = f.adapter.goToScene('post_event.events_choice');
    expect(newspaper.choices.some(c => c.id === scene)).toBe(true);
    expect(f.adapter.goToScene('generalitat_health_card').sceneId).toBe('generalitat_health_card');
    propose(f, 'healthcare_spending', -2, list);
    expect(f.q.npc_government_can_influence).toBe(true);
    answer(f, 'stop');
    expect(f.q.party_resources).toBe(3);
    expect(f.q.healthcare_spending).toBe(-1);
    propose(f, 'healthcare_spending', -2, list);
    answer(f, 'accept');
    expect(f.q.healthcare_spending).toBe(-2);
    expect(f.q.active_mods.healthcare_copaid.active).toBe(true);
  });
  it('falls back and restarts the timer when the selected minister has no applicable proposal', () => {
    const f = setup({health_minister_party:'erc'});
    expect(due(f)).toBeNull();
    expect(f.q.government_does_things_timer).toBeGreaterThanOrEqual(8);
    expect(f.adapter.engine.state.sceneId).toBe(scene + '.fallback');
  });
  it('selects among NPC cabinet parties and excludes formal supporters and the player', () => {
    const f = setup({cat_coalition:['erc','psc','icv'],cat_coalition_support:['cup'],
      president_party:'psc',health_minister_party:'psc',education_minister_party:'icv',
      psc_ideology:'Social Democracy, Federalism',icv_ideology:'Green Left, Plurinationalism'});
    expect(new Set(f.candidates().map(item => item.party))).toEqual(new Set(['psc','icv']));
  });
  it('resolves successor aliases and independent appointments to the president', () => {
    const f = setup({cat_coalition:['ciu','erc'],parlament_current_ciu:'cdc',president_party:'ciu',
      health_minister_party:'indp.',cdc_ideology:'Liberal-Conservatism, Pact-Pragmatism'});
    expect(due(f)).toMatchObject({party:'cdc',holder:'health_minister_party',field:'healthcare_spending'});
  });
  it('keeps a feasible pending proposal when only its selection weight changes', () => {
    const f = setup();
    due(f);
    f.q.generalitat_surplus = 1;
    answer(f, 'accept');
    expect(f.q.healthcare_spending).toBe(-2);
  });
  it('repeated synchronization preserves law ramps and does not award rewards again', () => {
    const f = setup({cat_coalition:['cdc'], ...Object.fromEntries(offices.map(field => [field,'cdc'])),
      parlament_current_ciu:'cdc',cdc_ideology:'Centre-right Liberalism, Pragmatic Independence'});
    propose(f, 'economy_atc_level', 1);
    answer(f, 'accept');
    (gameLib as any).engineTick(f.q);
    const law = structuredClone(f.q.active_mods.economy_atc1);
    const votes = f.q.cdc_parlament_barcelona_middle_support;
    const burdens = f.q.burdens_done;
    f.adapter.goToScene('generalitat_setpolicy');
    f.adapter.goToScene('generalitat_setpolicy');
    expect(f.q.active_mods.economy_atc1).toEqual(law);
    expect(f.q.cdc_parlament_barcelona_middle_support).toBe(votes);
    expect(f.q.burdens_done).toBe(burdens);
  });
  it('delivers the initiative in the newspaper after an actual monthly tick', () => {
    const f = setup();
    f.q.government_does_things_timer = 1;
    f.q.cup_leadership_timer = 30;
    f.q.month_actions = 1;
    const frame = f.adapter.goToScene('post_event');
    expect(f.q.month, JSON.stringify(frame.choices.map(c => c.id))).toBe(5);
    expect(f.q.npc_government_initiative).toBeNull();
    expect(frame.choices.some(c => c.id === scene)).toBe(true);
    const proposal = structuredClone(f.q.npc_government_initiative);
    const timer = f.q.government_does_things_timer;
    f.adapter.goToScene('post_event');
    expect(f.q.month).toBe(5);
    expect(f.q.government_does_things_timer).toBe(timer);
    expect(f.q.npc_government_initiative).toEqual(proposal);
    expect(f.errors.mock.calls.filter(call => String(call[0]).includes('action'))).toEqual([]);
  });

  it('preserves unread dated news before an initiative can advance the month', () => {
    const f = setup({ month: 1, time: 5 });
    f.q.government_does_things_timer = 1;
    f.q.month_actions = 1;
    const frame = f.adapter.goToScene('post_event');
    expect(f.q.month).toBe(1);
    expect(f.q.government_does_things_timer).toBe(1);
    expect(frame.choices.some(c => c.id === 'barcenas_leak')).toBe(true);
    expect(f.q.npc_government_initiative).toBeNull();
  });
});

describe('NPC ideology, portfolios and agreements', () => {
  it.each([[-0.01, 2], [-0.5, 2], [-1, 4], [-2.3, 9.2], [-5, 20]])
  ('weights CiU service and pay cuts at surplus %s with weight %s', (surplus, weight) => {
    const f = setup({ education_minister_party: 'ciu', economy_minister_party: 'ciu',
      generalitat_surplus: surplus });
    const cuts = f.candidates().filter(item => item.kind === 'cut');
    expect(cuts.map(item => item.field).sort()).toEqual([
      'education_spending', 'healthcare_spending', 'public_salary_level',
    ]);
    for (const cut of cuts) expect(cut.weight).toBeCloseTo(weight);
  });

  it('scales moderate-left cuts while retaining their stricter deficit threshold', () => {
    const f = setup({ ciu_ideology: 'Social Democracy, Federalism', generalitat_surplus: -2 });
    expect(f.candidates().some(item => item.kind === 'cut')).toBe(false);
    f.q.generalitat_surplus = -3;
    const cut = f.candidates().find(item => item.kind === 'cut');
    expect(cut).toMatchObject({ field: 'healthcare_spending', to: -2, weight: 3 });
  });

  it.each([0, 0.5])('does not propose CiU cuts without a deficit at surplus %s', surplus => {
    const f = setup({ generalitat_surplus: surplus });
    expect(f.candidates().some(item => item.kind === 'cut')).toBe(false);
  });

  it('does not reopen a budget already attempted this year', () => {
    const f = setup({ economy_minister_party: 'ciu', budget_year: 2012, budget_year_attempted: 2013,
      ciu_ideology: 'Social Democracy, Federalism' });
    expect(f.candidates().some((item: any) => item.kind === 'budget')).toBe(false);
  });
  it('uses a party’s changed ideology and only its own ministry', () => {
    const f = setup();
    expect(f.candidates().map((item: any) => item.field)).toEqual(['healthcare_spending']);
    f.q.ciu_ideology = 'Social Democracy, Federalism';
    expect(f.candidates().some((item: any) => item.field === 'healthcare_spending' && item.to === 0)).toBe(true);
    expect(f.candidates().some((item: any) => item.kind === 'cut')).toBe(false);
    expect(f.candidates().every((item: any) => f.q[item.holder] === 'ciu')).toBe(true);
  });

  it.each(['cabinet', 'external_support'])('excludes cuts breaching a %s agreement and postpones empty agendas', scope => {
    const f = setup({ coalition_policy_commitments: [{ kind: 'healthcare_spending', target: 0, scope,
      signatories: ['ciu', 'erc'], status: 'pending_budget' }] });
    expect(due(f)).toBeNull();
    expect(f.q.government_does_things_timer).toBeGreaterThanOrEqual(8);
    expect(f.q.healthcare_spending).toBe(-1);
  });

  it('keeps socioeconomic and territorial preferences independent', () => {
    const f = setup({ cat_coalition: ['psc'], ...Object.fromEntries(offices.map(field => [field, 'psc'])),
      psc_ideology: 'Anticapitalism, Centralism' });
    expect(f.candidates().some((item: any) => item.kind === 'inspection')).toBe(true);
    expect(f.candidates().some((item: any) => item.field === 'bnl_union_level' && item.to === 1)).toBe(false);
    f.q.psc_ideology = 'Anticapitalism, Unilateralism';
    expect(f.candidates().some((item: any) => item.field === 'bnl_union_level' && item.to === 1)).toBe(true);
  });

  it.each([
    [0, 85, true], [1, 59, true], [1, 60, false], [1, 85, false],
    [2, 20, false], [3, 20, false],
  ])('gates Madrid diplomacy at roadmap %i and momentum %i: %s', (roadmap, momentum, allowed) => {
    const f = setup({ foreign_minister_party: 'ciu', ciu_roadmap: roadmap,
      independence_movement: momentum, cat_spa_relations: 20, art155_ever: false });
    expect(f.candidates().some(item => item.kind === 'diplomacy')).toBe(allowed);
  });

  it('offers a broad existing policy catalogue with capped ordinary service transitions', () => {
    const f = setup({ cat_coalition: ['icv'], ...Object.fromEntries(offices.map(field => [field, 'icv'])),
      healthcare_spending: 2, education_spending: 2, social_spending_cap: 3, generalitat_surplus: 0,
      icv_ideology: 'Green Left, Plurinationalism', parlament_lgbt_passed: true });
    expect(new Set(f.candidates().map((item: any) => item.field)).size).toBeGreaterThan(10);
    expect(f.candidates().some((item: any) => ['education_spending', 'healthcare_spending'].includes(item.field) && item.to === 3)).toBe(false);
    expect(f.candidates().some((item: any) => item.to === 'segregation')).toBe(true);
  });
});

describe('NPC initiative responses and enactment', () => {
  it('credits the real separate PSC ballot in an entirely NPC government', () => {
    const f = setup({ cat_coalition: ['psc'], ...Object.fromEntries(offices.map(field => [field, 'psc'])),
      psc_ideology: 'Social Democracy, Federalism', psc_split: true,
      congreso_parties_catalunya: ['psc', 'psoe'], congreso_constituencies: ['catalunya'],
      psc_congreso_catalunya_support: 25, psoe_congreso_catalunya_support: 0, abstain_congreso_catalunya_support: 25 });
    const playerBefore = f.q.erc_parlament_barcelona_young_support;
    propose(f, 'healthcare_spending', 0);
    answer(f, 'accept');
    expect(f.q.psc_congreso_catalunya_support).toBeGreaterThan(25);
    expect(f.q.psoe_congreso_catalunya_support).toBe(0);
    expect(f.q.erc_parlament_barcelona_young_support).toBe(playerBefore);
  });
  it('accepts an actual cut, applies the policy modifiers and costs no executive resources', () => {
    const f = setup();
    due(f);
    answer(f, 'accept');
    expect(f.q.healthcare_spending).toBe(-2);
    expect(f.q.active_mods.healthcare_copaid.active).toBe(true);
    expect(f.q.government_budget).toBe(7);
    expect(f.q.party_resources).toBe(4);
    expect(f.q.burdens_broken).toBe(true);
  });

  it('protest halves only the player’s direct losses while keeping policy effects and NPC losses', () => {
    const raw = setup();
    due(raw);
    answer(raw, 'accept');
    const protested = setup();
    due(protested);
    answer(protested, 'protest');
    expect(protested.q.healthcare_spending).toBe(raw.q.healthcare_spending);
    expect(protested.q.active_mods.healthcare_copaid.def.targets).toEqual(raw.q.active_mods.healthcare_copaid.def.targets);
    expect(20 - protested.q.erc_parlament_barcelona_young_support).toBeCloseTo((20 - raw.q.erc_parlament_barcelona_young_support) / 2);
    expect(protested.q.ciu_parlament_barcelona_young_support).toBeCloseTo(raw.q.ciu_parlament_barcelona_young_support);
    expect(protested.q.ciu_relations).toBe(55);
    expect(protested.q.coalition_rupture_recheck).toBe(true);
  });

  it.each([[55, 2], [64, 2], [65, 1], [80, 1]])('stopping at %i relations costs %i resources', (relations, cost) => {
    const f = setup({ ciu_relations: relations });
    due(f);
    answer(f, 'stop');
    expect(f.q.healthcare_spending).toBe(-1);
    expect(f.q.party_resources).toBe(4 - cost);
    expect(f.q.government_budget).toBe(7);
    expect(f.q.government_does_things_timer).toBeGreaterThanOrEqual(8);
    f.adapter.goToScene(`${scene}.stop`);
    expect(f.q.party_resources).toBe(4 - cost);
  });

  it.each([{ ciu_relations: 54 }, { party_resources: 1 }, { cat_coalition: ['ciu'], cat_coalition_support: [], ciu_relations: 80 }])(
    'guards disabled stop responses at resolution: %j', overrides => {
      const f = setup(overrides);
      due(f);
      const frame = f.adapter.goToScene(scene);
      expect(frame.choices.find(c => c.id.endsWith('.stop'))?.canChoose).toBe(false);
      const before = f.q.party_resources;
      f.adapter.goToScene(`${scene}.stop`);
      expect(f.q.party_resources).toBe(before);
      expect(f.q.healthcare_spending).toBe(-1);
    });

  it('allows formal external support to stop an action but excludes abstention', () => {
    const f = setup({ cat_coalition: ['ciu'], cat_coalition_support: ['erc'] });
    due(f);
    answer(f, 'stop');
    const abstain = setup({ cat_coalition: ['ciu'], cat_coalition_abstain: ['erc'] });
    due(abstain);
    abstain.adapter.goToScene(scene);
    expect(abstain.q.npc_government_can_stop).toBe(false);
  });

  it('redirects a compatible NPC’s cut into a real budget proposal without enacting taxes early', () => {
    const f = setup({ cat_coalition: ['psc', 'erc'], president_party: 'psc', economy_minister_party: 'psc',
      health_minister_party: 'psc', psc_ideology: 'Centrist Social Democracy, Federalism', psc_relations: 60,
      generalitat_surplus: -2.5, budget_year: 2012, budget_year_attempted: 2012, tax_policy_level: 0 });
    propose(f, 'healthcare_spending', -2);
    answer(f, 'alternative');
    expect(f.q.healthcare_spending).toBe(-1);
    expect(f.q.tax_policy_level).toBe(0);
    expect(f.q.budget_draft.tax_policy_level).toBe(1);
    expect(f.q.budget_draft_origin.proposer_party).toBe('psc');
    expect(f.q.government_budget).toBe(7);
    const frame = f.adapter.goToScene(`${scene}.resolve`);
    // Repeated resolution cannot submit or enact a second initiative.
    expect(f.q.tax_policy_level).toBe(0);
    expect(frame.html).toBeDefined();
  });

  it('rechecks an agreement or lost ministry before charging or enacting', () => {
    for (const change of ['agreement', 'authority']) {
      const f = setup();
      due(f);
      f.adapter.goToScene(scene);
      if (change === 'agreement') f.q.coalition_policy_commitments = [{ kind: 'healthcare_spending', target: -1, status: 'applied' }];
      else f.q.health_minister_party = 'erc';
      f.adapter.goToScene(`${scene}.protest`);
      expect(f.q.healthcare_spending).toBe(-1);
      expect(f.q.ciu_relations).toBe(60);
      expect(f.q.npc_government_initiative).toBeNull();
    }
  });

  it('enacts agricultural diversification through the shared setter', () => {
    const f = setup({ cat_coalition: ['icv'], ...Object.fromEntries(offices.map(field => [field, 'icv'])) });
    propose(f, 'territory_livestock', -1);
    answer(f, 'accept');
    expect(f.q.territory_livestock).toBe(-1);
    expect(f.q.active_mods.territory_livestock.def.targets.welfare_index_growth).toBe(0.025);
  });

  it('preserves the ATC baseline and requires the real LGBTI prerequisite', () => {
    const f = setup({ cat_coalition: ['cdc'], ...Object.fromEntries(offices.map(field => [field, 'cdc'])),
      parlament_current_ciu: 'cdc', cdc_ideology: 'Centre-right Liberalism, Pragmatic Independence',
      independence_movement: 80, independence_trust: 70 });
    propose(f, 'economy_atc_level', 1);
    answer(f, 'accept');
    (gameLib as any).engineTick(f.q);
    propose(f, 'economy_atc_level', 2);
    answer(f, 'accept');
    expect(f.q.active_mods.economy_atc1.hidden).toBe(true);
    expect(f.q.active_mods.economy_atc1.ticks_active).toBe(1);
    expect(f.q.independence_trust).toBeGreaterThan(60);
    f.q.cdc_ideology = 'Social Democracy, Independence';
    f.q.healthcare_sexual_level = 1;
    expect(f.candidates().some((item: any) => item.field === 'healthcare_sexual_level')).toBe(false);
    f.q.parlament_lgbt_passed = true;
    expect(f.candidates().some((item: any) => item.field === 'healthcare_sexual_level' && item.to === 2)).toBe(true);
  });

  it('keeps only the realized current tourism package when policy returns to neutral', () => {
    const f = setup({ cat_coalition: ['icv'], ...Object.fromEntries(offices.map(field => [field, 'icv'])),
      icv_ideology: 'Green Left, Plurinationalism', bnl_tourism_level: 0 });
    const initial = f.q.icv_parlament_barcelona_young_support;
    propose(f, 'bnl_tourism_level', -1);
    answer(f, 'accept');
    expect(f.q.bnl_tourism_electoral_routes.length).toBeGreaterThan(0);
    f.q.icv_ideology = 'Centre-right Liberalism, Plurinationalism';
    propose(f, 'bnl_tourism_level', 0);
    answer(f, 'accept');
    expect(f.q.icv_parlament_barcelona_young_support).toBeCloseTo(initial);
    expect(f.q.bnl_tourism_electoral_routes).toEqual([]);
  });

  it('revises the curriculum without restarting its ramp and records the current electoral package', () => {
    const f = setup({ cat_coalition: ['icv'], ...Object.fromEntries(offices.map(field => [field, 'icv'])),
      icv_ideology: 'Green Left, Plurinationalism', education_spending: 1 });
    propose(f, 'education_policy_main', 'democratic');
    answer(f, 'accept');
    for (let i = 0; i < 3; i++) (gameLib as any).engineTick(f.q);
    f.q.education_policy_next_revision_month = 0;
    propose(f, 'education_policy_main', 'language');
    answer(f, 'accept');
    expect(f.q.active_mods.education_school_policy.ticks_active).toBe(3);
    expect(f.q.active_mods.education_school_policy.def.electoral_transfers.length).toBeGreaterThan(0);
    expect(f.q.education_policy_secondary).toBe('humanities');
  });

  it('carries protest into a passed NPC tax budget and leaves other parties’ losses unchanged', () => {
    function pass(response: string) {
      const f = setup({ cat_coalition: ['psc', 'erc'], president_party: 'psc', economy_minister_party: 'psc',
        health_minister_party: 'psc', psc_ideology: 'Social Democracy, Federalism', psc_relations: 60,
        generalitat_surplus: -0.5, budget_year: 2012, budget_year_attempted: 2012, tax_policy_level: 0 });
      for (const party of f.q.parties) f.q[`${party}_parlament_s`] = 0;
      Object.assign(f.q, { psc_parlament_s: 70, erc_parlament_s: 30, ppc_parlament_s: 35 });
      propose(f, 'tax_policy_level', 1);
      answer(f, response);
      const result = f.adapter.engine.getCurrentChoices()!;
      f.adapter.choose(result.findIndex((c: any) => c.id.endsWith('.review_budget')));
      f.adapter.goToScene('parlament_budget.vote_yes');
      expect(f.q.parlament_budget_can_pass).toBe(true);
      f.adapter.goToScene('parlament_budget.pass');
      expect(f.errors.mock.calls.filter(call => String(call[0]).includes('action'))).toEqual([]);
      expect(f.q.tax_policy_level).toBe(1);
      expect(f.errors.mock.calls.filter(call => String(call[0]).includes('action'))).toEqual([]);
      return f.q;
    }
    const raw = pass('accept');
    const protested = pass('protest');
    expect(20 - protested.erc_parlament_barcelona_middle_support).toBeCloseTo((20 - raw.erc_parlament_barcelona_middle_support) / 2);
    expect(protested.psc_parlament_barcelona_middle_support).toBeCloseTo(raw.psc_parlament_barcelona_middle_support);
  });
});
