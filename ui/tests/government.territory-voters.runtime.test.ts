import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');
const provinces = ['barcelona', 'girona', 'lleida', 'tarragona'];
const groups = ['buss', 'ind', 'middle', 'young', 'retired', 'rural', 'unemployed'];
type Q = Record<string, any>;
const key = (party: string, province = 'barcelona', group = 'ind') => `${party}_parlament_${province}_${group}_support`;

function setup(overrides: Q = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Q;
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    erc_in_jxsi: false, cup_in_jxsi: false, erc_in_jxcat: false, cup_in_jxcat: false,
    jxsi_in_congreso: false, jxcat_in_congreso: false, jxsi_in_local: false, jxcat_in_local: false,
    cat_coalition: ['erc'], cat_coalition_support: [], cat_coalition_abstain: [],
    bnl_minister_party: 'erc', territory_minister_party: 'erc', economy_minister_party: 'erc',
    president_party: 'erc', cat_caretaker_gov: false, party_resources: 20, government_budget: 100,
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv', year: 2014, month: 5,
    bnl_icf_level: 0, bnl_icf_prepared_branch: null, bnl_inspection_level: 0,
    bnl_union_level: 0, bnl_tourism_level: 0,
    territory_renewables: 0, territory_livestock: 0, territory_water: 0,
    territory_tmobilitat: 0, territory_metro: 0, territory_metro_approved: 0,
    territory_roads: 1, territory_roads_approved: 0, territory_b40_proposed: false,
    territory_rail_approved: 0, active_mods: {}, mod_log: [], countdowns: [],
    parlament_climate_change_passed: true, burdens_done: 0, burdens_goal: 4, burdens_time: 12,
  });
  const stocks: Q = { erc: 8, cup: 5, ciu: 10, psc: 16, icv: 10, cs: 7, ppc: 10, abstain: 34 };
  for (const province of provinces) for (const group of groups) {
    for (const party of [...q.parties, 'abstain']) q[key(party, province, group)] = stocks[party] || 0;
    q[`parlament_${province}_${group}_pop`] = 1000;
  }
  for (const party of q.parties_bcn) q[`${party}_local_barcelona_support`] = 0;
  Object.assign(q, {
    erc_local_barcelona_support: 15, cup_local_barcelona_support: 8, ciu_local_barcelona_support: 15,
    psc_local_barcelona_support: 20, icv_local_barcelona_support: 20, cs_local_barcelona_support: 12,
    pp_local_barcelona_support: 10,
  });
  for (const province of q.congreso_constituencies) {
    for (const party of [...q[`congreso_parties_${province}`], 'abstain'])
      q[`${party}_congreso_${province}_support`] = 0;
    Object.assign(q, { [`pp_congreso_${province}_support`]: 30, [`psoe_congreso_${province}_support`]: 30,
      [`iu_congreso_${province}_support`]: 15, [`cs_congreso_${province}_support`]: 10,
      [`abstain_congreso_${province}_support`]: 15 });
  }
  q.erc_congreso_catalunya_support = 10;
  q.psoe_congreso_catalunya_support -= 10;
  Object.assign(q, overrides);
  return { adapter, q };
}

function action(adapter: DendryAdapter, scene: string) {
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  try {
    adapter.goToScene(scene);
    expect(errors.mock.calls.filter(([message]) => !String(message).startsWith('Page loaded from unexpected origin'))).toEqual([]);
  } finally { errors.mockRestore(); }
}

function support(q: Q) {
  return Object.fromEntries(Object.entries(q).filter(([k]) => /(_parlament_.*_support|_congreso_.*_support|_local_barcelona_support)$/.test(k)));
}

function checkPools(q: Q, before: Q) {
  for (const province of provinces) for (const group of groups) {
    const parties = [...q.parties, 'abstain'];
    expect(parties.reduce((n, p) => n + q[key(p, province, group)], 0))
      .toBeCloseTo(parties.reduce((n, p) => n + before[key(p, province, group)], 0), 10);
  }
  for (const province of q.congreso_constituencies) {
    const parties = [...q[`congreso_parties_${province}`], 'abstain'];
    expect(parties.reduce((n, p) => n + q[`${p}_congreso_${province}_support`], 0))
      .toBeCloseTo(parties.reduce((n, p) => n + before[`${p}_congreso_${province}_support`], 0), 10);
  }
  expect(q.parties_bcn.reduce((n: number, p: string) => n + q[`${p}_local_barcelona_support`], 0))
    .toBeCloseTo(q.parties_bcn.reduce((n: number, p: string) => n + before[`${p}_local_barcelona_support`], 0), 10);
  for (const value of Object.values(support(q))) { expect(Number.isFinite(value)).toBe(true); expect(value).toBeGreaterThanOrEqual(0); }
}

function fund(adapter: DendryAdapter, project: string) {
  const q=adapter.qualities as Q;
  Object.assign(q,{player_party:'erc',player_in_jxsi:false,player_in_jxcat:false,
    economy_minister_party:'erc',bnl_minister_party:'erc',territory_minister_party:'erc',president_party:'erc',cat_coalition:['erc'],
    generalitat_surplus:4,social_spending_cap:0,tax_policy_level:1,
    education_spending:0,healthcare_spending:0,public_salary_level:0,
    education_prepared_level:null,healthcare_prepared_level:null,public_salary_prepared_level:null,
    coalition_policy_commitments:[],parlament_size:135,parlament_s_majority:68,
    budget_year:2013,erc_parlament_s:70});
  for(const party of q.parties) if(party!='erc') q[party+'_parlament_s']=0;
  action(adapter,'generalitat_economy_card.prepare_budget');
  action(adapter,'generalitat_economy_card.budget_'+project);
  expect(q.economy_budget_proposal.projects[project]).toBe(true);
  action(adapter,'generalitat_economy_card.submit_budget');
  expect(q.parlament_budget_can_pass).toBe(true);
  action(adapter,'parlament_budget.pass');
}

function rejectRepeatedBudget(adapter: DendryAdapter) {
  const errors=vi.spyOn(console,'error').mockImplementation(()=>{});
  try {
    adapter.goToScene('parlament_budget.pass');
    expect(errors.mock.calls.some(([message])=>String(message).includes('Budget approval conditions changed'))).toBe(true);
  } finally { errors.mockRestore(); }
}

describe('Business and Labour targeted voter transfers', () => {
  it('ERC inspection gains from the federal left, PSC and abstention, leaving unrelated worker votes alone', () => {
    const { adapter, q } = setup(); const before = support(q);
    action(adapter, 'generalitat_bnl_card.inspection1');
    expect(q.government_budget).toBe(99);
    expect(q[key('erc')] - before[key('erc')]).toBeCloseTo(.6);
    expect(before[key('icv')] - q[key('icv')]).toBeCloseTo(.24);
    expect(before[key('psc')] - q[key('psc')]).toBeCloseTo(.21);
    expect(q[key('cs')]).toBe(before[key('cs')]);
    expect(q[key('ciu')]).toBe(before[key('ciu')]);
    checkPools(q, before);
  });
  it('CUP draws workers from ERC and the federal left rather than PSC', () => {
    const { adapter, q } = setup({ player_party: 'cup', bnl_minister_party: 'cup', cat_coalition: ['cup'] });
    const before = support(q); action(adapter, 'generalitat_bnl_card.inspection1');
    expect(before[key('erc')] - q[key('erc')]).toBeCloseTo(.18);
    expect(before[key('icv')] - q[key('icv')]).toBeCloseTo(.27);
    expect(q[key('psc')]).toBe(before[key('psc')]); checkPools(q, before);
  });
  it('allocates 80% to the minister and 20% to the other unique cabinet carrier', () => {
    const { adapter, q } = setup({ cat_coalition: ['erc', 'ciu', 'ciu'] });
    const before = support(q); action(adapter, 'generalitat_bnl_card.inspection1');
    expect(q[key('erc')] - before[key('erc')]).toBeCloseTo(.48);
    expect(q[key('ciu')] - before[key('ciu')]).toBeCloseTo(.12);
    checkPools(q, before);
  });
  it('keeps a joined ERC/CUP donor route unrealized rather than reallocating it', () => {
    const { adapter, q } = setup({ player_party: 'cup', bnl_minister_party: 'jxsi',
      player_in_jxsi: true, erc_in_jxsi: true, cup_in_jxsi: true, cat_coalition: ['jxsi', 'erc', 'cup'] });
    for (const p of provinces) for (const g of groups) {
      q[key('jxsi', p, g)] = q[key('erc', p, g)] + q[key('cup', p, g)];
      q[key('erc', p, g)] = q[key('cup', p, g)] = 0;
    }
    const before = support(q); action(adapter, 'generalitat_bnl_card.inspection1');
    expect(q[key('jxsi')] - before[key('jxsi')]).toBeCloseTo(.6 * .7);
    expect(q[key('erc')]).toBe(0); expect(q[key('cup')]).toBe(0); checkPools(q, before);
  });
  it('uses the current CiU carrier ahead of an independently surviving PDeCAT donor', () => {
    const { adapter, q } = setup({ parlament_current_ciu: 'cdc', pdcat_split: true });
    for (const p of provinces) for (const g of groups) {
      q[key('cdc',p,g)] = 8; q[key('pdcat',p,g)] = 2; q[key('ciu',p,g)] = 0;
    }
    const before = support(q); action(adapter, 'generalitat_bnl_card.icf1');
    expect(before[key('cdc','barcelona','buss')] - q[key('cdc','barcelona','buss')]).toBeCloseTo(.275);
    expect(q[key('pdcat','barcelona','buss')]).toBe(before[key('pdcat','barcelona','buss')]);
    checkPools(q,before);
  });
  it('leaves an unavailable donor share unrealized and never creates its ballot', () => {
    const {adapter,q}=setup();
    for(const p of provinces) for(const g of groups) {
      q[key('ppc',p,g)]+=q[key('cs',p,g)]; q[key('cs',p,g)]=0;
    }
    const before=support(q); action(adapter,'generalitat_bnl_card.icf1');
    expect(q[key('erc','barcelona','buss')]-before[key('erc','barcelona','buss')]).toBeCloseTo(.55*.8);
    expect(q[key('cs','barcelona','buss')]).toBe(0); checkPools(q,before);
  });
  it('clips each transfer to half of its donor stock and bases spillovers on actual movement', () => {
    const {adapter,q}=setup();
    for(const p of provinces) for(const g of groups) {
      q[key('psc',p,g)]+=q[key('icv',p,g)]-.1; q[key('icv',p,g)]=.1;
    }
    const before=support(q); action(adapter,'generalitat_bnl_card.inspection1');
    expect(q[key('icv')]).toBeCloseTo(.05);
    expect(q[key('erc')]-before[key('erc')]).toBeCloseTo(.05+.21+.15);
    expect(q.erc_congreso_catalunya_support-before.erc_congreso_catalunya_support).toBeLessThan(.1);
    checkPools(q,before);
  });
  it.each(['tourism_less','tourism_more','unions_sovereignty','unions_established'])('round-trips %s without farming even under independence saturation', first => {
    const { adapter, q } = setup();
    for (const p of provinces) for (const g of groups) {
      q[key('erc',p,g)] = 28; q[key('ciu',p,g)] = 20; q[key('abstain',p,g)] = 4;
    }
    const opposite: Q = { tourism_less:'tourism_more', tourism_more:'tourism_less',
      unions_sovereignty:'unions_established', unions_established:'unions_sovereignty' };
    const before = support(q);
    for (let i=0;i<3;i++) {
      action(adapter, 'generalitat_bnl_card.'+first);
      action(adapter, 'generalitat_bnl_card.'+opposite[first]);
    }
    for (const [k,v] of Object.entries(before)) expect(q[k]).toBeCloseTo(v,10);
    expect(q[first.startsWith('tourism') ? 'bnl_tourism_electoral_routes' : 'bnl_union_electoral_routes']).toEqual([]);
  });
  it('established-union preference transfers workers toward PSC and the federal left', () => {
    const { adapter, q } = setup(); const before=support(q);
    action(adapter,'generalitat_bnl_card.unions_established');
    expect(q[key('psc')]).toBeGreaterThan(before[key('psc')]);
    expect(q[key('icv')]).toBeGreaterThan(before[key('icv')]);
    expect(q[key('abstain')]).toBe(before[key('abstain')]);
  });
  it('ICF preparation earns no votes and snapshots ownership for the budget', () => {
    const { adapter,q }=setup({bnl_icf_level:1}); const before=support(q);
    action(adapter,'generalitat_bnl_card.icf_cooperative');
    expect(support(q)).toEqual(before);
    expect(q.active_mods.bnl_icf_proposal.def).toMatchObject({electoral_origin:'erc',electoral_cabinet:['erc']});
  });
  it('does not create a Congreso CUP ballot or change voters outside Catalunya', () => {
    const {adapter,q}=setup({player_party:'cup',bnl_minister_party:'cup',cat_coalition:['cup']}); const before=support(q);
    action(adapter,'generalitat_bnl_card.inspection1');
    expect(q.cup_congreso_catalunya_support).toBe(0);
    for(const p of q.congreso_constituencies) if(p!='catalunya')
      for(const party of q[`congreso_parties_${p}`]) expect(q[`${party}_congreso_${p}_support`]).toBe(before[`${party}_congreso_${p}_support`]);
  });
  it('keeps dormant side-contest unity lists unchanged and credits the fielded ERC ballots', () => {
    const {adapter,q}=setup({player_in_jxsi:true,erc_in_jxsi:true,bnl_minister_party:'jxsi',
      cat_coalition:['jxsi'],jxsi_in_congreso:false,jxsi_in_local:false,
      jxsi_congreso_catalunya_support:3,jxsi_local_barcelona_support:4});
    for(const p of provinces) for(const g of groups) {
      q[key('jxsi',p,g)]=q[key('erc',p,g)]; q[key('erc',p,g)]=0;
    }
    const before=support(q); action(adapter,'generalitat_bnl_card.inspection1');
    expect(q.jxsi_congreso_catalunya_support).toBe(before.jxsi_congreso_catalunya_support);
    expect(q.jxsi_local_barcelona_support).toBe(before.jxsi_local_barcelona_support);
    expect(q.erc_congreso_catalunya_support).toBeGreaterThan(before.erc_congreso_catalunya_support);
    expect(q.erc_local_barcelona_support).toBeGreaterThan(before.erc_local_barcelona_support);
    checkPools(q,before);
  });
});

describe('Territory policy ownership and electoral milestones', () => {
  it.each(['livestock_more','livestock_less'])('returns %s to neutral without cumulative votes', first => {
    const {adapter,q}=setup(); const before=support(q);
    action(adapter,'generalitat_territory_card.'+first);
    action(adapter,'generalitat_territory_card.'+(first=='livestock_more'?'livestock_less':'livestock_more'));
    for(const [k,v] of Object.entries(before)) expect(q[k]).toBeCloseTo(v,10);
    expect(q.territory_livestock_electoral_routes).toEqual([]);
  });
  it.each(['water2','tmobilitat2'])('%s preparation keeps electoral gains for delivery', actionName => {
    const {adapter,q}=setup({territory_water:1,territory_tmobilitat:1}); const before=support(q);
    action(adapter,'generalitat_territory_card.'+actionName);
    expect(support(q)).toEqual(before);
    expect(q.active_mods['territory_'+actionName].def.electoral_origin).toBe('erc');
  });
  it.each(['roads2','roads3','metro1','metro2','rail1','rail2'])('%s awards modest geographically limited preparation credit once', name => {
    const {adapter,q}=setup({territory_roads:name=='roads3'?2:1,
      territory_metro:name=='metro2'?1:0,territory_metro_approved:name=='metro2'?1:0,
      territory_rail_approved:name=='rail2'?1:0,territory_tmobilitat:name=='rail2'?2:0});
    const before=support(q), modBefore=q.erc_redbelt_mod;
    action(adapter,'generalitat_territory_card.'+name);
    expect(q.government_budget).toBeLessThan(100);
    const after=support(q); action(adapter,'generalitat_territory_card.'+name);
    expect(support(q)).toEqual(after);
    expect(q.erc_redbelt_mod).toBe(modBefore); checkPools(q,before);
    if(['roads2','rail1','rail2'].includes(name)) {
      for(const group of groups) expect(q[key('erc','barcelona',group)]).toBe(before[key('erc','barcelona',group)]);
      expect(q.erc_local_barcelona_support).toBe(before.erc_local_barcelona_support);
    }
    if(name=='roads3') expect(q.erc_local_barcelona_support).toBe(before.erc_local_barcelona_support);
  });
  it('completion credits the initiating ministry despite a changed government, with no repeat reward', () => {
    const {adapter,q}=setup(); action(adapter,'generalitat_territory_card.metro1');
    const before=support(q);
    Object.assign(q,{player_party:'cup',territory_minister_party:'cup',cat_coalition:['cup'],territory_metro1_trigger:true});
    action(adapter,'post_event');
    expect(q[key('erc','barcelona','young')]-before[key('erc','barcelona','young')]).toBeCloseTo(.9*.6);
    expect(q[key('cup','barcelona','young')]).toBe(before[key('cup','barcelona','young')]);
    const after=support(q); action(adapter,'post_event'); expect(support(q)).toEqual(after);
    expect(q.territory_metro1_trigger).toBe(false); checkPools(q,before);
  });
  it('gives the prepared metro one lifetime reward across preparation, actual budget funding and completion', () => {
    const {adapter,q}=setup({generalitat_surplus:4,social_spending_cap:0,tax_policy_level:1,
      education_spending:0,healthcare_spending:0,public_salary_level:0,
      education_prepared_level:null,healthcare_prepared_level:null,public_salary_prepared_level:null,
      coalition_policy_commitments:[],parlament_size:135,parlament_s_majority:68,
      budget_year:2013,erc_parlament_s:70});
    for(const party of q.parties) if(party!='erc') q[party+'_parlament_s']=0;
    const before=support(q); action(adapter,'generalitat_territory_card.metro1');
    expect(q[key('erc','barcelona','young')]-before[key('erc','barcelona','young')]).toBeCloseTo(.9*.15);
    action(adapter,'generalitat_economy_card.prepare_budget');
    action(adapter,'generalitat_economy_card.budget_metro1');
    action(adapter,'generalitat_economy_card.submit_budget');
    expect(q.parlament_budget_can_pass).toBe(true);
    action(adapter,'parlament_budget.pass');
    expect(q[key('erc','barcelona','young')]-before[key('erc','barcelona','young')]).toBeCloseTo(.9*.4);
    expect(q.active_mods.territory_metro1.def.electoral_origin).toBe('erc');
    q.territory_metro1_trigger=true; action(adapter,'post_event');
    expect(q[key('erc','barcelona','young')]-before[key('erc','barcelona','young')]).toBeCloseTo(.9);
    checkPools(q,before);
  });
  it.each(['guarantees','cooperative'])('funds the CUP %s ICF branch with its own magnitude and original ministry ownership', branch => {
    const {adapter,q}=setup({player_party:'cup',bnl_minister_party:'cup',economy_minister_party:'cup',
      cat_coalition:['cup'],bnl_icf_level:1});
    const before=support(q); action(adapter,'generalitat_bnl_card.icf_'+branch);
    expect(support(q)).toEqual(before);
    fund(adapter,'icf');
    expect(q.bnl_icf_level).toBe(2);
    const group=branch=='guarantees'?'buss':'ind';
    expect(q[key('cup','barcelona',group)]-before[key('cup','barcelona',group)])
      .toBeCloseTo(branch=='guarantees'?.85*.7:.65);
    const after=support(q); rejectRepeatedBudget(adapter); expect(support(q)).toEqual(after);
    checkPools(q,before);
  });
  it.each(['water2','water3'])('credits %s only when its actual budget funds delivery', project => {
    const {adapter,q}=setup({territory_water:project=='water2'?1:2}); const before=support(q);
    action(adapter,'generalitat_territory_card.'+project); expect(support(q)).toEqual(before);
    fund(adapter,project);
    expect(q.territory_water).toBe(project=='water2'?2:3);
    expect(q[key('erc')]-before[key('erc')]).toBeCloseTo(project=='water2'?.4:.65);
    const after=support(q); rejectRepeatedBudget(adapter); expect(support(q)).toEqual(after);
    checkPools(q,before);
  });
  it('splits Catalonia-wide ticketing credit between actual budget funding and delivery outside Barcelona', () => {
    const {adapter,q}=setup({territory_tmobilitat:1}); const before=support(q);
    action(adapter,'generalitat_territory_card.tmobilitat2'); expect(support(q)).toEqual(before);
    fund(adapter,'tmobilitat2');
    expect(q[key('erc','girona','young')]-before[key('erc','girona','young')]).toBeCloseTo(.5*.4);
    expect(q.active_mods.territory_tmobilitat2.def.electoral_origin).toBe('erc');
    Object.assign(q,{player_party:'cup',territory_minister_party:'cup',cat_coalition:['cup'],territory_tmobilitat2_trigger:true});
    action(adapter,'post_event');
    expect(q[key('erc','girona','young')]-before[key('erc','girona','young')]).toBeCloseTo(.5);
    expect(q[key('erc','barcelona','young')]).toBe(before[key('erc','barcelona','young')]);
    expect(q.erc_local_barcelona_support).toBe(before.erc_local_barcelona_support);
    checkPools(q,before);
  });
  it('retains reversible package records as serializable current state', () => {
    const {adapter,q}=setup(); const before=support(q);
    action(adapter,'generalitat_bnl_card.tourism_less');
    const saved=JSON.parse(JSON.stringify(q)); Object.assign(q,saved);
    action(adapter,'generalitat_bnl_card.tourism_more');
    for(const [k,v] of Object.entries(before)) expect(q[k]).toBeCloseTo(v,10);
  });
  it('rejects stale Territory arrivals in caretaker government', () => {
    const {adapter,q}=setup({cat_caretaker_gov:true}); const before=support(q);
    action(adapter,'generalitat_territory_card.water1');
    expect(q.government_budget).toBe(100); expect(q.territory_water).toBe(0); expect(support(q)).toEqual(before);
  });
});
