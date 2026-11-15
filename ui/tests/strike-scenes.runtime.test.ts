import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, process.env.ITACA_STRIKE_GAME_JSON || '../../out/game.json'), 'utf8');
const sectors = [
  ['fgc', 'territory'], ['firefighters', 'interior'], ['healthcare', 'health'],
  ['teachers', 'education'], ['university', 'education'], ['public_pay', 'economy'],
  ['metal', 'bnl'],
] as const;
const agreementIds: Record<string, string | undefined> = {
  fgc: 'fgc_working_conditions_agreement', firefighters: 'firefighters_staffing_agreement',
  healthcare: 'primary_care_workload_agreement', university: 'university_tuition_agreement',
  metal: 'metal_collective_agreement',
};
const errors = vi.spyOn(console, 'error').mockImplementation(() => {});

afterEach(() => {
  const actionErrors = errors.mock.calls.filter(([message]) => String(message).includes('Scene action error'));
  errors.mockClear();
  expect(actionErrors).toEqual([]);
});

function setup(sector: string, ministry: string, overrides: Record<string, unknown> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    erc_in_gob: true, erc_supporting: false, cup_in_gob: false, cup_supporting: false,
    jxsi_in_gob: false, jxsi_supporting: false, jxcat_in_gob: false, jxcat_supporting: false,
    cat_coalition: ['erc', 'psc'], cat_coalition_support: [], cat_caretaker_gov: false,
    president_party: 'psc', parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    [`${ministry}_minister_party`]: 'psc', [`${ministry}_minister`]: 'Test Minister',
    [`${ministry}_minister_gender`]: 'f', psc_relations: 55,
    government_budget: 2, party_resources: 2, difficulty: 1,
    education_spending: sector === 'teachers' ? -1 : 0, healthcare_spending: 0,
    public_salary_level: 0, social_spending_cap: 2, economy_cuts: 0,
    fla_escalated: false, fla_severe: false, active_mods: {}, mod_log: [],
    strike_triggered: sector,
    ...overrides,
  });
  const frame = adapter.goToScene(`strike_${sector}`);
  return { adapter, q, frame };
}

describe.each(sectors)('%s strike options', (sector, ministry) => {
  it.each([
    [-1, 44, false], [-1, 45, true],
    [0, 54, false], [0, 55, true],
    [1, 54, false], [1, 55, true],
  ])('pressure at difficulty %i and relations %i has eligibility %s', (difficulty, relations, allowed) => {
    const { frame } = setup(sector, ministry, { difficulty, psc_relations: relations });
    const pressure = frame.choices.find(choice => choice.id === `strike_${sector}.pressure`);
    expect(pressure).toBeDefined();
    expect(pressure!.canChoose).toBe(allowed);
    if (!allowed) expect(pressure!.subtitle?.trim()).toBeTruthy();
  });

  it.each(['cabinet', 'support', 'opposition', 'minister'])('offers the correct options for %s', role => {
    const responsible = role === 'cabinet' || role === 'minister';
    const { frame } = setup(sector, ministry, {
      erc_in_gob: responsible, erc_supporting: role === 'support',
      cat_coalition: responsible ? ['erc', 'psc'] : ['psc'],
      cat_coalition_support: role === 'support' ? ['erc'] : [],
      [`${ministry}_minister_party`]: role === 'minister' ? 'erc' : 'psc',
    });
    const option = (name: string) => frame.choices.find(choice => choice.id === `strike_${sector}.${name}`);
    const protest = option('protest');
    expect(protest).toBeDefined();
    expect(protest!.canChoose).toBe(role === 'opposition');
    if (role !== 'opposition') expect(protest!.subtitle?.trim()).toBeTruthy();
    expect(Boolean(option('pressure'))).toBe(role === 'cabinet' || role === 'support');
    for (const name of sector === 'metal' ? ['workers', 'compromise', 'industry'] : ['accept']) {
      expect(Boolean(option(name))).toBe(role === 'minister');
      if (role === 'minister') expect(option(name)!.canChoose).toBe(true);
    }
    expect(option('quiet')!.canChoose).toBe(true);
  });

  it.each(['jxsi', 'jxcat'])('recognizes shared ministry control through %s', list => {
    const { frame } = setup(sector, ministry, {
      [`player_in_${list}`]: true, [`${list}_in_gob`]: true,
      [`${ministry}_minister_party`]: list, cat_coalition: [list],
    });
    expect(frame.choices.some(choice => choice.id === `strike_${sector}.pressure`)).toBe(false);
    const accept = sector === 'metal' ? 'workers' : 'accept';
    expect(frame.choices.find(choice => choice.id === `strike_${sector}.${accept}`)?.canChoose).toBe(true);
    expect(frame.choices.find(choice => choice.id === `strike_${sector}.protest`)?.canChoose).toBe(false);
  });

  it.each(['party_resources', 'government_budget'])('keeps pressure visible but disabled without %s', resource => {
    const { frame } = setup(sector, ministry, { [resource]: 0 });
    const pressure = frame.choices.find(choice => choice.id === `strike_${sector}.pressure`);
    expect(pressure).toBeDefined();
    expect(pressure!.canChoose).toBe(false);
    expect(pressure!.subtitle?.trim()).toBeTruthy();
  });

  it.each([
    ['erc', undefined], ['cup', undefined], ['erc', 'jxsi'], ['cup', 'jxcat'],
  ])('waives executive capacity for external support by %s through %s', (player, list) => {
    const vehicle = list || player;
    const { adapter, q, frame } = setup(sector, ministry, {
      player_party: player, erc_in_gob: false,
      [`${vehicle}_supporting`]: true,
      ...(list ? { [`player_in_${list}`]: true } : {}),
      cat_coalition: ['psc'], cat_coalition_support: [vehicle], government_budget: 0,
    });
    const index = frame.choices.findIndex(choice => choice.id === `strike_${sector}.pressure` && choice.canChoose);
    expect(index).toBeGreaterThanOrEqual(0);
    expect(frame.choices[index].subtitle).not.toContain('budget');
    adapter.choose(index);
    expect(q[`strike_${sector}_settled`]).toBe(true);
    expect(q.party_resources).toBe(1);
    expect(q.government_budget).toBe(0);
  });

  it.each([-1, 2])('leaves external supporters\' existing budget %i untouched', budget => {
    const { adapter, q, frame } = setup(sector, ministry, {
      erc_in_gob: false, erc_supporting: true,
      cat_coalition: ['psc'], cat_coalition_support: ['erc'], government_budget: budget,
    });
    adapter.choose(frame.choices.findIndex(choice => choice.id === `strike_${sector}.pressure` && choice.canChoose));
    expect(q[`strike_${sector}_settled`]).toBe(true);
    expect(q.government_budget).toBe(budget);
  });

  it.each(['indp.', 'ind.', 'independent'])('lets the presidency settle for an %s minister', affiliation => {
    const { adapter, q, frame } = setup(sector, ministry, {
      president_party: 'erc', [`${ministry}_minister_party`]: affiliation,
    });
    expect(frame.choices.some(choice => choice.id === `strike_${sector}.pressure`)).toBe(false);
    for (const name of sector === 'metal' ? ['workers', 'compromise', 'industry'] : ['accept']) {
      expect(frame.choices.find(choice => choice.id === `strike_${sector}.${name}`)?.canChoose).toBe(true);
    }
    const accept = sector === 'metal' ? 'workers' : 'accept';
    adapter.choose(frame.choices.findIndex(choice => choice.id === `strike_${sector}.${accept}`));
    expect(q[`strike_${sector}_settled`]).toBe(true);
    expect(q.government_budget).toBe(1);
  });

  it.each(['jxsi', 'jxcat'])('recognizes an independent ministry under the shared %s presidency', list => {
    const { frame } = setup(sector, ministry, {
      [`player_in_${list}`]: true, [`${list}_in_gob`]: true,
      president_party: list, [`${ministry}_minister_party`]: 'indp.', cat_coalition: [list],
    });
    const accept = sector === 'metal' ? 'workers' : 'accept';
    expect(frame.choices.find(choice => choice.id === `strike_${sector}.${accept}`)?.canChoose).toBe(true);
    expect(frame.choices.some(choice => choice.id === `strike_${sector}.pressure`)).toBe(false);
  });

  it('addresses pressure to the presidency party for an independent minister', () => {
    const { adapter, q, frame } = setup(sector, ministry, { [`${ministry}_minister_party`]: 'indp.' });
    const index = frame.choices.findIndex(choice => choice.id === `strike_${sector}.pressure` && choice.canChoose);
    expect(index).toBeGreaterThanOrEqual(0);
    expect(frame.choices[index].title).toContain('psc');
    expect(frame.choices[index].subtitle).toContain('psc');
    expect(frame.choices[index].title).not.toContain('indp.');
    adapter.choose(index);
    expect(q[`strike_${sector}_settled`]).toBe(true);
    expect(q.psc_relations).toBe(52);
  });

  it('starts a nine-step cooldown on arrival', () => {
    const { q } = setup(sector, ministry);
    expect(q.strike_timer).toBe(9);
  });

  it('settles through the visible pressure choice at 55 relations on hard', () => {
    const { adapter, q, frame } = setup(sector, ministry);
    const index = frame.choices.findIndex(choice => choice.id === `strike_${sector}.pressure` && choice.canChoose);
    expect(index).toBeGreaterThanOrEqual(0);
    adapter.choose(index);
    expect(q[`strike_${sector}_settled`]).toBe(true);
    expect(q.strike_triggered).toBeNull();
    expect(q.party_resources).toBe(1);
    expect(q.government_budget).toBe(1);
    const agreementId = agreementIds[sector];
    if (agreementId) {
      const agreement = q.active_mods[agreementId];
      expect(agreement.active).toBe(true);
      expect(agreement.def.icon).toBeUndefined();
      expect(agreement.def.colour).toBeUndefined();
      const visible = adapter.engine.gameLib.getLawsForUI(q).filter((law: any) => law.icon);
      expect(visible.some((law: any) => law.id === agreementId)).toBe(false);
      expect(agreement.def.targets.welfare_index_growth).toBeGreaterThan(0);
    }
    if (sector === 'metal') {
      expect(q.active_mods.metal_collective_agreement.def.settlement).toBe('compromise');
      expect(q.public_salary_level).toBe(0);
    }
  });
});

it.each([
  ['teachers', 'education', { social_spending_cap: -1 }],
  ['university', 'education', { education_spending: -1 }],
  ['healthcare', 'health', { healthcare_spending: -1 }],
  ['public_pay', 'economy', { economy_cuts: 1 }],
  ['teachers', 'education', { fla_escalated: true, fla_severe: true }],
  ['university', 'education', { fla_escalated: true }],
  ['healthcare', 'health', { fla_escalated: true }],
  ['public_pay', 'economy', { fla_escalated: true }],
])('%s pressure retains %s restrictions (%j)', (sector, ministry, overrides) => {
  const { frame } = setup(sector, ministry, {
    erc_in_gob: false, erc_supporting: true,
    cat_coalition: ['psc'], cat_coalition_support: ['erc'], government_budget: 0,
    ...overrides,
  });
  const pressure = frame.choices.find(choice => choice.id === `strike_${sector}.pressure`);
  expect(pressure).toBeDefined();
  expect(pressure!.canChoose).toBe(false);
  expect(pressure!.subtitle?.trim()).toBeTruthy();
});

it.each(['workers', 'compromise', 'industry'])('keeps the %s metal settlement active and out of the policy UI', response => {
  const { adapter, q, frame } = setup('metal', 'bnl', { bnl_minister_party: 'erc' });
  const index = frame.choices.findIndex(choice => choice.id === `strike_metal.${response}` && choice.canChoose);
  expect(index).toBeGreaterThanOrEqual(0);
  adapter.choose(index);
  const agreement = q.active_mods.metal_collective_agreement;
  expect(agreement.active).toBe(true);
  expect(agreement.def.settlement).toBe(response);
  expect(agreement.def.icon).toBeUndefined();
  expect(agreement.def.colour).toBeUndefined();
  expect(adapter.engine.gameLib.getLawsForUI(q).filter((law: any) => law.icon)
    .some((law: any) => law.id === 'metal_collective_agreement')).toBe(false);
});
