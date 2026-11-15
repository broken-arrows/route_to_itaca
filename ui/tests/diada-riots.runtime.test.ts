import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
const errors = vi.spyOn(console, 'error').mockImplementation(() => {});

afterEach(() => {
  const actionErrors = errors.mock.calls.filter(([message]) => String(message).includes('Scene action error'));
  errors.mockClear();
  expect(actionErrors).toEqual([]);
});

function setup(overrides: Record<string, unknown> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    erc_in_jxsi: false, erc_in_jxcat: false, cup_in_jxsi: false, cup_in_jxcat: false,
    erc_in_gob: true, erc_supporting: true, cup_in_gob: false, cup_supporting: false,
    jxsi_in_gob: false, jxcat_in_gob: false, jxsi_supporting: false, jxcat_supporting: false,
    cat_coalition: ['erc', 'ciu'], cat_coalition_support: [], cat_caretaker_gov: false,
    interior_minister_party: 'erc', president_party: 'ciu',
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    broken_roadmap: false, burdens_broken: false, fnc_formed: false, pxc_dissolved: false,
    independence_trust: 30, social_dissent: 50, cat_spa_relations: 20,
    erc_left_dissent: 10, erc_core_dissent: 10, erc_pragmatic_dissent: 10,
    cup_endavant_dissent: 10, cup_thirdsector_dissent: 10, cup_poblelliure_dissent: 10,
    ciu_relations: 50, psc_relations: 50, cup_relations: 50, icv_relations: 50,
    ...overrides,
  });
  const opening: Record<string, number> = { ciu: 20, erc: 15, cup: 10, icv: 10, psc: 10, cs: 10, abstain: 25 };
  for (const c of q.parlament_constituencies) {
    for (const d of q.parlament_demographics) {
      for (const p of [...q.parties, 'abstain']) q[`${p}_parlament_${c}_${d}_support`] = opening[p] || 0;
      for (const list of ['jxsi', 'jxcat']) {
        if (!q[`player_in_${list}`]) continue;
        q[`${list}_parlament_${c}_${d}_support`] = opening.ciu + opening[q.player_party];
        q[`ciu_parlament_${c}_${d}_support`] = 0;
        q[`${q.player_party}_parlament_${c}_${d}_support`] = 0;
      }
    }
  }
  const frame = adapter.goToScene('diada_riots');
  function choose(response: string) {
    const index = frame.choices.findIndex(choice => choice.id === `diada_riots.${response}` && choice.canChoose);
    expect(index).toBeGreaterThanOrEqual(0);
    return adapter.choose(index);
  }
  return { adapter, q, frame, choose };
}

const stock = (q: Record<string, any>, p: string, d = 'young', c = 'barcelona') =>
  q[`${p}_parlament_${c}_${d}_support`] as number;

describe('Diada riot responses', () => {
  it.each(['minister', 'cabinet', 'support', 'opposition'])('offers the correct choices for %s', role => {
    const governing = role === 'minister' || role === 'cabinet';
    const { frame } = setup({
      erc_in_gob: governing, erc_supporting: governing || role === 'support',
      interior_minister_party: role === 'minister' ? 'erc' : 'ciu',
      cat_coalition: governing ? ['erc', 'ciu'] : ['ciu'],
      cat_coalition_support: role === 'support' ? ['erc'] : [],
    });
    expect(frame.choices.some(choice => choice.id === 'diada_riots.mossos')).toBe(role === 'minister');
    const protest = frame.choices.find(choice => choice.id === 'diada_riots.protest');
    expect(protest?.canChoose).toBe(!governing);
    if (governing) expect(protest?.subtitle?.trim()).toBeTruthy();
    expect(frame.choices.find(choice => choice.id === 'diada_riots.nothing')?.canChoose).toBe(true);
  });

  it.each(['broken_roadmap', 'burdens_broken'])('blocks protest with %s, including external supporters', flag => {
    const { frame } = setup({ erc_in_gob: false, cat_coalition: ['ciu'], cat_coalition_support: ['erc'], [flag]: true });
    const protest = frame.choices.find(choice => choice.id === 'diada_riots.protest');
    expect(protest?.canChoose).toBe(false);
    expect(protest?.subtitle?.trim()).toBeTruthy();
  });

  it.each(['jxsi', 'jxcat'])('recognizes Interior control and pooled responsibility through %s', list => {
    const { q, frame, choose } = setup({
      [`player_in_${list}`]: true, [`erc_in_${list}`]: true, [`${list}_in_gob`]: true,
      cat_coalition: [list], interior_minister_party: list,
    });
    expect(frame.choices.find(choice => choice.id === 'diada_riots.mossos')?.canChoose).toBe(true);
    expect(frame.choices.find(choice => choice.id === 'diada_riots.protest')?.canChoose).toBe(false);
    choose('mossos');
    expect(stock(q, list)).toBeLessThan(35);
    expect(stock(q, 'erc')).toBe(0);
    expect(stock(q, 'ciu')).toBe(0);
    expect(stock(q, 'cup')).toBeGreaterThan(10);
  });

  it('trades trust and faction unity for restored order and institutional relations', () => {
    const { q, choose } = setup();
    const result = choose('mossos');
    expect(q.social_dissent).toBe(58);
    expect(q.independence_trust).toBe(20);
    expect(q.cat_spa_relations).toBe(26);
    expect(q.erc_left_dissent).toBe(16);
    expect(q.erc_core_dissent).toBe(14);
    expect(q.erc_pragmatic_dissent).toBe(11);
    expect(q.ciu_relations).toBe(53);
    expect(q.psc_relations).toBe(53);
    expect(q.cup_relations).toBe(44);
    expect(q.icv_relations).toBe(46);
    expect(stock(q, 'erc')).toBeLessThan(15);
    expect(15 - stock(q, 'erc')).toBeGreaterThan(20 - stock(q, 'ciu'));
    expect(stock(q, 'cup')).toBeGreaterThan(10);
    expect(stock(q, 'fnc')).toBe(0);
    expect(result.choices.some(choice => choice.id === 'post_event' && choice.canChoose)).toBe(true);
  });

  it('makes a CUP deployment internally costly and never transfers votes to its own ballot', () => {
    const { q, choose } = setup({
      player_party: 'cup', erc_in_gob: false, erc_supporting: false,
      cup_in_gob: true, cup_supporting: true, cat_coalition: ['cup', 'ciu'],
      interior_minister_party: 'cup', fnc_formed: true, pxc_dissolved: true,
    });
    choose('mossos');
    expect(q.cup_endavant_dissent).toBe(20);
    expect(q.cup_thirdsector_dissent).toBe(18);
    expect(q.cup_poblelliure_dissent).toBe(14);
    expect(stock(q, 'cup')).toBeLessThan(10);
    expect(stock(q, 'fnc')).toBeGreaterThan(0);
  });

  it('lets an external supporter gain from protesting without restoring global trust', () => {
    const { q, choose } = setup({
      erc_in_gob: false, cat_coalition: ['ciu'], cat_coalition_support: ['erc'], interior_minister_party: 'ciu',
    });
    choose('protest');
    expect(stock(q, 'erc')).toBeGreaterThan(15);
    expect(stock(q, 'ciu')).toBeLessThan(20);
    expect(q.independence_trust).toBe(28);
    expect(q.cat_spa_relations).toBe(16);
    expect(q.ciu_relations).toBe(46);
  });

  it.each(['cabinet', 'support', 'opposition'])('silence retains appropriate responsibility for %s', role => {
    const { q, choose } = setup({
      erc_in_gob: role === 'cabinet', erc_supporting: role !== 'opposition',
      cat_coalition: role === 'cabinet' ? ['erc', 'ciu'] : ['ciu'],
      cat_coalition_support: role === 'support' ? ['erc'] : [], interior_minister_party: 'ciu',
    });
    choose('nothing');
    if (role === 'opposition') expect(stock(q, 'erc')).toBe(15);
    else expect(stock(q, 'erc')).toBeLessThan(15);
    expect(q.cat_spa_relations).toBe(18);
  });

  it.each(['mossos', 'protest', 'nothing'])('conserves every electorate for %s and leaves untargeted cells alone', response => {
    const { q, choose } = setup(response === 'protest' ? {
      erc_in_gob: false, erc_supporting: false, cat_coalition: ['ciu'], interior_minister_party: 'ciu',
    } : {});
    choose(response);
    for (const c of q.parlament_constituencies) {
      for (const d of q.parlament_demographics) {
        expect([...q.parties, 'abstain'].reduce((sum, p) => sum + stock(q, p, d, c), 0)).toBeCloseTo(100, 10);
        for (const p of [...q.parties, 'abstain']) expect(stock(q, p, d, c)).toBeGreaterThanOrEqual(0);
      }
    }
    expect(stock(q, 'erc', 'retired')).toBe(15);
    expect(stock(q, 'cup', 'buss')).toBe(10);
  });

  it('rejects a forced hidden intervention before applying effects', () => {
    const { adapter, q } = setup({ interior_minister_party: 'ciu' });
    adapter.goToScene('diada_riots.mossos');
    expect(q.independence_trust).toBe(30);
    expect(q.social_dissent).toBe(50);
    expect(q.cat_spa_relations).toBe(20);
    expect(stock(q, 'erc')).toBe(15);
  });

  it('routes losses to abstention when every alternative is in the cabinet', () => {
    const { q, choose } = setup({ cat_coalition: ['erc', 'ciu', 'cup', 'icv'] });
    choose('mossos');
    expect(stock(q, 'abstain')).toBeGreaterThan(25);
    expect(stock(q, 'fnc')).toBe(0);
    expect([...q.parties, 'abstain'].reduce((sum, p) => sum + stock(q, p), 0)).toBeCloseTo(100, 10);
  });

  it.each([2015, 2016, 2017, 2018, 2019])('remains reachable from the %i report and resets the response', year => {
    const { adapter, q, choose } = setup();
    choose('mossos');
    Object.assign(q, { year, month: 9, rubicon: false });
    let frame = adapter.goToScene(`${year}diada`);
    frame = adapter.choose(frame.choices.findIndex(choice => choice.id === `${year}diada.pass`));
    const riots = frame.choices.findIndex(choice => choice.id === `${year}diada.riots`);
    expect(riots).toBeGreaterThanOrEqual(0);
    frame = adapter.choose(riots);
    expect(q.diada_riots_response).toBeFalsy();
    expect(frame.choices.some(choice => choice.id === 'diada_riots.mossos' && choice.canChoose)).toBe(true);
  });
});
