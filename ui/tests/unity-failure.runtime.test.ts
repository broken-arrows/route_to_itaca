import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
const rupture = 'generalitat_coalition_rupture';
const offices = ['president', 'vicepresident', 'speaker', 'economy_minister', 'interior_minister',
  'health_minister', 'education_minister', 'territory_minister', 'foreign_minister', 'bnl_minister'];
let errors: string[];

beforeEach(() => {
  errors = [];
  vi.spyOn(console, 'error').mockImplementation((...args) => errors.push(args.map(String).join(' ')));
});
afterEach(() => {
  vi.restoreAllMocks();
  expect(errors).toEqual([]);
});

function setup(list = 'jxsi', player = 'erc', overrides: Record<string, any> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  for (const party of q.parties) {
    q[`${party}_in_jxsi`] = false;
    q[`${party}_in_jxcat`] = false;
    q[`${party}_parlament_s`] = 0;
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics)
      q[`${party}_parlament_${province}_${demo}_support`] = 0;
  }
  Object.assign(q, {
    player_party: player, player_in_jxsi: list === 'jxsi', player_in_jxcat: list === 'jxcat',
    parlament_current_ciu: 'cdc', parlament_current_icv: 'icv',
    [`${list}_formed`]: true, [`${list}_dissolved`]: false, [`${list}_formation_num`]: 40,
    [`${list}_cup_formation_num`]: 15, [`${list}_roadmap`]: 2,
    [`${list}_parlament_s`]: 70, [`${list}_parlament_pv`]: 45,
    [`${list}_in_gob`]: true, [`${list}_in_congreso`]: false, [`${list}_in_local`]: false,
    cdc_relations: 85, erc_relations: 85, cup_relations: 85, psc_relations: 85,
    cdc_leader: 'Artur Mas', erc_leader: 'Oriol Junqueras', cup_leader: 'Antonio Baños',
    cdc_roadmap: 2, erc_roadmap: 2, cup_roadmap: 2, player_roadmap: 2,
    cat_coalition: [list], cat_coalition_support: [], cat_coalition_abstain: [],
    cat_caretaker_gov: false, erc_in_gob: true, cup_in_gob: true, cdc_in_gob: true,
    year: 2016, month: 3, week: 1, time: 40, rubicon: false, difficulty: 0,
    roadmap_year: 2018, next_election_time: 80, next_election_year: 2019,
    next_spanish_election_time: 90, budget_year: 2016, budget_year_attempted: 2016,
    month_actions: 0, month_actions_last: 0, party_resources: 10,
    government_budget: 10, generalitat_surplus: 1, social_spending_cap: 2,
    education_spending: 0, healthcare_spending: 0, public_salary_level: 0,
    tax_policy_level: 1, economy_cuts: 0, countdowns: [],
    broken_roadmap: false, broken_roadmap_trigger: false, broken_roadmap_expires_trigger: false,
    burdens_broken: false, burdens_goal: 4, burdens_time: 12, burdens_done: 0,
    unity_failed_mandates: [], coalition_rupture_trigger: false, coalition_rupture_recheck: false,
    coalition_rupture_list: '', coalition_policy_commitments: [], coalition_indy_commitments: [],
    consultation_pending: false, referendum_pending: false, parlament_sovereignty_declaration: false,
    government_does_things_timer: 99, strike_timer: 99, ...overrides,
  });
  for (const party of ['cdc', 'erc', 'cup']) q[`${party}_in_${list}`] = true;
  for (const office of offices) {
    const affiliation = ['education_minister', 'territory_minister'].includes(office) ? 'cup'
      : ['vicepresident', 'health_minister', 'foreign_minister'].includes(office) ? 'erc' : 'cdc';
    q[`${office}_party`] = list;
    q[`${office}_affiliation`] = affiliation;
    q[office] = office === 'president' ? 'Artur Mas' : `${affiliation} ${office}`;
    q[`${office}_gender`] = office === 'education_minister' ? 'f' : 'm';
  }
  for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
    q[`${list}_parlament_${province}_${demo}_support`] = 20;
    q[`abstain_parlament_${province}_${demo}_support`] = 80;
  }
  adapter.engine.game.tagLookup = {
    ...adapter.engine.game.tagLookup,
    event: Object.fromEntries([rupture, 'broken_roadmap', 'burdens_start', 'generalitat_coalition_indy_breach'].map(id => [id, true])),
    election: {},
  };
  return { adapter, q, list, player };
}
type Fixture = ReturnType<typeof setup>;

function post(f: Fixture) {
  // Exercise the compiled monthly/no-time arrival in both lists. JxCat's existing
  // end-game navigation remains outside these mechanics tests.
  const engine = f.adapter.engine as any;
  engine._runActions(engine.game.scenes.post_event.onArrival, 'unity failure post_event');
}
function pick(f: Fixture, id: string) {
  const index = f.adapter.engine.choiceCache.findIndex(c => c.id === id);
  expect(index, id).toBeGreaterThanOrEqual(0);
  expect(f.adapter.engine.choiceCache[index].canChoose, id).toBe(true);
  return f.adapter.choose(index);
}
function open(f: Fixture) {
  const frame = f.adapter.goToScene('post_event.events_choice');
  expect(frame.choices.some(c => c.id === rupture && c.canChoose)).toBe(true);
  return pick(f, rupture);
}

describe('unity-list failed mandate penalties', () => {
  for (const list of ['jxsi', 'jxcat']) for (const player of ['erc', 'cup']) {
    it.each([-1, 0, 1])(`${list}/${player}: roadmap hit scales with difficulty %s`, difficulty => {
      const f = setup(list, player, { difficulty, broken_roadmap: true, cdc_relations: 95 });
      post(f);
      expect(f.q.cdc_relations).toBe(95 - (40 + 5 * difficulty));
      expect(f.q[player + '_relations']).toBe(85);
      const other = player === 'erc' ? 'cup' : 'erc';
      expect(f.q[other + '_relations']).toBe(85 - (40 + 5 * difficulty));
      expect(f.q.psc_relations).toBe(85);
      expect(f.q.burdens_broken).toBe(true);
    });
    it.each([-1, 0, 1])(`${list}/${player}: burdens hit scales with difficulty %s`, difficulty => {
      const f = setup(list, player, { difficulty, burdens_broken: true });
      post(f);
      expect(f.q.cdc_relations).toBe(85 - (25 + 5 * difficulty));
      expect(f.q[player + '_relations']).toBe(85);
    });
  }

  it.each([15, 30, 45])('bounds new damage without healing existing relations of %s', before => {
    const f = setup('jxsi', 'erc', { broken_roadmap: true, cdc_relations: before });
    post(f);
    expect(f.q.cdc_relations).toBe(Math.min(before, 30));
  });

  it('does not penalize an outside CUP or a standalone player', () => {
    const f = setup('jxsi', 'erc', { broken_roadmap: true });
    f.q.cup_in_jxsi = false;
    post(f);
    expect(f.q.cup_relations).toBe(85);
    const solo = setup('jxsi', 'erc', { player_in_jxsi: false, broken_roadmap: true });
    solo.q.erc_in_jxsi = false;
    solo.q.cat_coalition = ['erc', 'cdc'];
    post(solo);
    expect(solo.q.cdc_relations).toBe(85);
  });

  it('charges once across no-time returns, save/load and a later monthly tick', () => {
    const f = setup('jxsi', 'erc', { broken_roadmap: true });
    post(f);
    post(f);
    const saved = f.adapter.exportStateJSON();
    f.adapter.importStateJSON(saved);
    f.q = f.adapter.qualities;
    post(f);
    expect(f.q.cdc_relations).toBe(45);
    expect(f.q.cup_relations).toBe(45);
    f.q.month_actions = 1;
    post(f);
    expect(f.q.month).toBe(4);
    expect(f.q.cdc_relations).toBe(45);
    f.q.broken_roadmap_expires_trigger = true;
    post(f);
    expect(f.q.broken_roadmap).toBe(0);
    expect(f.q.cdc_relations).toBe(45); // The remaining broken burdens are the same failure.
  });

  it('allows a new burdens cycle to fail once after recovery', () => {
    const f = setup('jxsi', 'erc', { burdens_broken: true });
    post(f);
    expect(f.q.cdc_relations).toBe(60);
    f.q.burdens_restart_trigger = true;
    f.adapter.goToScene('burdens_start');
    pick(f, 'burdens_start.burdens_continue');
    expect(f.q.burdens_broken).toBe(false);
    f.q.burdens_broken = true;
    post(f);
    expect(f.q.cdc_relations).toBe(35);
    post(f);
    expect(f.q.cdc_relations).toBe(35);
  });

  it('recognizes an unmet annual burdens deadline on its actual monthly transition', () => {
    const f = setup('jxsi', 'erc', { burdens_time: 1, burdens_done: 3, month_actions: 1 });
    post(f);
    expect(f.q.burdens_broken).toBe(1);
    expect(f.q.cdc_relations).toBe(60);
  });

  it('delivers the roadmap notice before offering rupture, then queues it without another month', () => {
    const f = setup('jxsi', 'erc', { roadmap_year: 2015, month_actions: 1, cdc_relations: 70 });
    f.adapter.goToScene('post_event');
    expect(f.q.broken_roadmap_trigger).toBe(true);
    expect(f.q.cdc_relations).toBe(30);
    expect(f.q.coalition_rupture_trigger).toBe(false);
    const month = f.q.month;
    pick(f, 'broken_roadmap');
    expect(f.q.roadmap_year).toBe(0);
    expect(f.q.broken_roadmap_trigger).toBe(false);
    pick(f, 'root');
    expect(f.q.month).toBe(month);
    expect(f.q.cdc_relations).toBe(30);
    expect(f.q.coalition_rupture_trigger).toBe(true);
    open(f);
  });
});

describe('forced unity-list rupture', () => {
  it.each([31, 32, 39, 40])('uses the 40-point rescue threshold from relations %s', relations => {
    const f = setup('jxsi', 'erc', {
      burdens_broken: true, unity_failed_mandates: ['burdens'], cdc_relations: relations,
    });
    post(f);
    expect(f.q.coalition_rupture_trigger).toBe(relations < 40);
    if (relations < 40) {
      const frame = open(f);
      expect(frame.choices.find(c => c.id === rupture + '.policy')?.canChoose).toBe(relations >= 32);
      expect(frame.choices.some(c => c.id === rupture + '.elections')).toBe(false);
    }
  });

  it('checks each constituent even when average relations remain excellent', () => {
    const f = setup('jxsi', 'cup', {
      burdens_broken: true, unity_failed_mandates: ['burdens'], cdc_relations: 95, erc_relations: 30,
    });
    post(f);
    expect(f.q.jxsi_dissent).toBeLessThan(30);
    expect(f.q.coalition_rupture_target).toBe('erc');
    expect(f.q.coalition_rupture_list).toBe('jxsi');
    open(f);
  });

  it('preserves the list after delivering an affordable +8 policy concession', () => {
    const f = setup('jxsi', 'erc', {
      burdens_broken: true, unity_failed_mandates: ['burdens'], cdc_relations: 35,
    });
    post(f);
    open(f);
    pick(f, rupture + '.policy');
    expect(f.q.cdc_relations).toBe(43);
    expect(f.q.tax_policy_level).toBe(0);
    expect(f.q.coalition_rupture_saved).toBe(true);
    post(f);
    expect(f.q.coalition_rupture_trigger).toBe(false);
    expect(f.q.player_in_jxsi).toBe(true);
  });

  for (const list of ['jxsi', 'jxcat']) for (const player of ['erc', 'cup']) {
    it(`${list}/${player}: forced dissolution conserves seats and offices without a voluntary-exit hit`, () => {
      const f = setup(list, player, { broken_roadmap: true, cdc_relations: 65 });
      post(f);
      const holders = Object.fromEntries(offices.map(office => [office, [f.q[office], f.q[office + '_gender']]]));
      const relations = [f.q.cdc_relations, f.q.erc_relations, f.q.cup_relations];
      open(f);
      pick(f, rupture + '.leave');
      expect(f.q[list + '_dissolved']).toBe(true);
      expect(f.q['player_in_' + list]).toBe(false);
      expect([f.q.cdc_relations, f.q.erc_relations, f.q.cup_relations]).toEqual(relations);
      const core = list === 'jxcat' ? 'jxcat' : 'cdc';
      expect(f.q.cat_coalition).toEqual([core, 'erc', 'cup']);
      expect(f.q[core + '_parlament_s'] + f.q.erc_parlament_s + f.q.cup_parlament_s).toBe(70);
      for (const office of offices) {
        expect([f.q[office], f.q[office + '_gender']]).toEqual(holders[office]);
        const affiliation = f.q[office + '_affiliation'];
        expect(f.q[office + '_party']).toBe(affiliation === 'cdc' ? core : affiliation);
      }
      for (const province of f.q.parlament_constituencies) for (const demo of f.q.parlament_demographics) {
        const suffix = `_parlament_${province}_${demo}_support`;
        expect(f.q['abstain' + suffix]).toBe(81);
        expect(f.q[core + suffix] + f.q['erc' + suffix] + f.q['cup' + suffix]).toBeCloseTo(19);
      }
    });
  }

  it('keeps an internal law concession alive and penalizes an actual missed deadline', () => {
    const f = setup('jxsi', 'cup', {
      burdens_broken: true, unity_failed_mandates: ['burdens'], erc_relations: 35,
      parlament_sovereignty_declaration: true,
    });
    post(f);
    open(f);
    pick(f, rupture + '.independence');
    expect(f.q.erc_relations).toBe(43);
    const record = f.q.coalition_indy_commitments[0];
    post(f);
    expect(record.status).toBe('pending');
    record.deadline = f.q.year * 12 + f.q.month;
    post(f);
    expect(record.status).toBe('broken');
    f.adapter.goToScene('post_event.events_choice');
    pick(f, 'generalitat_coalition_indy_breach');
    expect(record.status).toBe('penalized');
    expect(f.q.erc_relations).toBe(28);
  });

  it('does not create or retain rupture events while caretaker', () => {
    const f = setup('jxsi', 'erc', {
      cat_caretaker_gov: true, broken_roadmap: true, cdc_relations: 65,
      coalition_rupture_trigger: true, coalition_rupture_target: 'cdc', coalition_rupture_list: 'jxsi',
      month_actions: 1,
    });
    post(f);
    expect(f.q.cdc_relations).toBe(30);
    expect(f.q.coalition_rupture_trigger).toBe(false);
    expect(f.q.month).toBe(4);
    const frame = f.adapter.goToScene('post_event.events_choice');
    expect(frame.choices.some(c => c.id === rupture)).toBe(false);
    f.q.cat_caretaker_gov = false;
    post(f);
    expect(f.q.coalition_rupture_trigger).toBe(true);
    expect(f.q.cdc_relations).toBe(30);
  });

  it('leaves ordinary cabinet rupture thresholds unchanged', () => {
    const f = setup('jxsi', 'erc', { player_in_jxsi: false, coalition_rupture_recheck: true });
    f.q.cat_coalition = ['erc', 'cdc'];
    f.q.cdc_relations = 35;
    f.q.burdens_broken = true;
    post(f);
    expect(f.q.coalition_rupture_trigger).toBe(false);
    f.q.cdc_relations = 29;
    f.q.coalition_rupture_recheck = true;
    post(f);
    expect(f.q.coalition_rupture_trigger).toBe(true);
    expect(f.q.coalition_rupture_list).toBe('');
    expect(open(f).choices.some(c => c.id === rupture + '.elections')).toBe(true);
  });

  it('still expels the player through the ordinary government breakup route', () => {
    const f = setup('jxsi', 'erc', {
      player_in_jxsi: false, coalition_rupture_recheck: true, cdc_relations: 29,
    });
    f.q.cat_coalition = ['cdc', 'erc', 'cup'];
    f.q.president_party = 'cdc';
    for (const office of offices) f.q[office + '_party'] = f.q[office + '_affiliation'];
    post(f);
    open(f);
    pick(f, rupture + '.leave');
    expect(f.q.cat_coalition).toEqual(['cdc']);
    expect(f.q.erc_in_gob).toBe(false);
    expect(f.q.cup_in_gob).toBe(false);
    expect(f.q.president).toBe('Artur Mas');
    expect(f.q.cdc_relations).toBe(29);
    expect(f.q.burdens_time).toBe(0);
  });

  it('allows ERC to rebuild the capped relationship after forced dissolution', () => {
    const f = setup('jxsi', 'erc', { broken_roadmap: true, cdc_relations: 65, unio_split: true });
    post(f);
    open(f);
    pick(f, rupture + '.leave');
    pick(f, 'jxsi_dissolution.separate');
    expect(f.q.cdc_relations).toBe(30);
    expect(f.q.coalition_rupture_trigger).toBe(false);
    f.adapter.goToScene('erc_inter_party');
    pick(f, 'erc_inter_party.cdc_relations');
    expect(f.q.cdc_relations).toBeGreaterThan(30);
  });
});
