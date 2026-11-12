import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import type { Frame } from '../src/engine/types';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
const cooperating = ['ciu', 'cdc', 'dl', 'pdcat', 'jxcat', 'junts', 'jxsi', 'erc', 'cup'];
type Q = Record<string, any>;

function setup(overrides: Q = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Q;
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    cat_caretaker_gov: false, party_resources: 5, parlament_non_binding_consultations: true,
    consultation_pending: true, consultation_happened: false, consultation_question: 2,
    health_minister_party: 'ciu', education_minister_party: 'ciu',
    generalitat_health_timer: 0, generalitat_education_timer: 0,
    independence_trust: 50, independence_movement: 60, burdens_done: 0,
    election_history_events: [], consultation_day_trigger: true,
    year: 2014, month: 11, week: 2, psc_leader: 'Miquel Iceta',
    active_mods: {}, mod_log: [],
    ...overrides,
  });
  return { adapter, q };
}

function enter(adapter: DendryAdapter, scene: string) {
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  try {
    const frame = adapter.goToScene(scene);
    expect(errors.mock.calls).toEqual([]);
    return frame;
  } finally { errors.mockRestore(); }
}

function choose(adapter: DendryAdapter, frame: Frame, id: string) {
  const index = frame.choices.findIndex(choice => choice.id === id);
  expect(index).toBeGreaterThanOrEqual(0);
  expect(frame.choices[index].canChoose).toBe(true);
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  try {
    const next = adapter.choose(index);
    expect(errors.mock.calls).toEqual([]);
    return next;
  } finally { errors.mockRestore(); }
}

function clearLocal(q: Q) {
  for (const party of cooperating) for (const area of ['rb', 'interior']) q[`${party}_local_${area}_pct`] = 0;
  q.local_barcelona_wp = 'psc';
}

describe('consultation preparation', () => {
  it('uses the initialized 2011 local holdings before any municipal election', () => {
    const { adapter, q } = setup();
    expect(q.consultation_health_preparations).toBe(0);
    expect(q.consultation_education_preparations).toBe(0);
    expect(q.local_barcelona_wp).toBe('ciu');
    expect(q.ciu_local_rb_pct).toBe(10.3);
    expect(q.ciu_local_interior_pct + q.erc_local_interior_pct).toBeCloseTo(85.9);
    enter(adapter, 'nonbinding_consultation');
    expect(q.consultation_preparation).toBe(6);
  });

  it.each(['health', 'education'])('charges for two %s preparations and blocks further uses', ministry => {
    const { adapter, q } = setup({ [`${ministry}_minister_party`]: 'erc' });
    const id = `generalitat_${ministry}_card.prepare_consultation`;
    for (let used = 0; used < 2; used++) {
      q[`generalitat_${ministry}_timer`] = 0;
      const before = q.month_actions;
      const menu = enter(adapter, `generalitat_${ministry}_card`);
      choose(adapter, menu, id);
      expect(q.party_resources).toBe(4 - used);
      expect(q[`consultation_${ministry}_preparations`]).toBe(used + 1);
      expect(q.month_actions).toBe(before + 1);
      expect(q.burdens_done).toBe(used + 1);
    }
    const menu = enter(adapter, `generalitat_${ministry}_card`);
    expect(menu.choices.find(choice => choice.id === id)?.canChoose).toBe(false);
    enter(adapter, id);
    expect(q.party_resources).toBe(3);
    expect(q[`consultation_${ministry}_preparations`]).toBe(2);
    expect(q.burdens_done).toBe(2);
  });

  it.each(['health', 'education'])('gates %s preparation on the law, resources and an unfinished consultation', ministry => {
    const { adapter, q } = setup({ [`${ministry}_minister_party`]: 'erc' });
    const id = `generalitat_${ministry}_card.prepare_consultation`;
    for (const changes of [
      { parlament_non_binding_consultations: false, consultation_happened: false, party_resources: 5 },
      { parlament_non_binding_consultations: true, consultation_happened: true, party_resources: 5 },
      { parlament_non_binding_consultations: true, consultation_happened: false, party_resources: 0 },
    ]) {
      Object.assign(q, changes);
      const menu = enter(adapter, `generalitat_${ministry}_card`);
      const choice = menu.choices.find(choice => choice.id === id);
      expect(choice?.canChoose || false).toBe(false);
      enter(adapter, id);
      expect(q.party_resources).toBe(changes.party_resources);
      expect(q[`consultation_${ministry}_preparations`]).toBe(0);
      expect(q.burdens_done).toBe(0);
    }
  });

  it('honours joint-list access, preserves preparation through saves, and excludes controlled NPC allowances', () => {
    const { adapter, q } = setup({ player_in_jxsi: true, health_minister_party: 'jxsi', education_minister_party: 'jxsi' });
    delete q.consultation_health_preparations; // Saves made before this addition.
    choose(adapter, enter(adapter, 'generalitat_health_card'), 'generalitat_health_card.prepare_consultation');
    expect(q.consultation_health_preparations).toBe(1);
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());
    enter(restored, 'nonbinding_consultation');
    expect(restored.qualities.consultation_preparation).toBe(5);
    expect(restored.qualities.consultation_health_preparations).toBe(1);
  });

  it.each(['ciu', 'cdc', 'dl', 'pdcat', 'jxcat', 'junts', 'jxsi', 'erc', 'cup', 'psc', 'indp.'])(
    'applies the NPC allowance once for %s and does not stack it with completed work', owner => {
      const { adapter, q } = setup({ player_party: 'cup', health_minister_party: owner, education_minister_party: 'psc' });
      clearLocal(q);
      enter(adapter, 'nonbinding_consultation');
      expect(q.consultation_preparation).toBe(cooperating.includes(owner) && owner !== 'cup' ? 1 : 0);
      q.consultation_health_preparations = 2;
      enter(adapter, 'nonbinding_consultation');
      expect(q.consultation_preparation).toBe(2);
    },
  );

  it.each([[24.9, 0], [25, 1], [49.9, 1], [50, 2]])('scores pooled local holdings at %s%%', (share, points) => {
    const { adapter, q } = setup({ health_minister_party: 'erc', education_minister_party: 'erc' });
    clearLocal(q);
    q.erc_local_rb_pct = share / 2;
    q.cdc_local_rb_pct = share / 2;
    q.cup_local_interior_pct = share;
    enter(adapter, 'nonbinding_consultation');
    expect(q.consultation_preparation).toBe(points * 2);
  });

  it.each([['psc', 0], ['bcomu', 1], ['icv', 1], ['erc', 2], ['jxcat', 2]])('scores Barcelona under %s', (owner, points) => {
    const { adapter, q } = setup({ health_minister_party: 'erc', education_minister_party: 'erc' });
    clearLocal(q);
    q.local_barcelona_wp = owner;
    enter(adapter, 'nonbinding_consultation');
    expect(q.consultation_preparation).toBe(points);
  });

  it('routes failure without calculating or rendering results or awarding completion', () => {
    const { adapter, q } = setup({ health_minister_party: 'erc', education_minister_party: 'erc', roadmap_year: 2014 });
    clearLocal(q);
    q.ciu_local_interior_pct = 50;
    q.local_barcelona_wp = 'bcomu'; // Exactly 3: the upper failure boundary.
    const menu = enter(adapter, 'nonbinding_consultation');
    expect(q.consultation_preparation).toBe(3);
    expect(menu.choices.map(choice => choice.id)).toEqual(['nonbinding_consultation.failure']);
    const failed = choose(adapter, menu, 'nonbinding_consultation.failure');
    expect(failed.html).not.toContain('data-widget="result-bars"');
    expect(q.consultation_total_votes).toBeUndefined();
    expect(q.consultation_happened).toBe(true);
    expect(q.consultation_pending).toBe(false);
    expect(q.burdens_done).toBe(0);
    expect(q.game_achievement_calcotada).toBeFalsy();
    expect(q.election_history_events).toEqual([]);
    expect(q.independence_trust).toBe(35);
    expect(q.independence_movement).toBe(55);
  });

  it('allows exactly four points to reach the troubled result route', () => {
    const { adapter, q } = setup({ health_minister_party: 'erc', education_minister_party: 'erc' });
    const menu = enter(adapter, 'nonbinding_consultation');
    expect(q.consultation_preparation).toBe(4);
    expect(menu.choices.map(choice => choice.id)).toEqual(['nonbinding_consultation.double']);
    const results = choose(adapter, menu, 'nonbinding_consultation.double');
    expect(results.html).toContain('data-widget="result-bars"');
    expect(results.choices.some(choice => choice.id === 'nonbinding_consultation.troubled')).toBe(true);
  });

  it.each([1, 2, 3])('reduces participation and conserves the modeled electorate for question %s', question => {
    const branch = ['direct', 'double', 'convoluted'][question - 1];
    const { adapter: good, q: success } = setup({ consultation_question: question });
    const { adapter: bad, q: troubled } = setup({ consultation_question: question, health_minister_party: 'erc' });
    choose(good, enter(good, 'nonbinding_consultation'), `nonbinding_consultation.${branch}`);
    const results = choose(bad, enter(bad, 'nonbinding_consultation'), `nonbinding_consultation.${branch}`);
    expect(success.consultation_preparation).toBe(6);
    expect(troubled.consultation_preparation).toBe(5);
    expect(results.html).toContain('data-widget="result-bars"');
    expect(troubled.consultation_total_votes).toBeLessThan(success.consultation_total_votes);
    expect(troubled.consultation_participation_pc).toBeLessThan(success.consultation_participation_pc);
    expect(troubled.consultation_total_votes + troubled.consultation_abst)
      .toBe(success.consultation_total_votes + success.consultation_abst);
    expect(troubled.consultation_total_votes).toBeCloseTo(success.consultation_total_votes * .8, -1);
    expect(results.choices.some(choice => choice.id === 'nonbinding_consultation.after')).toBe(false);
    const aftermath = choose(bad, results, 'nonbinding_consultation.troubled');
    expect(aftermath.sceneId).toBe('nonbinding_consultation.troubled');
    expect(troubled.independence_trust).toBe(45);
    expect(troubled.jxsi_formation_countdown).toBe(5);
    expect(troubled.plebiscite_election_countdown).toBe(4);
  });

  it('keeps the existing successful aftermath and completion rewards', () => {
    const { adapter, q } = setup();
    const results = choose(adapter, enter(adapter, 'nonbinding_consultation'), 'nonbinding_consultation.double');
    expect(results.choices.some(choice => choice.id === 'nonbinding_consultation.troubled')).toBe(false);
    choose(adapter, results, 'nonbinding_consultation.after');
    expect(q.independence_trust).toBe(50);
    expect(q.independence_movement).toBe(60);
    expect(q.burdens_done).toBe(1);
    expect(q.game_achievement_calcotada).toBeTruthy();
    expect(q.election_history_events).toHaveLength(1);
    expect(q.unio_breakup_countdown).toBe(3);
    expect(q.jxsi_formation_countdown).toBe(5);
    expect(q.plebiscite_election_countdown).toBe(4);
  });

  it.each(['ciu', 'erc'])('preserves an actual No victory with Health held by %s', healthParty => {
    const { adapter, q } = setup({ consultation_question: 1, health_minister_party: healthParty });
    for (const province of q.parlament_constituencies) for (const demo of q.parlament_demographics) {
      for (const party of [...q.parties, 'abstain']) q[`${party}_parlament_${province}_${demo}_support`] = 0;
      q[`unio_parlament_${province}_${demo}_support`] = 40;
      q[`abstain_parlament_${province}_${demo}_support`] = 60;
    }
    const results = choose(adapter, enter(adapter, 'nonbinding_consultation'), 'nonbinding_consultation.direct');
    expect(results.choices.map(choice => choice.id)).toEqual(['nonbinding_consultation.after_no']);
    choose(adapter, results, 'nonbinding_consultation.after_no');
    expect(q.independence_trust).toBe(20);
    expect(q.independence_movement).toBe(30);
    expect(q.parlament_snap_election_countdown).toBe(1);
  });
});
