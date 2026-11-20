import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
type Q = Record<string, any>;

function isolate(adapter: DendryAdapter) {
  (adapter.engine as any).game.tagLookup.event = { consultation_trial: true };
  (adapter.engine as any).game.tagLookup.election = {};
}

function setup(overrides: Q = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Q;
  Object.assign(q, {
    year: 2014, month: 11, week: 1, rubicon: false,
    month_actions: 0, month_actions_last: 0, player_party: 'erc',
    parlament_current_ciu: 'ciu', ciu_roadmap: 2, erc_roadmap: 2,
    consultation_day_trigger: true, consultation_question: 2,
    government_does_things_timer: 100, ...overrides,
  });
  isolate(adapter);
  return { adapter, q };
}

function enter(adapter: DendryAdapter, id: string) {
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  try {
    const frame = adapter.goToScene(id);
    expect(errors.mock.calls).toEqual([]);
    return frame;
  } finally { errors.mockRestore(); }
}

function hold(adapter: DendryAdapter, q: Q) {
  enter(adapter, 'nonbinding_consultation');
  expect(q.consultation_preparation).toBeGreaterThanOrEqual(4);
  q.countdowns = ['consultation_trial'];
}

function convict(adapter: DendryAdapter, q: Q) {
  q.consultation_trial_trigger = true;
  return enter(adapter, 'consultation_trial');
}

function continueVerdict(adapter: DendryAdapter) {
  let frame = adapter.choose(0);
  while (/^consultation_trial\.(ciu|erc|cup)_leadership$/.test(frame.sceneId)) frame = adapter.choose(0);
  return frame;
}

describe('consultation verdict', () => {
  it('shows changed party leadership in order before calling elections', () => {
    const { adapter, q } = setup({ vicepresident: 'Oriol Junqueras', vicepresident_party: 'erc' });
    hold(adapter, q);
    convict(adapter, q);
    const time = q.time;
    const first = adapter.choose(0);
    expect(first.sceneId).toBe('consultation_trial.ciu_leadership');
    expect(first.html).toContain('Artur Mas');
    expect(first.html).toContain('Carles Puigdemont');
    expect(q.cat_caretaker_gov).not.toBe(true);
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());
    isolate(restored);
    const second = restored.choose(0);
    expect(second.sceneId).toBe('consultation_trial.erc_leadership');
    expect(second.html).toContain('Oriol Junqueras');
    expect(second.html).toContain('Marta Rovira');
    expect(second.html.match(/Artur Mas/g)).toHaveLength(first.html.match(/Artur Mas/g)!.length);
    expect(restored.qualities.time).toBe(time);
    expect(restored.choose(0)).toMatchObject({ sceneId: 'snap_election', title: 'New Parlament elections called!' });
    expect(restored.qualities.cat_caretaker_gov).toBe(true);
  });

  it('skips leadership pages when only public officeholders are replaced', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    q.ciu_leader = 'Carles Puigdemont';
    convict(adapter, q);
    expect(q.education_minister).not.toBe('Irene Rigau');
    expect(adapter.choose(0).sceneId).toBe('snap_election');
  });

  it('shows an ERC-only change and returns to events when the president is unaffected', () => {
    const { adapter, q } = setup({ vicepresident: 'Oriol Junqueras', vicepresident_party: 'erc' });
    hold(adapter, q);
    Object.assign(q, { president: 'Carles Puigdemont', ciu_leader: 'Carles Puigdemont' });
    convict(adapter, q);
    expect(adapter.choose(0).sceneId).toBe('consultation_trial.erc_leadership');
    const frame = adapter.choose(0);
    expect(frame.sceneId.startsWith('consultation_trial.')).toBe(false);
    expect(q.cat_caretaker_gov).not.toBe(true);
  });

  it('shows a CUP leadership change without adding pages for ministries', () => {
    const { adapter, q } = setup({ education_minister: 'Anna Gabriel', education_minister_party: 'cup' });
    hold(adapter, q);
    Object.assign(q, { president: 'Carles Puigdemont', ciu_leader: 'Carles Puigdemont', cup_leader: 'Anna Gabriel' });
    convict(adapter, q);
    const frame = adapter.choose(0);
    expect(frame.sceneId).toBe('consultation_trial.cup_leadership');
    expect(frame.html).toContain('Anna Gabriel');
    expect(frame.html).toContain(q.cup_leader);
    expect(adapter.choose(0).sceneId.startsWith('consultation_trial.')).toBe(false);
  });

  it('exposes the original officials and their genders for prose after cabinet changes', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    const officials = structuredClone(q.consultation_officials);
    q.president = 'Carles Puigdemont';
    q.president_gender = 'm';
    convict(adapter, q);
    for (const [index, role] of ['president', 'vp', 'education'].entries()) {
      const field = `consultation_trial_${role}`;
      expect(q[field]).toBe(officials[index].name);
      expect(q[`${field}_gender`]).toBe(officials[index].gender);
      const bindings = JSON.parse(readFileSync(resolve(import.meta.dirname, '../../source/data/localization.json'), 'utf8'));
      expect(bindings.grammarBindings[field]).toBe(`${field}_gender`);
    }
  });

  it('freezes the three officeholders and affiliations once and preserves them through save/load', () => {
    const { adapter, q } = setup({ vicepresident: 'Oriol Junqueras', vicepresident_party: 'jxsi', vicepresident_affiliation: 'erc' });
    hold(adapter, q);
    const officials = structuredClone(q.consultation_officials);
    expect(officials.map((p: Q) => p.name)).toEqual(['Artur Mas', 'Oriol Junqueras', 'Irene Rigau']);
    expect(officials[1].affiliation).toBe('erc');
    q.president = 'Carles Puigdemont';
    enter(adapter, 'nonbinding_consultation');
    expect(q.consultation_officials).toEqual(officials);
    expect(q.countdowns.filter((id: string) => id === 'consultation_trial')).toHaveLength(1);
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());
    expect(restored.qualities.consultation_officials).toEqual(officials);
    expect(restored.qualities.consultation_trial_countdown).toBe(18);
  });

  it('does not prosecute a consultation cancelled for insufficient preparation', () => {
    const { adapter, q } = setup();
    for (const party of q.parties) for (const area of ['rb', 'interior']) q[`${party}_local_${area}_pct`] = 0;
    q.local_barcelona_wp = 'psc';
    enter(adapter, 'nonbinding_consultation');
    expect(q.consultation_preparation).toBeLessThan(4);
    expect(q.consultation_officials).toBeUndefined();
    expect(q.consultation_trial_countdown).toBeUndefined();
  });

  it('appears exactly eighteen calendar months later and holds the pending time advance', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    for (let month = 1; month <= 18; month++) {
      q.month_actions = 1;
      const frame = enter(adapter, 'post_event');
      expect(frame.choices.some(c => c.id === 'consultation_trial')).toBe(month === 18);
    }
    expect([q.year, q.month]).toEqual([2016, 5]);
    q.month_actions = 1;
    const pending = enter(adapter, 'post_event');
    expect([q.year, q.month]).toEqual([2016, 5]);
    const index = pending.choices.findIndex(c => c.id === 'consultation_trial');
    const verdict = adapter.choose(index);
    expect(verdict.sceneId).toBe('consultation_trial');
    expect(q.consultation_verdict.year).toBe(2016);
    const savedVerdict = structuredClone(q.consultation_verdict);
    enter(adapter, 'consultation_trial');
    expect(q.consultation_verdict).toEqual(savedVerdict);
    const after = enter(adapter, 'post_event');
    expect(after.choices.some(c => c.id === 'consultation_trial')).toBe(false);
  });

  it('counts Rubicon weeks and triggers across the December rollover', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    Object.assign(q, { rubicon: true, year: 2016, month: 12, week: 1, consultation_trial_countdown: 4 });
    for (let week = 2; week <= 4; week++) {
      q.month_actions_last = 2;
      q.month_actions = 1;
      const frame = enter(adapter, 'post_event');
      expect(q.week).toBe(week);
      expect(q.consultation_trial_countdown).toBe(5 - week);
      expect(frame.choices.some(c => c.id === 'consultation_trial')).toBe(false);
    }
    q.month_actions_last = 2;
    q.month_actions = 1;
    const frame = enter(adapter, 'post_event');
    expect([q.year, q.month, q.week]).toEqual([2017, 1, 1]);
    expect(q.consultation_trial_countdown).toBe(-1);
    expect(frame.choices.some(c => c.id === 'consultation_trial')).toBe(true);
  });

  it.each([
    ['ciu', 0, 'Francesc Homs', 'm'], ['cdc', 1, 'Francesc Homs', 'm'],
    ['pdcat', 2, 'Carles Puigdemont', 'm'], ['junts', 3, 'Carles Puigdemont', 'm'],
  ])('replaces Mas using the live %s roadmap %s', (party, roadmap, name, gender) => {
    const { adapter, q } = setup();
    hold(adapter, q);
    Object.assign(q, { parlament_current_ciu: party, [`${party}_roadmap`]: roadmap, [`${party}_leader`]: 'Artur Mas', president_party: party });
    convict(adapter, q);
    expect(q.president).toBe(name);
    expect(q.president_gender).toBe(gender);
    expect(q[`${party}_leader`]).toBe(name);
    expect(q[`${party}_leader_gender`]).toBe(gender);
    expect(q.parlament_current_ciu_leader).toBe(name);
    expect(q.education_minister).toBe('Meritxell Ruiz');
    expect(q.president_party).toBe(party);
    const leadership = adapter.choose(0);
    expect(leadership.sceneId).toBe('consultation_trial.ciu_leadership');
    expect(leadership.html).toContain(name);
    expect(adapter.choose(0).sceneId).toBe('snap_election');
  });

  it.each([[2018, 'Marta Rovira', 'f'], [2019, 'Pere Aragonès', 'm']])('replaces Junqueras using the verdict year %s', (year, name, gender) => {
    const { adapter, q } = setup({ vicepresident: 'Oriol Junqueras', vicepresident_party: 'erc', vicepresident_gender: 'm' });
    hold(adapter, q);
    q.year = year;
    q.economy_minister = 'Oriol Junqueras';
    q.economy_minister_party = 'erc';
    convict(adapter, q);
    expect(q.erc_leader).toBe(name);
    expect(q.erc_leader_gender).toBe(gender);
    expect(q.vicepresident).toBe(name);
    expect(q.economy_minister).toBe(name);
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());
    isolate(restored);
    restored.qualities.year = 2019;
    enter(restored, 'post_event');
    expect(restored.qualities.erc_leader).toBe(name);
  });

  it('promotes Munté as acting president and awards the achievement on investiture', () => {
    const { adapter, q } = setup({ ciu_roadmap: 1 });
    hold(adapter, q);
    q.vicepresident = 'Neus Munté';
    convict(adapter, q);
    expect(q.president).toBe('Neus Munté');
    expect(q.vicepresident).not.toBe('Neus Munté');
    expect(q.vicepresident).not.toBe('Joana Ortega');
    expect(q.ciu_leader).toBe('Francesc Homs');
    expect(continueVerdict(adapter).sceneId).toBe('snap_election');
    adapter.choose(0);
    expect(q.cat_caretaker_gov).toBe(true);
    expect(q.game_achievement_moneda_de_canvi).toBeUndefined();

    Object.assign(q, {
      cat_coalition: ['ciu'], cat_coalition_support: [], cat_coalition_abstain: [],
      parlament_coalition_proposal_president: 'ciu',
      parlament_coalition_proposal_president_name: 'Neus Munté',
      parlament_coalition_proposal_president_gender: 'f',
    });
    enter(adapter, 'parlament_coalition_resolve');
    expect(q.cat_caretaker_gov).toBe(false);
    expect(q.game_achievement_moneda_de_canvi).toBe(1);
  });

  it('updates shared-list leadership and preserves constituent affiliations and ministry allocations', () => {
    const { adapter, q } = setup({
      president_party: 'jxsi', president_affiliation: 'cdc',
      vicepresident: 'Oriol Junqueras', vicepresident_party: 'jxsi', vicepresident_affiliation: 'erc',
      education_minister_party: 'jxsi', education_minister_affiliation: 'cdc',
      jxsi_leader: 'Artur Mas', cat_coalition: ['jxsi'],
    });
    hold(adapter, q);
    convict(adapter, q);
    expect(q.jxsi_leader).toBe('Carles Puigdemont');
    expect(q.erc_leader).toBe('Marta Rovira');
    expect(q.vicepresident).toBe('Marta Rovira');
    expect(q.vicepresident_party).toBe('jxsi');
    expect(q.vicepresident_affiliation).toBe('erc');
    expect(q.education_minister).toBe('Meritxell Ruiz');
    expect(q.education_minister_affiliation).toBe('cdc');
    expect(q.cat_coalition).toEqual(['jxsi']);
  });

  it('does not remove Junqueras when he held none of the prosecuted offices', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    convict(adapter, q);
    expect(q.erc_leader).toBe('Oriol Junqueras');
  });

  it('removes a former VP who has since become president, and a former education minister now serving as Speaker', () => {
    const { adapter, q } = setup({ vicepresident: 'Oriol Junqueras', vicepresident_party: 'erc' });
    hold(adapter, q);
    Object.assign(q, { president: 'Oriol Junqueras', president_party: 'erc', vicepresident: 'Carles Puigdemont', vicepresident_party: 'ciu', vicepresident_gender: 'm', speaker: 'Irene Rigau', snap_budget_distrust: true, budget_failed_times: 2 });
    const prose = document.createElement('div');
    prose.innerHTML = convict(adapter, q).html;
    expect(prose.textContent).toContain('Oriol Junqueras is now legally forced to step down');
    expect(q.president).toBe('Carles Puigdemont');
    expect(q.president_party).toBe('ciu');
    expect(q.vicepresident).not.toBe('Carles Puigdemont');
    expect(q.speaker).not.toBe('Irene Rigau');
    expect(q.education_minister).not.toBe('Irene Rigau');
    const frame = continueVerdict(adapter);
    expect(frame.sceneId).toBe('snap_election');
    prose.innerHTML = frame.html;
    expect(prose.textContent).toContain('has forced Oriol Junqueras from office');
    expect(prose.textContent).toContain('Carles Puigdemont is set to lead it following succession rules');
    expect(prose.textContent).not.toContain('Repeated failures to pass');
    expect(q.cat_caretaker_gov).toBe(true);
    const catalog = JSON.parse(readFileSync(resolve(import.meta.dirname, '../../source/locales/ca/content.json'), 'utf8'));
    const translated = adapter.refreshLocale('ca', catalog)!;
    expect(translated.title).toBe(catalog[frame.title!]);
    prose.innerHTML = translated.html;
    expect(prose.textContent).toContain('ha obligat Oriol Junqueras a deixar el càrrec');
    adapter.refreshLocale('en', {});
    adapter.choose(0);
    // Older saves retain the verdict's routing flag after the trial is over.
    q.consultation_trial_snap_election = true;
    q.budget_failed_times = 1;
    const laterElection = enter(adapter, 'snap_election');
    expect(laterElection.title).toBe('New Parlament elections called!');
    expect(laterElection.html).toContain('Repeated failures to pass');
  });

  it('promotes an eligible current VP with her party and affiliation, then uses the ordinary snap-election scene', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    Object.assign(q, { year: 2016, month: 12, vicepresident: 'Marta Rovira', vicepresident_gender: 'f', vicepresident_party: 'jxsi', vicepresident_affiliation: 'erc' });
    const time = q.time;
    const votes = q.parlament_constituencies.flatMap((area: string) =>
      q.parlament_demographics.map((group: string) => q[`erc_parlament_${area}_${group}_support`]));
    convict(adapter, q);
    expect(q.president).toBe('Marta Rovira');
    expect(q.president_gender).toBe('f');
    expect(q.president_party).toBe('jxsi');
    expect(q.president_affiliation).toBe('erc');
    expect(q.vicepresident).not.toBe('Marta Rovira');
    expect(q.ciu_leader).toBe('Carles Puigdemont');
    const frame = continueVerdict(adapter);
    expect(frame.sceneId).toBe('snap_election');
    expect(q.cat_caretaker_gov).toBe(true);
    expect([q.next_election_year, q.next_election_month, q.next_election_time]).toEqual([2017, 2, time + 2]);
    expect(q.parlament_constituencies.flatMap((area: string) =>
      q.parlament_demographics.map((group: string) => q[`erc_parlament_${area}_${group}_support`]))).toEqual(votes);
  });

  it.each(['convicted', 'absent'])('uses an existing alternative when the current VP is %s', (vp) => {
    const { adapter, q } = setup();
    hold(adapter, q);
    if (vp === 'absent') q.vicepresident = null;
    convict(adapter, q);
    expect(q.president).toBe('Carles Puigdemont');
    expect(q.president_party).toBe('ciu');
    expect(continueVerdict(adapter).sceneId).toBe('snap_election');
    expect(q.cat_caretaker_gov).toBe(true);
  });

  it('does not call elections when the current president is not among the defendants', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    q.president = 'Carles Puigdemont';
    const election = [q.next_election_year, q.next_election_month, q.next_election_time];
    convict(adapter, q);
    expect(continueVerdict(adapter).sceneId).not.toBe('snap_election');
    expect(q.cat_caretaker_gov).not.toBe(true);
    expect([q.next_election_year, q.next_election_month, q.next_election_time]).toEqual(election);
  });

  it('replaces a disqualified caretaker without rescheduling elections or counting an acting president', () => {
    const { adapter, q } = setup({ ciu_roadmap: 1 });
    hold(adapter, q);
    Object.assign(q, {
      cat_caretaker_gov: true, vicepresident: 'Marta Rovira', vicepresident_party: 'erc', vicepresident_gender: 'f',
      next_election_year: 2014, next_election_month: 12, next_election_week: 1, next_election_time: q.time + 1,
    });
    const time = q.time;
    convict(adapter, q);
    expect(q.president).toBe('Marta Rovira');
    expect(q.president_party).toBe('erc');
    expect(q.vicepresident).not.toBe('Marta Rovira');
    expect(q.ciu_leader).toBe('Francesc Homs');
    expect(continueVerdict(adapter).sceneId).not.toBe('snap_election');
    expect(q.cat_caretaker_gov).toBe(true);
    expect([q.next_election_year, q.next_election_month, q.next_election_week, q.next_election_time]).toEqual([2014, 12, 1, time + 1]);
    expect(q.woman_generalitat_presidenta_ever).toBe(false);
  });

  it('skips the presidential snap-election path during Rubicon', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    Object.assign(q, { rubicon: true, vicepresident: 'Marta Rovira', vicepresident_party: 'erc', vicepresident_gender: 'f' });
    const election = [q.next_election_year, q.next_election_month, q.next_election_time];
    convict(adapter, q);
    expect(q.president).toBe('Carles Puigdemont');
    expect(q.vicepresident).toBe('Marta Rovira');
    expect(continueVerdict(adapter).sceneId).not.toBe('snap_election');
    expect(q.cat_caretaker_gov).not.toBe(true);
    expect([q.next_election_year, q.next_election_month, q.next_election_time]).toEqual(election);
  });

  it('keeps later incumbents, government parties and agreements while changing a former officeholder’s party leadership', () => {
    const { adapter, q } = setup({ vicepresident: 'Oriol Junqueras', vicepresident_party: 'erc' });
    hold(adapter, q);
    Object.assign(q, { president: 'Carles Puigdemont', ciu_leader: 'Carles Puigdemont', vicepresident: 'Pere Aragonès', education_minister: 'Meritxell Ruiz', cat_coalition: ['ciu', 'erc'], cat_coalition_support: ['cup'] });
    const coalition = structuredClone(q.cat_coalition);
    const support = structuredClone(q.cat_coalition_support);
    convict(adapter, q);
    expect(q.president).toBe('Carles Puigdemont');
    expect(q.ciu_leader).toBe('Carles Puigdemont');
    expect(q.vicepresident).toBe('Pere Aragonès');
    expect(q.erc_leader).toBe('Marta Rovira');
    expect(q.cat_coalition).toEqual(coalition);
    expect(q.cat_coalition_support).toEqual(support);
  });

  it('prevents convicted ministers returning through future hard-coded cabinet nominations', () => {
    const { adapter, q } = setup();
    hold(adapter, q);
    convict(adapter, q);
    Object.assign(q, { parlament_coalition_proposal_president: 'ciu', parlament_coalition_proposal_vp: 'ciu', parlament_coalition_proposal_education: 'ciu' });
    (adapter.engine as any).state.jumpSceneId = 'main';
    enter(adapter, 'parlament_coalition_setnames');
    const banned = q.consultation_verdict.defendants.map((p: Q) => p.name);
    for (const office of ['president', 'vp', 'education']) {
      expect(banned).not.toContain(q[`parlament_coalition_proposal_${office}_name`]);
    }
  });

  it('deduplicates defendants and never appoints a successor also convicted in the case', () => {
    const { adapter, q } = setup({ vicepresident: 'Oriol Junqueras', vicepresident_party: 'erc', education_minister: 'Marta Rovira', education_minister_party: 'erc' });
    hold(adapter, q);
    convict(adapter, q);
    expect(q.erc_leader).toBe('Pere Aragonès');
    expect(q.vicepresident).not.toBe('Marta Rovira');
    const { adapter: second, q: shared } = setup({ vicepresident: 'Artur Mas' });
    hold(second, shared);
    convict(second, shared);
    expect(shared.consultation_verdict.defendants).toHaveLength(2);
  });

  it.each(['Neus Munté', 'Francesc Homs'])('can replace a later cabinet when %s and Ruiz are among the defendants', (vicepresident) => {
    const { adapter, q } = setup({ ciu_roadmap: 1, vicepresident, education_minister: 'Meritxell Ruiz' });
    hold(adapter, q);
    convict(adapter, q);
    const banned = q.consultation_verdict.defendants.map((person: Q) => person.name);
    for (const office of ['president', 'vicepresident', 'education_minister']) expect(banned).not.toContain(q[office]);
    expect(banned).not.toContain(q.ciu_leader);
  });

  it('leaves an old save without recorded organisers alone', () => {
    const { adapter, q } = setup({ consultation_happened: true });
    enter(adapter, 'post_event');
    expect(q.consultation_verdict).toBeUndefined();
    expect(q.president).toBe('Artur Mas');
  });
});
