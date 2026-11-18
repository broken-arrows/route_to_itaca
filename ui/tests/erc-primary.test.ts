import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
type Q = Record<string, any>;
const challengers = ['marta_rovira', 'pere_aragones', 'raul_romeva', 'carme_forcadell'];
const factions = ['left', 'core', 'pragmatic', 'civic'];
const primaryScene = 'junqueras_dethroned';

beforeEach(() => vi.spyOn(console, 'error').mockImplementation(() => {}));
afterEach(() => {
  expect(console.error).not.toHaveBeenCalled();
  vi.restoreAllMocks();
});

function setup(overrides: Q = {}, weights = [20, 40, 20, 30]) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([11, 22, 33, 44, 55]);
  adapter.goToScene('root.start');
  adapter.goToScene('root.esquerra_2');
  const q = adapter.qualities as Q;
  Object.assign(q, {
    year: 2019, month: 2, week: 1, difficulty: 1,
    player_party: 'erc', erc_leader: 'Oriol Junqueras', erc_leader_gender: 'm',
    erc_roadmap: 2, player_roadmap: 2, independence_trust: 50,
    erc_parlament_s: 21, party_resources: 3, max_advisors: 3,
    oriol_junqueras_advisor: 0, n_advisors: 2,
    erc_in_jxsi: false, erc_in_jxcat: false,
    month_actions: 0, month_actions_last: 0, government_does_things_timer: 100,
  });
  for (const id of challengers) q[`${id}_advisor_available`] = 1;
  for (const [i, faction] of factions.entries()) {
    q[`erc_${faction}_strength`] = weights[i];
    q[`erc_${faction}_dissent`] = 10;
  }
  Object.assign(q, overrides);
  (adapter.engine as any).game.tagLookup.event = { [primaryScene]: true };
  (adapter.engine as any).game.tagLookup.election = {};
  return { adapter, q };
}

function choose(adapter: DendryAdapter, id: string) {
  const index = adapter.currentFrame().choices.findIndex(c => c.id === id && c.canChoose);
  expect(index, `${id} unavailable at ${adapter.currentFrame().sceneId}`).toBeGreaterThanOrEqual(0);
  return adapter.choose(index);
}

function contest(adapter: DendryAdapter, endorsement?: string) {
  adapter.goToScene(primaryScene);
  choose(adapter, `${primaryScene}.start_push_out`);
  return vote(adapter, endorsement);
}

function vote(adapter: DendryAdapter, endorsement?: string) {
  const frame = choose(adapter, `${primaryScene}.${endorsement ? 'back_' + endorsement : 'cast_ballot'}`);
  return frame.sceneId.startsWith(`${primaryScene}.back_`)
    ? choose(adapter, `${primaryScene}.cast_ballot`)
    : frame;
}

function finish(adapter: DendryAdapter, q: Q) {
  if (q.erc_primary.needsRunoff && !q.erc_primary.resolved) vote(adapter);
  expect(adapter.currentFrame().sceneId).toBe(`${primaryScene}.result`);
}

function percentages(q: Q, key: string) {
  return gameLib.sceneResultBars(q, q[key].rows).map(row => row.share);
}

describe('ERC extraordinary leadership primary', () => {
  it('uses recruitability rather than appointment and excludes Aragonès until 2019', () => {
    for (const year of [2018, 2019]) {
      const { adapter, q } = setup({ year, marta_rovira_advisor: 0, n_advisors: 1,
        raul_romeva_advisor_available: 0, carme_forcadell_advisor: 0 });
      adapter.goToScene(primaryScene);
      const ids = q.erc_primary.candidates.map((c: Q) => c.id);
      expect(ids).toContain('marta_rovira');
      expect(ids).toContain('carme_forcadell');
      expect(ids).not.toContain('raul_romeva');
      expect(ids.includes('pere_aragones')).toBe(year === 2019);
    }
  });

  it('does not restore departed candidates even with a stale availability flag', () => {
    const { adapter, q } = setup({ raul_romeva_departed: true, carme_forcadell_departed: true });
    adapter.goToScene(primaryScene);
    expect(q.erc_primary.candidates.map((c: Q) => c.id)).toEqual(['oriol_junqueras', 'marta_rovira', 'pere_aragones']);
  });

  it('normally reelects Junqueras outright and preserves his advisory place permanently', () => {
    const { adapter, q } = setup();
    const frame = contest(adapter);
    expect(frame.html).toContain('result-bars');
    expect(q.erc_primary.needsRunoff).toBe(false);
    expect(q.erc_primary.winner).toBe('oriol_junqueras');
    expect(q.erc_primary.winningShare).toBeCloseTo(60);
    expect(q.party_resources).toBe(2);
    expect(frame.sceneId).toBe(`${primaryScene}.result`);
    expect(frame.choices.filter(c => c.canChoose).map(c => c.id)).toEqual(['post_event']);
    finish(adapter, q);
    expect(q.junqueras_reelected).toBe(true);
    expect(q.oriol_junqueras_advisor).toBe(1);
    expect(q.n_advisors).toBe(3);
    expect(q.erc_core_dissent).toBeLessThan(10);
    adapter.goToScene('erc_shuffle_leadership.rm_main');
    expect(adapter.currentFrame().choices.find(c => c.id.endsWith('.remove_oriol_junqueras'))?.canChoose).toBe(false);
  });

  it('can return Junqueras in a runoff despite losing his first-round majority', () => {
    const { adapter, q } = setup({ independence_trust: 20 });
    contest(adapter);
    expect(q.erc_primary.first.find((r: Q) => r.id === 'oriol_junqueras').share).toBeLessThan(50);
    expect(q.erc_primary.needsRunoff).toBe(true);
    expect(adapter.currentFrame().sceneId).toBe(`${primaryScene}.runoff`);
    vote(adapter);
    expect(q.erc_primary.winner).toBe('oriol_junqueras');
    expect(adapter.currentFrame().html).toContain('erc_primary_second_round');
    expect(adapter.currentFrame().sceneId).toBe(`${primaryScene}.result`);
    expect(q.junqueras_reelected).toBe(true);
  });

  it('presents only eligible candidates and keeps voting available after spending the last resource', () => {
    const { adapter, q } = setup({ year: 2018, party_resources: 1, raul_romeva_advisor_available: 0 });
    adapter.goToScene(primaryScene);
    const briefing = choose(adapter, `${primaryScene}.start_push_out`);
    const candidates = briefing.choices.filter(c => c.id.includes('.back_'));
    expect(candidates.map(c => c.id.split('.back_')[1]).sort()).toEqual(['carme_forcadell', 'marta_rovira', 'oriol_junqueras']);
    expect(candidates.every(c => !c.canChoose)).toBe(true);
    expect(briefing.html).not.toContain('result-bars');
    vote(adapter);
    expect(q.party_resources).toBe(0);
  });

  it('can push a challenger into the runoff and charges once per ballot', () => {
    const baseline = setup({ independence_trust: 10 });
    contest(baseline.adapter);
    expect(baseline.q.erc_primary.first[1].id).toBe('marta_rovira');

    const { adapter, q } = setup({ independence_trust: 10 });
    contest(adapter, 'raul_romeva');
    expect(q.erc_primary.first[1].id).toBe('raul_romeva');
    expect(q.party_resources).toBe(1);
    const first = JSON.parse(JSON.stringify(q.erc_primary.first));
    const briefing = adapter.currentFrame();
    expect(briefing.sceneId).toBe(`${primaryScene}.runoff`);
    const finalists = briefing.choices.filter(c => c.id.includes('.back_')).map(c => c.id.split('.back_')[1]).sort();
    expect(finalists).toEqual(['oriol_junqueras', 'raul_romeva']);
    expect(briefing.html).toContain('erc_primary_first_round');
    expect(briefing.html).not.toContain('erc_primary_second_round');
    const before = q.erc_primary.runoff.find((r: Q) => r.id === 'raul_romeva').share;
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());
    const loaded = restored.qualities as Q;
    vote(restored, 'raul_romeva');
    expect(loaded.erc_primary.runoff.find((r: Q) => r.id === 'raul_romeva').share).toBeGreaterThan(before);
    expect(loaded.erc_primary.first).toEqual(first);
    expect(loaded.party_resources).toBe(0);
    restored.goToScene(`${primaryScene}.back_raul_romeva`);
    expect(loaded.party_resources).toBe(0);
    expect(percentages(loaded, 'erc_primary_second_round').reduce((sum, n) => sum + n, 0)).toBeCloseTo(100, 10);
  });

  it('gives a backed underdog more votes while the clear favourite can still win', () => {
    const baseline = setup();
    contest(baseline.adapter);
    const { adapter, q } = setup();
    contest(adapter, 'marta_rovira');
    expect(q.erc_primary_first_marta_rovira).toBeGreaterThan(baseline.q.erc_primary_first_marta_rovira);
    expect(q.erc_primary.winner).toBe('oriol_junqueras');
    expect(q.party_resources).toBe(1);
  });

  it('lets a save from the previous flow continue past a ballot already shown', () => {
    const { adapter } = setup({ independence_trust: 10 });
    contest(adapter);
    const first = JSON.parse(adapter.exportStateJSON());
    // Before the campaign choices, the briefing scene displayed first-ballot results.
    first.sceneId = `${primaryScene}.start_push_out`;
    first.qualities.erc_primary_campaign_runoff = false;
    const restored = DendryAdapter.fromJSONText(gameText);
    const frame = restored.importStateJSON(JSON.stringify(first));
    expect(frame.choices.map(c => c.id)).toEqual([`${primaryScene}.cast_ballot`]);
    expect(vote(restored).sceneId).toBe(`${primaryScene}.runoff`);

    vote(adapter);
    const runoff = JSON.parse(adapter.exportStateJSON());
    runoff.sceneId = `${primaryScene}.runoff`;
    delete runoff.qualities.erc_primary_campaign_runoff;
    expect(restored.importStateJSON(JSON.stringify(runoff)).choices.map(c => c.id))
      .toEqual([`${primaryScene}.cast_ballot`]);
    expect(vote(restored).sceneId).toBe(`${primaryScene}.result`);
    expect(restored.qualities.junqueras_reelected).toBe(true);
    expect(restored.qualities.party_resources).toBe(2);
  });

  it.each([
    ['marta_rovira', [65, 15, 10, 10], 2, 'Marta Rovira', 'f'],
    ['pere_aragones', [10, 10, 75, 5], 1, 'Pere Aragonès', 'm'],
    ['raul_romeva', [25, 15, 15, 45], 2, 'Raül Romeva', 'm'],
    ['carme_forcadell', [10, 10, 5, 75], 3, 'Carme Forcadell', 'f'],
  ])('%s can win through their coalition without already being appointed', (id, weights, roadmap, name, gender) => {
    const { adapter, q } = setup({ independence_trust: 10, erc_parlament_s: 10,
      erc_roadmap: roadmap, [`${id}_advisor`]: 0 }, weights as number[]);
    const offices = ['president', 'vicepresident', 'speaker', 'economy_minister'].map(key => q[key]);
    contest(adapter);
    finish(adapter, q);
    expect(q.erc_primary.winner).toBe(id);
    expect(q.erc_leader).toBe(name);
    expect(q.erc_leader_gender).toBe(gender);
    expect(q[`${id}_advisor`]).toBe(1);
    expect(q.oriol_junqueras_advisor_available).toBe(0);
    expect(q.n_advisors).toBeLessThanOrEqual(q.max_advisors);
    expect(['president', 'vicepresident', 'speaker', 'economy_minister'].map(key => q[key])).toEqual(offices);
    expect(q.erc_roadmap).toBe(roadmap);
    expect(factions.every(f => Number.isFinite(q[`erc_${f}_dissent`]))).toBe(true);
  });

  it('has finite, conserving percentages with no faction strength', () => {
    const { adapter, q } = setup({}, [0, 0, 0, 0]);
    contest(adapter);
    expect(percentages(q, 'erc_primary_first_round').reduce((sum, n) => sum + n, 0)).toBeCloseTo(100, 10);
    expect(q.erc_primary.first.every((row: Q) => Number.isFinite(row.share))).toBe(true);
    expect(q.erc_primary.winner).toBe('oriol_junqueras');
  });

  it.each(['jxsi', 'jxcat'])('attributes %s representatives and only demobilizes ERC’s share', list => {
    const { adapter, q } = setup({ [`erc_in_${list}`]: true, [`cup_in_${list}`]: true,
      [`${list}_formation_num`]: 60, [`${list}_cup_formation_num`]: 20,
      [`${list}_parlament_s`]: 75, erc_parlament_s: 0 });
    const cell = `${list}_parlament_barcelona_middle_support`;
    q[cell] = 20;
    const abstain = q.abstain_parlament_barcelona_middle_support;
    adapter.goToScene(primaryScene);
    expect(q.erc_primary.seats).toBeCloseTo(36);
    expect(q.erc_primary.electoralEffect).toBe(8);
    expect(q[cell]).toBeCloseTo(20 - 20 * 0.01 * 0.48);
    expect(q.abstain_parlament_barcelona_middle_support - abstain).toBeCloseTo(20 - q[cell]);
    const after = q[cell];
    adapter.goToScene(primaryScene);
    expect(q[cell]).toBe(after);

    const parties = [...q.parties, 'abstain'];
    const total = () => parties.reduce((sum: number, p: string) => sum + q[`${p}_parlament_barcelona_middle_support`], 0);
    const beforeResult = total();
    choose(adapter, `${primaryScene}.start_push_out`);
    vote(adapter);
    finish(adapter, q);
    const closeResult = Math.max(0, Math.min(1, (65 - q.erc_primary.winningShare) / 15));
    expect(q[cell]).toBeCloseTo(after * (1 - 0.48 * (0.02 + 0.03 * closeResult)));
    expect(total()).toBeCloseTo(beforeResult);
    const settled = q[cell];
    adapter.goToScene(`${primaryScene}.result`);
    expect(q[cell]).toBe(settled);
  });

  it.each([
    ['oriol_junqueras', 2, true],
    ['oriol_junqueras', 3, false],
    ['pere_aragones', 2, true],
    ['pere_aragones', 3, false],
  ])('%s at roadmap %s resolves the civic departures once', (winner, roadmap, paluzieLeaves) => {
    const aragones = winner === 'pere_aragones';
    const { adapter, q } = setup({
      erc_roadmap: roadmap, elisenda_paluzie_advisor: 1, joan_laporta_advisor: 1, n_advisors: 4, max_advisors: 8,
      ...(aragones ? { independence_trust: 10, erc_parlament_s: 10 } : {}),
    }, aragones ? [10, 10, 75, 5] : undefined);
    contest(adapter);
    finish(adapter, q);
    expect(q.erc_primary.winner).toBe(winner);
    expect(q.joan_laporta_departed).toBe(true);
    expect(q.joan_laporta_advisor_available).toBe(0);
    expect(q.joan_laporta_advisor).toBe(0);
    expect(Boolean(q.elisenda_paluzie_departed)).toBe(paluzieLeaves);
    expect(q.elisenda_paluzie_advisor).toBe(paluzieLeaves ? 0 : 1);
    expect(q.elisenda_paluzie_advisor_available).toBe(paluzieLeaves ? 0 : 1);
    const count = paluzieLeaves ? 3 : 4;
    expect(q.n_advisors).toBe(count);
    adapter.goToScene(`${primaryScene}.result`);
    expect(q.n_advisors).toBe(count);
  });

  it('hands over Forcadell’s Speaker seat while retaining Laporta and protecting her from a later Civic rupture', () => {
    const { adapter, q } = setup({
      year: 2015, speaker: 'Carme Forcadell', speaker_party: 'jxsi',
      independence_trust: 10, erc_parlament_s: 10, erc_roadmap: 3, joan_laporta_advisor: 1, max_advisors: 8,
    }, [10, 10, 5, 75]);
    contest(adapter);
    finish(adapter, q);
    expect(q.erc_leader).toBe('Carme Forcadell');
    expect(q.erc_primary_new_speaker).toBe('Anna Simó');
    expect(q.speaker).toBe('Anna Simó');
    expect(q.speaker_party).toBe('jxsi');
    expect(q.joan_laporta_advisor).toBe(1);
    q.erc_civic_dissent = 55;
    adapter.goToScene('erc_civic_split');
    expect(q.erc_civic_resign).toBe(true);
    expect(q.carme_forcadell_advisor).toBe(1);
    expect(q.carme_forcadell_departed).toBeFalsy();
  });

  it.each([
    [3, 49, true],
    [3, 48, false],
    [2, 49, false],
  ])('resolves Bosch’s departure on roadmap %i with %i opening Pragmatic dissent', (roadmap, dissent, leaves) => {
    const { adapter, q } = setup({
      independence_trust: 10, erc_parlament_s: 10, erc_roadmap: roadmap,
      erc_pragmatic_dissent: dissent, alfred_bosch_advisor: 1, carme_forcadell_advisor: 1, n_advisors: 4, max_advisors: 8,
    }, [10, 10, 5, 75]);
    contest(adapter);
    finish(adapter, q);
    expect(q.erc_leader).toBe('Carme Forcadell');
    expect(Boolean(q.alfred_bosch_departed)).toBe(leaves);
    expect(q.alfred_bosch_advisor).toBe(leaves ? 0 : 1);
    expect(q.alfred_bosch_advisor_available).toBe(leaves ? 0 : 1);
    expect(q.n_advisors).toBe(leaves ? 3 : 4);
  });

  it.each([
    ['Pere Aragonès', 'erc', 'erc', 'economy', 'Josep Huguet'],
    ['Marta Rovira', 'jxsi', 'jxsi', 'vp', 'Pere Aragonès'],
    ['Raül Romeva', 'jxcat', 'jxcat', 'foreign', 'Alfred Bosch'],
  ])('gives %s distinct cabinet nominations on %s', (leader, presidentParty, officeParty, office, replacement) => {
    const { adapter, q } = setup({
      erc_leader: leader, [`${presidentParty}_leader`]: leader,
      erc_in_jxsi: presidentParty === 'jxsi', jxsi_formation_num: 40,
      erc_in_jxcat: presidentParty === 'jxcat', alfred_bosch_advisor_available: 1,
      parlament_coalition_proposal_president: presidentParty,
      parlament_coalition_proposal_vp: presidentParty,
      [`parlament_coalition_proposal_${office}`]: officeParty,
    });
    (adapter.engine as any).state.jumpSceneId = 'main';
    adapter.goToScene('parlament_coalition_setnames');
    expect(q.parlament_coalition_proposal_president_name).toBe(leader);
    expect(q[`parlament_coalition_proposal_${office}_name`]).toBe(replacement);
    if (presidentParty !== 'erc') expect(q[`parlament_coalition_proposal_${office}_affiliation`]).toBe('erc');
  });

  it.each([
    ['parlament_coalition_majority', 2015, 'Anna Simó', 'f'],
    ['parlament_coalition_joint_cup.jxsi_speaker', 2018, 'Roger Torrent', 'm'],
  ])('chooses a JxSí Speaker in %s when Forcadell leads ERC', (scene, year, speaker, gender) => {
    const { adapter, q } = setup({
      year, erc_leader: 'Carme Forcadell', erc_in_jxsi: true,
      jxsi_leader: 'Artur Mas', parlament_coalition_majority_party: 'jxsi',
      parlament_coalition_majority_s: 72,
    });
    adapter.goToScene(scene);
    expect(q.speaker_party).toBe('jxsi');
    expect(q.speaker).toBe(speaker);
    expect(q.speaker_gender).toBe(gender);
  });

  it('makes trust and electoral success independently strengthen the incumbent', () => {
    const supports: number[] = [];
    for (const overrides of [{}, { independence_trust: 80 }, { erc_parlament_s: 35 }, { independence_trust: 80, erc_parlament_s: 35 }]) {
      const { adapter, q } = setup(overrides);
      contest(adapter);
      supports.push(q.erc_primary.first.find((r: Q) => r.id === 'oriol_junqueras').share);
    }
    expect(supports[1]).toBeGreaterThan(supports[0]);
    expect(supports[2]).toBeGreaterThan(supports[0]);
    expect(supports[3]).toBeGreaterThan(Math.max(supports[1], supports[2]));
  });

  it('preserves frozen results, percentages, resource cost and both ballots across save/load', () => {
    const { adapter, q } = setup({ independence_trust: 10, erc_parlament_s: 10 });
    contest(adapter);
    expect(q.erc_primary.needsRunoff).toBe(true);
    const primary = JSON.parse(JSON.stringify(q.erc_primary));
    expect(percentages(q, 'erc_primary_first_round').reduce((sum, n) => sum + n, 0)).toBeCloseTo(100, 10);
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());
    const loaded = restored.qualities as Q;
    loaded.independence_trust = 100;
    loaded.erc_parlament_s = 80;
    expect(restored.currentFrame().sceneId).toBe(`${primaryScene}.runoff`);
    expect(loaded.erc_primary).toEqual(primary);
    vote(restored);
    const runoffShares = percentages(loaded, 'erc_primary_second_round');
    expect(runoffShares.reduce((sum, n) => sum + n, 0)).toBeCloseTo(100, 10);
    expect(runoffShares.every(n => Number.isFinite(n) && n >= 0 && n <= 100)).toBe(true);
    expect(restored.currentFrame().sceneId).toBe(`${primaryScene}.result`);
    expect(loaded.erc_leader).toBe('Marta Rovira');
    expect(loaded.party_resources).toBe(2);
    const dissent = factions.map(f => loaded[`erc_${f}_dissent`]);
    const count = loaded.n_advisors;
    restored.goToScene(`${primaryScene}.result`);
    expect(factions.map(f => loaded[`erc_${f}_dissent`])).toEqual(dissent);
    expect(loaded.n_advisors).toBe(count);
  });

  it('restores Junqueras at full capacity and records the displaced advisor once', () => {
    const { adapter, q } = setup({ sergi_sabria_advisor: 1, n_advisors: 3, party_resources: 0 });
    adapter.goToScene(primaryScene);
    expect(adapter.currentFrame().choices.find(c => c.id.endsWith('.start_push_out'))?.canChoose).toBe(false);
    choose(adapter, `${primaryScene}.wait_no`);
    expect(q.oriol_junqueras_advisor).toBe(1);
    expect(q.sergi_sabria_advisor).toBe(0);
    expect(q.erc_primary_displaced_advisor).toBe('Sergi Sabrià');
    expect(q.n_advisors).toBe(3);
    expect(q.party_resources).toBe(0);
    expect(q.erc_core_dissent).toBe(13);
    adapter.goToScene(`${primaryScene}.wait_no`);
    expect(q.erc_core_dissent).toBe(13);
    choose(adapter, `${primaryScene}.cancel`);
    expect(q.erc_primary).toBeNull();
    expect(q.junqueras_dethroned).toBeFalsy();
    expect(adapter.currentFrame().choices.some(c => c.id === primaryScene)).toBe(false);
  });

  it('keeps the serving presidential candidate after a convicted Junqueras wins internally', () => {
    const { adapter, q } = setup({ erc_leader: 'Marta Rovira', erc_leader_gender: 'f',
      consultation_verdict: { defendants: [{ name: 'Oriol Junqueras' }] } });
    contest(adapter);
    finish(adapter, q);
    expect(q.junqueras_reelected).toBe(true);
    expect(q.erc_leader).toBe('Marta Rovira');
    expect(q.erc_leader_gender).toBe('f');
  });

  it('keeps the year and Rubicon restrictions in the leadership crisis', () => {
    const { adapter, q } = setup({ year: 2012, oriol_junqueras_advisor: 1, n_advisors: 3 });
    adapter.goToScene('erc_shuffle_leadership.rm_main');
    expect(adapter.currentFrame().choices.find(c => c.id.endsWith('.remove_oriol_junqueras'))?.canChoose).toBe(true);
    q.oriol_junqueras_advisor = 0;
    const engine = adapter.engine as any;
    expect(engine._compileChoices(engine.game.scenes['post_event.events_choice']).some((c: Q) => c.id === primaryScene)).toBe(false);
    q.year = 2013;
    q.rubicon = true;
    adapter.goToScene(primaryScene);
    expect(adapter.currentFrame().choices.find(c => c.id.endsWith('.start_push_out'))?.canChoose).toBe(false);
  });

  it('unlocks Aragonès recruitment on ordinary post_event visits in 2019', () => {
    const { adapter, q } = setup({ pere_aragones_advisor_available: 0, oriol_junqueras_advisor: 1 });
    adapter.goToScene('post_event');
    expect(q.pere_aragones_advisor_available).toBe(1);
    q.pere_aragones_advisor_available = 0;
    q.pere_aragones_departed = true;
    adapter.goToScene('post_event');
    expect(q.pere_aragones_advisor_available).toBe(0);
  });

  it('removes a resolved contest from the real pending-event menu', () => {
    const { adapter, q } = setup();
    contest(adapter);
    finish(adapter, q);
    choose(adapter, 'post_event');
    expect(q.junqueras_dethroned).toBe(true);
    expect(q.erc_primary.resolved).toBe(true);
    expect(adapter.currentFrame().choices.some(c => c.id === primaryScene)).toBe(false);
  });

  it('queues the crisis only after Junqueras has been available to serve on the board', () => {
    const { adapter, q } = setup({ oriol_junqueras_advisor_available: 0 });
    const engine = adapter.engine as any;
    const queued = () => engine._compileChoices(engine.game.scenes['post_event.events_choice']).some((c: Q) => c.id === primaryScene);
    expect(queued()).toBe(false);
    q.oriol_junqueras_advisor_available = 1;
    expect(queued()).toBe(true);
  });
});

describe('ERC leadership in other elections', () => {
  it.each(['erc', 'jxsi'])('changes territorial recruitment on the %s Congreso ballot in Catalunya', ballot => {
    const { q } = setup();
    const baseline = JSON.parse(JSON.stringify(q));
    baseline._prev_spa_cat_spa = baseline.cat_spa_relations + 8;
    if (ballot === 'jxsi') {
      baseline.erc_in_jxsi = baseline.jxsi_in_congreso = true;
      baseline.jxsi_congreso_catalunya_support = baseline.erc_congreso_catalunya_support;
      baseline.erc_congreso_catalunya_support = 0;
    }
    function tick(leader: string) {
      const state = structuredClone(baseline);
      state.erc_leader = leader;
      gameLib.engineTick(state);
      return state;
    }
    const control = tick('Oriol Junqueras');
    const pragmatic = tick('Pere Aragonès');
    const civic = tick('Carme Forcadell');
    const key = `${ballot}_congreso_catalunya_support`;
    expect(pragmatic[key]).toBeLessThan(control[key]);
    expect(civic[key]).toBeGreaterThan(control[key]);
    if (ballot === 'jxsi') expect(civic.erc_congreso_catalunya_support).toBe(0);
    for (const key of Object.keys(control).filter(key => key.endsWith('_congreso_madrid_support'))) {
      expect(civic[key]).toBe(control[key]);
    }
  });

  it.each([
    ['Marta Rovira', 'ciu', 2, 0.75, 0],
    ['Pere Aragonès', 'erc', 1.5, 0, 0],
    ['Pere Aragonès', 'ciu', 0, 0, 0],
    ['Raül Romeva', 'ciu', 0, 0.6, 0.8],
    ['Carme Forcadell', 'ciu', 0.5, 2, 0],
  ])('applies %s’s city, Red Belt and interior appeal at the municipal election', (leader, government, redBelt, interior, city) => {
    function election(ercLeader: string) {
      const { adapter, q } = setup({
        erc_leader: ercLeader, cat_coalition: [government], welfare_index: 65, independence_movement: 72,
        independence_trust: 50, erc_redbelt_mod: 5, cup_interior_mod: -30, erc_local_barcelona_support: 14,
      });
      (adapter.engine as any).state.jumpSceneId = 'main';
      adapter.goToScene('election_algorithm.local_elections');
      return q;
    }
    const baseline = election('Oriol Junqueras');
    const updated = election(leader);
    expect(updated.erc_local_redbelt - baseline.erc_local_redbelt).toBeCloseTo(redBelt);
    expect(updated.erc_local_interior - baseline.erc_local_interior).toBeCloseTo(interior);
    if (city) expect(updated.erc_local_barcelona_pv).toBeGreaterThan(baseline.erc_local_barcelona_pv);
    else expect(updated.erc_local_barcelona_pv).toBe(baseline.erc_local_barcelona_pv);
    expect(updated.erc_local_barcelona_support).toBe(14);
  });

  it('adds Romeva’s Barcelona city appeal to the active joint ticket', () => {
    const { adapter, q } = setup({
      erc_in_jxcat: true, jxcat_in_local: true, erc_local_barcelona_support: 0, jxcat_local_barcelona_support: 20,
    });
    (adapter.engine as any).state.jumpSceneId = 'main';
    adapter.goToScene('election_algorithm.local_elections');
    const baseline = q.jxcat_local_barcelona_pv;
    q.erc_leader = 'Raül Romeva';
    adapter.goToScene('election_algorithm.local_elections');
    expect(q.jxcat_local_barcelona_pv).toBeGreaterThan(baseline);
    expect(q.erc_local_barcelona_pv).toBe(0);
  });
});
