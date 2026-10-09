import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');
let arrivalErrors: string[];

beforeEach(() => {
  arrivalErrors = [];
  vi.spyOn(console, 'info').mockImplementation(() => {});
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation((...args) => {
    if (!String(args[0]).startsWith('Page loaded from unexpected origin')) {
      arrivalErrors.push(args.map(String).join(' '));
    }
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  expect(arrivalErrors).toEqual([]);
});

function january(overrides: Record<string, unknown> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  Object.assign(adapter.qualities, {
    year: 2013, month: 1, week: 1, time: 5, rubicon: false,
    player_party: 'erc', month_actions: 0, countdowns: [],
    ciu_erc_coalition_phase: 1,
    parlament_current_ciu: 'ciu', parlament_current_ciu_s: 50,
    erc_parlament_s: 21, speaker_party: 'ciu', parlament_coalition_push: 0,
    ...overrides,
  });
  return adapter;
}

function choices(adapter: DendryAdapter): string[] {
  return adapter.engine.choiceCache.map((choice: { id: string }) => choice.id);
}

function pick(adapter: DendryAdapter, id: string) {
  const index = choices(adapter).indexOf(id);
  expect(index, `missing ${id}`).toBeGreaterThanOrEqual(0);
  expect(adapter.engine.choiceCache[index].canChoose, id).toBe(true);
  adapter.choose(index);
}

function rejectCiuOffer(adapter: DendryAdapter) {
  pick(adapter, 'parlament_coalition_ciu_erc_phase');
  pick(adapter, 'parlament_coalition_ciu_erc_phase.policy_concessions');
  pick(adapter, 'parlament_coalition_ciu_erc_phase.push_leverage_concessions');
  pick(adapter, 'parlament_coalition_ciu_erc_phase.indy_concessions');
  pick(adapter, 'parlament_coalition_ciu_erc_phase.phase_text');
  pick(adapter, 'parlament_coalition_ciu_erc_phase.send_back');
  pick(adapter, 'post_event');
}

describe('coalition time advances after the current newspaper editions', () => {
  it.each([false, true])('preserves Barcenas and political paralysis, story first: %s', (storyFirst) => {
    const adapter = january();
    const visited = vi.spyOn(adapter.engine, 'goToScene');
    adapter.goToScene('post_event');
    expect(choices(adapter)).toEqual(['barcenas_leak', 'parlament_coalition_ciu_erc_phase']);

    if (storyFirst) {
      pick(adapter, 'barcenas_leak');
      pick(adapter, 'post_event');
    }
    rejectCiuOffer(adapter);

    if (!storyFirst) {
      expect(adapter.qualities.month).toBe(1);
      expect(adapter.qualities.month_actions).toBe(1);
      expect(adapter.qualities.coalition_phase_timer).toBe(1);
      expect(choices(adapter)).toEqual(['barcenas_leak']);
      // Rechecking the edition must neither consume the advance nor age timers.
      adapter.goToScene('post_event');
      expect(adapter.qualities.time).toBe(5);
      expect(adapter.qualities.coalition_phase_timer).toBe(1);
      pick(adapter, 'barcenas_leak');
      expect(adapter.qualities.corruption_pp).toBe(35);
      pick(adapter, 'post_event');
    }

    expect(adapter.qualities.month).toBe(2);
    expect(adapter.qualities.time).toBe(6);
    expect(adapter.qualities.month_actions).toBe(0);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
    expect(adapter.qualities.ciu_erc_coalition_phase).toBe(2);
    expect(adapter.qualities.corruption_pp).toBeCloseTo(34.93);
    expect(choices(adapter)).toEqual(['parlament_coalition_ciu_erc_phase']);
    expect(visited.mock.calls.some(([id]) => id === 'main' || id.startsWith('main.'))).toBe(false);
  });

  it('also waits for a pending election before consuming the month advance', () => {
    const adapter = january({
      next_local_election_year: 2013, next_local_election_month: 1, next_local_election_week: 1,
    });
    const visited = vi.spyOn(adapter.engine, 'goToScene');
    adapter.goToScene('parlament_coalition_ciu_erc_phase.new_phase_2');
    pick(adapter, 'post_event');

    expect(adapter.qualities.month).toBe(1);
    expect(adapter.qualities.coalition_phase_timer).toBe(1);
    expect(adapter.engine.state.sceneId).toBe('post_event.elections_choice');
    expect(choices(adapter)).toContain('local_elections');
    pick(adapter, 'local_elections');
    expect(adapter.qualities.next_local_election_year).toBe(2017);
    adapter.goToScene('post_event');
    expect(adapter.qualities.month).toBe(1);
    expect(adapter.qualities.month_actions).toBe(1);
    expect(choices(adapter)).toEqual(['barcenas_leak']);
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');

    expect(adapter.qualities.month).toBe(2);
    expect(adapter.qualities.time).toBe(6);
    expect(choices(adapter)).toEqual(['parlament_coalition_ciu_erc_phase']);
    expect(visited.mock.calls.some(([id]) => id === 'main' || id.startsWith('main.'))).toBe(false);
  });

  it('persists a deferred advance and timer across save/load', () => {
    const adapter = january();
    adapter.goToScene('post_event');
    rejectCiuOffer(adapter);
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());

    expect(restored.qualities.month).toBe(1);
    expect(restored.qualities.month_actions).toBe(1);
    expect(restored.qualities.coalition_phase_timer).toBe(1);
    expect(choices(restored)).toEqual(['barcenas_leak']);
    pick(restored, 'barcenas_leak');
    pick(restored, 'post_event');
    expect(restored.qualities.month).toBe(2);
    expect(restored.qualities.coalition_phase_timer).toBe(0);
    expect(choices(restored)).toEqual(['parlament_coalition_ciu_erc_phase']);
  });

  it('registers the shared timer in an older save without delaying an available round', () => {
    const adapter = january();
    delete adapter.qualities.coalition_phase_timer;
    adapter.qualities.timers = (adapter.qualities.timers as string[]).filter((id) => id !== 'coalition_phase');
    adapter.goToScene('post_event');
    expect(choices(adapter)).toContain('parlament_coalition_ciu_erc_phase');
    rejectCiuOffer(adapter);
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');
    expect(adapter.qualities.month).toBe(2);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
    expect((adapter.qualities.timers as string[]).filter((id) => id === 'coalition_phase')).toHaveLength(1);
  });

  it('keeps the opening Speaker delay before the first round', () => {
    const adapter = january({ ciu_erc_coalition_phase: 0 });
    adapter.goToScene('parlament_coalition_ciu_erc.erc');
    pick(adapter, 'parlament_coalition_ciu_erc.ciu_speaker');
    pick(adapter, 'post_event');
    expect(adapter.qualities.month).toBe(1);
    expect(adapter.qualities.coalition_phase_timer).toBe(1);
    expect(choices(adapter)).toEqual(['barcenas_leak']);
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');
    expect(adapter.qualities.month).toBe(2);
    expect(adapter.qualities.ciu_erc_coalition_phase).toBe(1);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
    expect(choices(adapter)).toEqual(['parlament_coalition_ciu_erc_phase']);
  });

  it('keeps successive rejected rounds a month apart and releases the desk after failure', () => {
    const adapter = january();
    const visited = vi.spyOn(adapter.engine, 'goToScene');
    adapter.goToScene('post_event');
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');
    rejectCiuOffer(adapter);
    rejectCiuOffer(adapter);
    expect(adapter.qualities.month).toBe(3);
    expect(adapter.qualities.time).toBe(7);
    expect(adapter.qualities.ciu_erc_coalition_phase).toBe(3);
    expect(visited.mock.calls.some(([id]) => id === 'main' || id.startsWith('main.'))).toBe(false);

    adapter.goToScene('parlament_coalition_ciu_erc_phase.next_phase');
    expect(adapter.qualities.ciu_erc_coalition_phase).toBe(0);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
    expect(adapter.qualities.parlament_repeat_election).toBe(true);
    pick(adapter, 'post_event');
    expect(adapter.engine.state.sceneId).toBe('main');
  });

  it('clears the timer on agreement and preserves unread news before an opening advance', () => {
    const adapter = january({ month_actions: 1, ciu_parlament_s: 50 });
    adapter.goToScene('post_event');
    pick(adapter, 'parlament_coalition_ciu_erc_phase');
    // Supply a cabinet-held Health portfolio for the accepted-package fixture.
    adapter.qualities.parlament_coalition_proposal_health = 'erc';
    pick(adapter, 'parlament_coalition_ciu_erc_phase.accept');
    expect(adapter.qualities.ciu_erc_coalition_phase).toBe(0);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
    pick(adapter, 'parlament_coalition_resolve');
    expect(adapter.qualities.parlament_coalition_resolution_valid).toBe(true);
    expect(adapter.qualities.month).toBe(1);
    expect(choices(adapter)).toEqual(['barcenas_leak']);
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');
    expect(adapter.qualities.month).toBe(2);
    expect(adapter.qualities.time).toBe(6);
    // The ordinary monthly update still dispatches newly triggered stories.
    expect(choices(adapter)).toEqual(['burdens_start']);
    pick(adapter, 'burdens_start');
    pick(adapter, 'burdens_start.burdens_continue');
    expect(adapter.engine.state.sceneId).toBe('main');
  });

  it('still advances an ordinary empty month and ignores the fallback Continue choice', () => {
    const adapter = january({ ciu_erc_coalition_phase: 0, month: 2, month_actions: 1 });
    adapter.goToScene('post_event');
    expect(adapter.qualities.month).toBe(3);
    expect(adapter.qualities.time).toBe(6);
    expect(adapter.qualities.month_actions).toBe(0);
    expect(adapter.engine.state.sceneId).toBe('main');
  });
});

function finishOpeningDelay(adapter: DendryAdapter, phase: string, roundKey: string, speaker: string) {
  expect(adapter.qualities.month).toBe(1);
  expect(adapter.qualities.month_actions).toBe(1);
  expect(adapter.qualities.coalition_phase_timer).toBe(1);
  expect(choices(adapter)).toEqual(['barcenas_leak']);
  pick(adapter, 'barcenas_leak');
  pick(adapter, 'post_event');
  expect(adapter.qualities.month).toBe(2);
  expect(adapter.qualities.time).toBe(6);
  expect(adapter.qualities.month_actions).toBe(0);
  expect(adapter.qualities.coalition_phase_timer).toBe(0);
  expect(adapter.qualities[roundKey]).toBe(1);
  expect(choices(adapter)).toEqual([phase]);
  pick(adapter, phase);
  expect(adapter.engine.state.sceneId).toBe(`${phase}.choices_menu`);
  expect(adapter.qualities.speaker_party).toBe(speaker);
}

describe('ERC-CUP and broad coalition Speaker routes preserve opening news', () => {
  it.each([
    ['erc', 'erc', 'erc_leader_accept_cup_speaker', 'cup'],
    ['erc', 'erc', 'erc_leader_retain_speaker', 'erc'],
    ['erc', 'cup', 'post_event', 'erc'],
    ['cup', 'erc', 'post_event', 'cup'],
    ['cup', 'cup', 'cup_leader_accept_erc_speaker', 'erc'],
    ['cup', 'cup', 'cup_leader_retain_speaker', 'cup'],
  ])('defers the %s-player/%s-led opening after %s', (player, leader, option, speaker) => {
    const adapter = january({
      ciu_erc_coalition_phase: 0, player_party: player,
      erc_parlament_s: leader === 'erc' ? 40 : 32,
      cup_parlament_s: leader === 'cup' ? 40 : 32,
      parlament_coalition_erc_cup_s: 72,
      parlament_erc_cup_government_available: true,
    });
    const visited = vi.spyOn(adapter.engine, 'goToScene');
    adapter.goToScene(`parlament_coalition_erc_cup.${player}_player_${leader}_led`);
    pick(adapter, option === 'post_event' ? option : `parlament_coalition_erc_cup.${option}`);
    finishOpeningDelay(adapter, `parlament_coalition_erc_cup_phase_${player}_player_${leader}_led`,
      'erc_cup_coalition_phase', speaker);
    expect(adapter.qualities.parlament_coalition_leader).toBe(leader);
    expect(visited.mock.calls.some(([id]) => id === 'main' || id.startsWith('main.'))).toBe(false);
  });

  const broadOpenings = ['republican', 'popular_unity', 'new_popular_front', 'minority_left_tripartite']
    .flatMap((kind) => ['erc', 'cup'].flatMap((player) =>
      ['accept_nominee', 'player_speaker'].map((option) => [kind, player, option])));

  it.each(broadOpenings)('defers %s for %s after %s', (kind, player, option) => {
    const adapter = january({
      ciu_erc_coalition_phase: 0, player_party: player, cup_broad_kind: kind,
      erc_parlament_s: 40, cup_parlament_s: 20, psc_parlament_s: 30,
      parlament_current_icv: 'icv', parlament_current_icv_s: 20, icv_parlament_s: 20,
    });
    const visited = vi.spyOn(adapter.engine, 'goToScene');
    adapter.goToScene('parlament_coalition_cup_broad.setup');
    const speaker = option === 'player_speaker' ? player : String(adapter.qualities.speaker_party);
    pick(adapter, `parlament_coalition_cup_broad.${option}`);
    finishOpeningDelay(adapter, 'parlament_coalition_cup_broad_phase', 'cup_broad_coalition_phase', speaker);
    expect(visited.mock.calls.some(([id]) => id === 'main' || id.startsWith('main.'))).toBe(false);
  });
});

describe('sovereignty interludes use the ordinary event queue', () => {
  it('keeps Banos departure once per playthrough after reload and all twelve new openings', () => {
    const event = 'parlament_coalition_sovereignty_banos';
    const adapter = january({
      ciu_erc_coalition_phase: 0, coalition_phase_timer: 1,
      parlament_coalition_sovereignty_phase: 2,
      parlament_coalition_sovereignty_banos_departure: true,
      cup_leader: 'Antonio Baños', cup_poblelliure_dissent: 10,
    });
    adapter.goToScene('post_event');
    pick(adapter, event);
    expect(adapter.qualities.cup_leader).toBe('Anna Gabriel');
    expect(adapter.qualities.cup_poblelliure_dissent).toBe(15);

    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(adapter.exportStateJSON());
    expect(restored.qualities.cup_poblelliure_dissent).toBe(15);
    pick(restored, event + '.erc_space');
    expect(choices(restored)).not.toContain(event);

    const openings = Object.keys(restored.engine.game.scenes).filter(id =>
      /^parlament_coalition_(joint_cup|ciu_erc_cup)\.\w+_player_\w+$/.test(id));
    expect(openings).toHaveLength(12);
    for (const opening of openings) {
      restored.qualities.player_party = opening.includes('.cup_player_') ? 'cup' : 'erc';
      restored.goToScene(opening);
      expect(restored.qualities.parlament_coalition_sovereignty_banos_departure).toBe(false);
      Object.assign(restored.qualities, {
        parlament_coalition_sovereignty_phase: 2, coalition_phase_timer: 1,
        parlament_coalition_sovereignty_banos_departure: true,
      });
      restored.goToScene('post_event');
      expect(choices(restored), opening).not.toContain(event);
      expect(restored.qualities.cup_poblelliure_dissent).toBe(15);
    }
  });

  it('drains a newly available Banos event and the dated news before opening the next phase', () => {
    const adapter = january({
      ciu_erc_coalition_phase: 0,
      parlament_coalition_sovereignty_phase: 2,
      parlament_coalition_sovereignty_kind: 'jxsi',
      parlament_coalition_leader: 'jxsi',
      parlament_coalition_sovereignty_cabinet: ['jxsi'],
      parlament_coalition_sovereignty_dispute_round: 2,
      parlament_coalition_sovereignty_mediation_round: 2,
      parlament_coalition_sovereignty_banos_departure: true,
      parlament_coalition_sovereignty_mas_pressure: 0,
      parlament_coalition_proposal_president_name: 'Artur Mas',
      coalition_phase_timer: 1, month_actions: 1,
    });
    adapter.goToScene('post_event');
    expect(adapter.qualities.month).toBe(1);
    expect(choices(adapter)).toContain('parlament_coalition_sovereignty_banos');
    expect(choices(adapter)).not.toContain('parlament_coalition_joint_cup_phase_erc_player_jxsi_led');
    pick(adapter, 'parlament_coalition_sovereignty_banos');
    pick(adapter, 'parlament_coalition_sovereignty_banos.erc_space');
    expect(adapter.qualities.month).toBe(1);
    expect(adapter.qualities.month_actions).toBe(1);
    expect(choices(adapter)).toEqual(['barcenas_leak']);
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');
    expect(adapter.qualities.month).toBe(2);
    expect(adapter.qualities.month_actions).toBe(0);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
    expect(choices(adapter)).toContain('parlament_coalition_joint_cup_phase_erc_player_jxsi_led');
  });

  it('mediation has its own round guard and can recur in a later round', () => {
    const adapter = january({
      ciu_erc_coalition_phase: 0,
      parlament_coalition_sovereignty_phase: 2, coalition_phase_timer: 1,
      parlament_coalition_sovereignty_dispute_round: 0,
      parlament_coalition_sovereignty_mediation_done: false,
      parlament_coalition_sovereignty_feud: 2,
      parlament_current_ciu: 'ciu',
      parlament_coalition_proposal_president_name: 'Carles Puigdemont',
    });
    const scene = adapter.engine.game.scenes.parlament_coalition_sovereignty_mediation;
    expect(adapter.engine._runPredicate(scene.viewIf, false)).toBe(true);
    adapter.qualities.parlament_coalition_sovereignty_dispute_round = 2;
    expect(adapter.engine._runPredicate(scene.viewIf, false)).toBe(true);
    adapter.qualities.parlament_coalition_sovereignty_mediation_round = 2;
    expect(adapter.engine._runPredicate(scene.viewIf, false)).toBe(false);
    adapter.qualities.parlament_coalition_sovereignty_phase = 3;
    expect(adapter.engine._runPredicate(scene.viewIf, false)).toBe(true);
  });
});

const rounds = [
  ['ciu_erc', 'ciu_erc_coalition_phase', 'new_phase_2', 'erc', 'ciu'],
  ['ciu_minority', 'ciu_min_coalition_phase', 'new_phase_2', 'erc', 'ciu'],
  ['erc_ciu', 'erc_ciu_coalition_phase', 'new_phase_2', 'erc', 'erc'],
  ['alternative_tripartite', 'alternative_tripartite_coalition_phase', 'new_phase_2', 'erc', 'icv'],
  ['left_tripartite', 'left_tripartite_coalition_phase', 'new_phase_2', 'erc', 'erc'],
  ['psc_tripartite', 'psc_tripartite_coalition_phase', 'new_phase_2', 'erc', 'psc'],
  ['cup_broad', 'cup_broad_coalition_phase', 'wait', 'erc', 'erc'],
  ['erc_cup_phase_erc_player_erc_led', 'erc_cup_coalition_phase', 'wait', 'erc', 'erc'],
  ['erc_cup_phase_erc_player_cup_led', 'erc_cup_coalition_phase', 'wait', 'erc', 'cup'],
  ['erc_cup_phase_cup_player_erc_led', 'erc_cup_coalition_phase', 'wait', 'cup', 'erc'],
  ['erc_cup_phase_cup_player_cup_led', 'erc_cup_coalition_phase', 'wait', 'cup', 'cup'],
] as const;

describe('all coalition rounds share one timer', () => {
  it.each(rounds)('defers %s until its current stories are resolved', (family, roundKey, wait, player, leader) => {
    const phase = `parlament_coalition_${family.includes('_phase_') ? family : `${family}_phase`}`;
    const adapter = january({
      ciu_erc_coalition_phase: 0, [roundKey]: 1,
      player_party: player, parlament_coalition_leader: leader,
    });
    adapter.goToScene(`${phase}.${wait}`);
    pick(adapter, 'post_event');
    expect(adapter.qualities[roundKey]).toBe(2);
    expect(adapter.qualities.month).toBe(1);
    expect(adapter.qualities.coalition_phase_timer).toBe(1);
    expect(choices(adapter)).toEqual(['barcenas_leak']);
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');
    expect(adapter.qualities.month).toBe(2);
    expect(adapter.qualities.time).toBe(6);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
    expect(choices(adapter)).toEqual([phase]);

    // A failed final round must release this shared timer for future formations.
    adapter.qualities[roundKey] = 3;
    adapter.qualities.coalition_phase_timer = 1;
    adapter.goToScene(`${phase}.new_elections`);
    expect(adapter.qualities[roundKey]).toBe(0);
    expect(adapter.qualities.coalition_phase_timer).toBe(0);
  });
});
