import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');

function setup(overrides: Record<string, unknown> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start');
  const q = adapter.qualities;
  for (const key of Object.keys(q)) if (key.endsWith('_congreso_s')) q[key] = 0;
  Object.assign(q, {
    player_party: 'erc', year: 2017, month: 6, time: 58, week: 1,
    month_actions: 0, month_actions_last: 0, rubicon: false,
    next_spanish_election_time: 94, censure_podemos_countdown: 6, censure_podemos_trigger: false,
    congreso_seats: { total: 350 }, congreso_s_majority: 176,
    congreso_parties: ['pp', 'psoe', 'up', 'podemos', 'erc', 'cup', 'pnv', 'pdcat', 'jxcat', 'cs', 'cc'],
    psoe_congreso_s: 100, up_congreso_s: 60, erc_congreso_s: 10, pnv_congreso_s: 6, pp_congreso_s: 174,
    spanish_pm: 'Mariano Rajoy', spanish_pm_party: 'pp', spanish_coalition: ['pp'], spa_caretaker_gov: false,
    psoe_leader: 'Pedro Sánchez', up_leader: 'Pablo Iglesias', up_leader_gender: 'm',
    corruption_pp: 55, cat_spa_relations: 55, party_resources: 10,
    psc_relations: 80, parlament_current_icv: 'icv', icv_relations: 80,
    parlament_current_ciu: 'pdcat', pdcat_roadmap: 2, pdcat_relations: 80,
    erc_in_jxsi: false, erc_in_jxcat: false, jxcat_in_congreso: false,
    countdowns: [], ...overrides,
  });
  adapter.goToScene('post_event');
  return adapter;
}

function available(adapter: DendryAdapter, kind: string) {
  return adapter.qualities.censure_pending === kind
    && (adapter.engine as any)._compileChoices((adapter.engine as any).game.scenes['post_event.events_choice'])
      .some((choice: any) => choice.id === 'spanish_censure');
}
function pick(adapter: DendryAdapter, id: string) {
  const frame = adapter.currentFrame();
  const index = frame.choices.findIndex(c => c.id === id);
  expect(index, id).toBeGreaterThanOrEqual(0);
  expect(frame.choices[index].canChoose, id).toBe(true);
  return adapter.choose(index);
}

describe('Spanish censure motions', () => {
  it('exposes one shared event and refreshes readiness in post_event without consuming attempts', () => {
    const a = setup();
    expect(a.currentFrame().choices.filter(c => c.id === 'spanish_censure')).toHaveLength(1);
    const before = JSON.stringify(a.qualities);
    expect(available(a, 'psoe')).toBe(true);
    expect(available(a, 'podemos')).toBe(false);
    expect(JSON.stringify(a.qualities)).toBe(before);
    expect(a.qualities.censure_attempted).toEqual({});
    a.qualities.psoe_congreso_s = 59;
    a.goToScene('post_event');
    expect(available(a, 'psoe')).toBe(false);
    expect(available(a, 'podemos')).toBe(true);
  });
  it.each([
    [{ psoe_congreso_s: 100, up_congreso_s: 90, psoe_congreso_pv: 20, up_congreso_pv: 30 }, 'psoe'],
    [{ psoe_congreso_s: 90, up_congreso_s: 100, psoe_congreso_pv: 30, up_congreso_pv: 20 }, 'podemos'],
    [{ psoe_congreso_s: 100, up_congreso_s: 100, psoe_congreso_pv: '20.50', up_congreso_pv: '21.50' }, 'podemos'],
    [{ psoe_congreso_s: 100, up_congreso_s: 100, psoe_congreso_pv: '21.50', up_congreso_pv: '20.50' }, 'psoe'],
    [{ psoe_congreso_s: 100, up_congreso_s: 100, psoe_congreso_pv: 25, up_congreso_pv: 25 }, 'psoe'],
  ])('ranks eligible challengers by seats, popular vote, then PSOE: %j', (overrides, expected) => {
    const a = setup(overrides as Record<string, unknown>);
    expect(a.qualities.censure_pending).toBe(expected);
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_current.kind).toBe(expected);
    expect(a.qualities.censure_pending).toBe('');
  });
  it('holds a requested time advance for an existing motion and discovers newly eligible motions after a tick', () => {
    const pending = setup({ month_actions: 1 });
    expect(pending.qualities.time).toBe(58);
    expect(available(pending, 'psoe')).toBe(true);
    const a = setup({ time: 57 });
    // Isolate the censure event from unrelated historical newspaper items.
    (a.engine as any).game.tagLookup.event = { spanish_censure: true };
    (a.engine as any).game.tagLookup.election = {};
    a.qualities.month_actions = 1;
    a.goToScene('post_event');
    expect(a.qualities.time).toBe(58);
    expect(available(a, 'psoe')).toBe(true);
  });
  it('requires one year, 60 seats and an opposing government', () => {
    expect(available(setup({ time: 57 }), 'psoe')).toBe(false);
    expect(available(setup({ psoe_congreso_s: 59 }), 'psoe')).toBe(false);
    expect(available(setup({ spanish_pm_party: 'psoe' }), 'psoe')).toBe(false);
    expect(available(setup({ spanish_coalition: ['pp', 'psoe'] }), 'psoe')).toBe(false);
    expect(available(setup(), 'psoe')).toBeTruthy();
  });

  it('starts the existing countdown on formation and keeps an expired opportunity', () => {
    const a = setup({ psoe_leader: 'Susana Díaz', corruption_pp: 0, cat_spa_relations: 20 });
    a.goToScene('congreso_coalition_right.pp_minority');
    expect(a.qualities.censure_podemos_countdown).toBe(6);
    expect(a.qualities.censure_podemos_trigger).toBe(false);
    expect(a.qualities.countdowns.filter((name: string) => name === 'censure_podemos')).toHaveLength(1);
    expect(available(a, 'podemos')).toBe(false);
    a.qualities.censure_podemos_trigger = true;
    a.qualities.censure_podemos_countdown = -1;
    a.qualities.spanish_pm_party = 'psoe';
    a.goToScene('post_event');
    expect(available(a, 'podemos')).toBe(false);
    a.qualities.spanish_pm_party = 'pp';
    a.goToScene('post_event');
    expect(available(a, 'podemos')).toBe(true);
    a.goToScene('congreso_coalition_right.pp_minority');
    expect(a.qualities.censure_podemos_countdown).toBe(6);
    expect(a.qualities.censure_podemos_trigger).toBe(false);
    expect(a.qualities.countdowns.filter((name: string) => name === 'censure_podemos')).toHaveLength(1);
  });
  it('fails on abstention despite more yes than no, preserves government and old promises, and consumes the attempt', () => {
    const a = setup({ countdowns: ['negotiated_cash'], negotiated_cash_countdown: 7, pp_congreso_s: 160, cc_congreso_s: 14 });
    const election = a.qualities.next_spanish_election_time;
    pick(a, 'spanish_censure');
    pick(a, 'spanish_censure.abstain');
    expect(a.qualities.censure_yes).toBe(166);
    expect(a.qualities.censure_yes).toBeGreaterThan(a.qualities.censure_no);
    expect(a.qualities.censure_won).toBe(false);
    expect(a.qualities.spanish_pm).toBe('Mariano Rajoy');
    expect(a.qualities.negotiated_cash_countdown).toBe(7);
    expect(a.qualities.next_spanish_election_time).toBe(election);
    pick(a, 'post_event');
    expect(available(a, 'psoe')).toBe(false);
    expect(a.qualities.censure_attempted.podemos).toBeUndefined();
    expect(available(a, 'podemos')).toBe(true);
  });

  it('makes decisive votes eligible for ordinary concessions and starts delivery only after success', () => {
    const a = setup({ countdowns: ['negotiated_funding_psoe'], negotiated_funding_psoe_countdown: 5, competence_renfe: true });
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_can_negotiate).toBe(true);
    pick(a, 'spanish_censure.negotiate');
    pick(a, 'congreso_coalition_negotiation.cash');
    expect(a.qualities.countdowns).not.toContain('negotiated_cash');
    pick(a, 'congreso_coalition_negotiation.resolve');
    expect(a.qualities.censure_won).toBe(true);
    expect(a.qualities.censure_yes).toBe(176);
    expect(a.qualities.spanish_pm).toBe('Pedro Sánchez');
    expect(a.qualities.spanish_coalition).toEqual(['psoe']);
    expect(a.qualities.negotiated_cash_countdown).toBe(8);
    expect(a.qualities.countdowns).not.toContain('negotiated_funding_psoe');
    expect(a.qualities.competence_renfe).toBe(true);
    expect(a.qualities.censure_negotiating).toBe(false);
    pick(a, 'post_event');
    expect(a.qualities.censure_pending).toBe('');
  });

  it('does not offer concessions when other parties already have 176 votes', () => {
    const a = setup({ psoe_congreso_s: 110, pp_congreso_s: 164 });
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_can_negotiate).toBe(false);
    pick(a, 'spanish_censure.oppose');
    expect(a.qualities.censure_won).toBe(true);
  });

  it('lets seatless CUP mediate but never cast votes or negotiate concessions', () => {
    const a = setup({ player_party: 'cup', psoe_leader: '???' });
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_player_seats).toBe(0);
    expect(a.qualities.censure_can_negotiate).toBe(false);
    pick(a, 'spanish_censure.mediate');
    const row = a.qualities.censure_current.votes.find((r: any) => r.party === 'psoe');
    expect(row.yes).toBe(60);
    expect(a.qualities.party_resources).toBe(7);
    expect(a.qualities.censure_can_mediate).toBe(false);
    pick(a, 'spanish_censure.observe');
  });

  it('keeps the interim split stable and uses a smaller mediated share outside exceptional circumstances', () => {
    const a = setup({ psoe_leader: '???', psc_relations: 70, icv_relations: 70, psoe_congreso_s: 101 });
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_current.votes.find((r: any) => r.party === 'psoe').yes).toBe(20);
    a.goToScene('spanish_censure.choices');
    expect(a.qualities.censure_current.votes.find((r: any) => r.party === 'psoe').yes).toBe(20);
    pick(a, 'spanish_censure.mediate');
    expect(a.qualities.censure_current.votes.find((r: any) => r.party === 'psoe').yes).toBe(40);
    expect(a.qualities.party_resources).toBe(8);
  });

  it('keeps Díaz opposed regardless of player contacts', () => {
    const a = setup({ psoe_leader: 'Susana Díaz', censure_attempted: { psoe: true } });
    pick(a, 'spanish_censure');
    const row = a.qualities.censure_current.votes.find((r: any) => r.party === 'psoe');
    expect(row.yes).toBe(0);
    expect(row.no).toBe(100);
    expect(a.qualities.censure_can_mediate).toBe(false);
  });

  it('requires exceptional conditions for Madina and carries his referendum restriction into concessions', () => {
    const a = setup({ psoe_leader: 'Eduardo Madina', psoe_congreso_s: 60, up_congreso_s: 100 });
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_current.votes.find((r: any) => r.party === 'psoe').yes).toBe(0);
    pick(a, 'spanish_censure.mediate');
    expect(a.qualities.party_resources).toBe(8);
    expect(a.qualities.censure_can_negotiate).toBe(true);
    pick(a, 'spanish_censure.negotiate');
    expect(a.currentFrame().choices.find(c => c.id === 'congreso_coalition_negotiation.referendum')?.canChoose).toBe(false);
    pick(a, 'congreso_coalition_negotiation.cash');
    pick(a, 'congreso_coalition_negotiation.resolve');
    expect(a.qualities.spanish_pm).toBe('Pablo Iglesias');
    expect(a.qualities.congreso_agreement_lead_party).toBe('up');
    expect(a.qualities.spanish_coalition).toEqual(['up']);
  });

  it('does not sell Madina support when political conditions or resources are missing', () => {
    for (const overrides of [{ cat_spa_relations: 20 }, { party_resources: 1 }, { up_congreso_s: 59 }]) {
      const a = setup({ psoe_leader: 'Eduardo Madina', psoe_congreso_s: 60, up_congreso_s: 100, ...overrides });
      if (!available(a, 'podemos')) continue;
      pick(a, 'spanish_censure');
      expect(a.qualities.censure_can_mediate).toBe(false);
    }
  });

  it('uses CiU persuasion to make a majority viable and brings PNV aboard without outreach', () => {
    const a = setup({ psoe_congreso_s: 90, up_congreso_s: 70, pdcat_congreso_s: 10, pp_congreso_s: 164,
      player_party: 'cup', erc_congreso_s: 0, censure_attempted: { psoe: true } });
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_yes_without_player).toBe(160);
    pick(a, 'spanish_censure.persuade_ciu');
    expect(a.qualities.censure_yes_without_player).toBe(176);
    expect(a.qualities.censure_can_negotiate).toBe(false);
    expect(a.qualities.censure_can_persuade_ciu).toBe(false);
    pick(a, 'spanish_censure.observe');
    expect(a.qualities.censure_won).toBe(true);
  });

  it('resets attempts for a new chamber, not a new PM', () => {
    const a = setup();
    pick(a, 'spanish_censure');
    pick(a, 'spanish_censure.abstain');
    a.qualities.spanish_pm_party = 'cs';
    a.goToScene('post_event');
    expect(a.qualities.censure_attempted.psoe).toBe(true);
    a.goToScene('congreso_election.post_election');
    expect(a.qualities.censure_attempted).toEqual({});
    expect(a.qualities.censure_podemos_countdown).toBe(6);
    expect(available(a, 'psoe')).toBe(false);
  });

  it('preserves mediated votes and the used attempt across save/load', () => {
    const a = setup({ psoe_leader: '???' });
    pick(a, 'spanish_censure');
    pick(a, 'spanish_censure.mediate');
    const saved = a.exportStateJSON();
    const restored = DendryAdapter.fromJSONText(gameText);
    restored.importStateJSON(saved);
    expect(restored.qualities.censure_current.votes.find((r: any) => r.party === 'psoe').yes).toBe(60);
    expect(restored.qualities.censure_attempted.podemos).toBe(true);
    expect(restored.qualities.censure_can_mediate).toBe(false);
    pick(restored, 'spanish_censure.support');
    expect(restored.qualities.censure_vote_result.outcomes.reduce((n: number, r: any) => n + r.votes, 0)).toBe(350);
  });

  it('retains ordinary post-election negotiation and its return destination', () => {
    const a = setup();
    a.qualities.congreso_left_carrier = 'up';
    a.qualities.congreso_coalition_progressive_talks = true;
    a.goToScene('congreso_coalition_progressive.progressive');
    pick(a, 'congreso_coalition_negotiation.cash');
    pick(a, 'congreso_coalition_negotiation.resolve');
    expect(a.qualities.spanish_pm).toBe('Pedro Sánchez');
    expect(a.qualities.spanish_coalition).toEqual(['psoe', 'up']);
    expect(a.qualities.negotiated_cash_countdown).toBe(8);
  });

  it('does not count a player joint delegation again as persuadable CiU seats', () => {
    const a = setup({ erc_in_jxcat: true, jxcat_in_congreso: true, jxcat_congreso_s: 10, erc_congreso_s: 0,
      censure_attempted: { psoe: true } });
    pick(a, 'spanish_censure');
    expect(a.qualities.censure_player_seats).toBe(10);
    expect(a.qualities.censure_current.votes.some((r: any) => r.party === 'jxcat')).toBe(false);
    expect(a.qualities.censure_can_persuade_ciu).toBe(false);
    pick(a, 'spanish_censure.support');
    expect(a.qualities.censure_yes).toBe(176);
  });
});

