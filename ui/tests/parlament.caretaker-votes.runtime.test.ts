import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
const votes = [
  ['parlament_exec_education_public', 'education_public_introduced'],
  ['parlament_exec_education_de_public', 'education_de_public_introduced'],
  ['parlament_exec_education_privatize', 'education_privatization_introduced'],
  ['parlament_exec_education_de_privatize', 'education_de_privatization_introduced'],
  ['parlament_exec_healthcare_public', 'healthcare_public_introduced'],
  ['parlament_exec_healthcare_de_public', 'healthcare_de_public_introduced'],
  ['parlament_exec_healthcare_privatize', 'healthcare_privatization_introduced'],
  ['parlament_exec_healthcare_de_privatize', 'healthcare_de_privatization_introduced'],
] as const;
let errors: string[];

beforeEach(() => {
  errors = [];
  vi.spyOn(console, 'error').mockImplementation((...args) => {
    errors.push(args.map(String).join(' '));
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  expect(errors).toEqual([]);
});

function setup(overrides: Record<string, unknown> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  for (const key of Object.keys(q)) if (key.endsWith('_parlament_s')) q[key] = 0;
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    erc_in_jxsi: false, erc_in_jxcat: false, cat_coalition: ['erc'],
    cat_coalition_support: [], cat_coalition_abstain: [],
    president_party: 'erc', education_minister_party: 'erc', health_minister_party: 'erc',
    erc_in_gob: true, erc_parlament_s: 80, ppc_parlament_s: 55, parlament_s_majority: 68,
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv', cat_caretaker_gov: false,
    year: 2013, month: 1, week: 1, time: 5, rubicon: false, difficulty: 0,
    month_actions: 0, month_actions_last: 0, next_election_year: 2016,
    next_election_month: 11, next_election_week: 1, next_election_time: 50,
    budget_year: 2013, government_budget: 10, party_resources: 10,
    education_spending: 2, social_spending_cap: 3, fla_escalated: false,
    coalition_policy_commitments: [], coalition_indy_commitments: [],
    coalition_rupture_trigger: false, government_does_things_timer: 99,
    parlament_snap_election: 0, ...overrides,
  });
  return { adapter, q };
}

function pick(adapter: DendryAdapter, id: string) {
  const index = adapter.engine.choiceCache.findIndex(choice => choice.id === id);
  expect(index, id).toBeGreaterThanOrEqual(0);
  expect(adapter.engine.choiceCache[index].canChoose, id).toBe(true);
  return adapter.choose(index);
}

describe('pending Parlament votes during caretaker government', () => {
  it.each(votes)('blocks %s with the institutional explanation and preserves its proposal', (id, flag) => {
    const { adapter, q } = setup({ [flag]: true });
    const active = adapter.goToScene('post_event.events_choice').choices.find(c => c.id === id)!;
    expect(active.canChoose).toBe(true);

    q.cat_caretaker_gov = true;
    const reason = adapter.goToScene('main').pinned.find(c => c.id === 'parlament_card')!.subtitle;
    expect(reason).toBeTruthy();
    const frame = adapter.goToScene('post_event.events_choice');
    const index = frame.choices.findIndex(c => c.id === id);
    expect(index).toBeGreaterThanOrEqual(0);
    expect(frame.choices[index]).toMatchObject({ canChoose: false, subtitle: reason });
    const before = JSON.stringify(q);
    expect(() => adapter.choose(index)).toThrow(/unavailable/);
    expect(JSON.stringify(q)).toBe(before);

    q.cat_caretaker_gov = false;
    const resumed = adapter.goToScene('post_event.events_choice').choices.find(c => c.id === id)!;
    expect(resumed.canChoose).toBe(true);
    expect(resumed.subtitle).toBe(active.subtitle);
    expect(q[flag]).toBe(true);
  });

  it('ignores a blocked-only edition when advancing time and resumes the votes after caretaker', () => {
    const { adapter, q } = setup({ cat_caretaker_gov: true, month_actions: 1 });
    // Isolate the real vote events so unrelated historical stories cannot mask a queue stall.
    adapter.engine.game.tagLookup = {
      ...adapter.engine.game.tagLookup,
      event: Object.fromEntries(votes.map(([id]) => [id, true])), election: {},
    };
    for (const [, flag] of votes) q[flag] = true;
    adapter.goToScene('post_event');
    expect(q.time).toBe(6);
    expect(q.month).toBe(2);
    expect(q.month_actions).toBe(0);
    expect(q.has_event).toBe(0);
    expect(adapter.engine.state.sceneId).toBe('main');
    for (const [, flag] of votes) expect(q[flag]).toBe(true);

    q.cat_caretaker_gov = false;
    q.month_actions = 1;
    const frame = adapter.goToScene('post_event');
    expect(q.time).toBe(6);
    expect(q.month_actions).toBe(1);
    expect(frame.sceneId).toBe('post_event.events_choice');
    expect(frame.choices).toHaveLength(votes.length);
    expect(frame.choices.every(c => c.canChoose)).toBe(true);
  });

  it('keeps a submitted bill blocked after dissolution while other news remains playable', () => {
    const { adapter, q } = setup();
    adapter.engine.game.tagLookup = {
      ...adapter.engine.game.tagLookup,
      event: { snap_election: true, parlament_exec_education_public: true, barcenas_leak: true },
      election: {},
    };
    adapter.goToScene('generalitat_education_card');
    pick(adapter, 'generalitat_education_card.education_public');
    expect(q.education_public_introduced).toBe(true);
    q.parlament_snap_election = 1;
    adapter.goToScene('post_event');
    pick(adapter, 'snap_election');
    expect(q.cat_caretaker_gov).toBe(true);
    const frame = pick(adapter, 'post_event');
    expect(frame.choices.find(c => c.id === 'parlament_exec_education_public')?.canChoose).toBe(false);
    expect(q.month).toBe(1);
    expect(q.month_actions).toBe(1);
    pick(adapter, 'barcenas_leak');
    pick(adapter, 'post_event');
    expect(q.month).toBe(2);
    expect(q.month_actions).toBe(0);
    expect(adapter.engine.state.sceneId).toBe('main');
    expect(q.education_public_introduced).toBe(true);
    expect(q.education_spending).toBe(2);
    expect(q.government_budget).toBe(10);
  });
});
