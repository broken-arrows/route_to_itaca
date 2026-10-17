import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');

function resolveCoalition(overrides: Record<string, unknown>) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  Object.assign(adapter.qualities, {
    player_party: 'erc',
    cat_coalition: ['psc', 'ecp'],
    cat_coalition_support: [],
    cat_coalition_abstain: ['erc'],
    parlament_coalition_proposal_president: 'psc',
    parlament_coalition_proposal_president_name: 'Salvador Illa',
    parlament_coalition_proposal_president_gender: 'm',
    speaker_party: 'erc',
    speaker: 'Roger Torrent',
    speaker_gender: 'm',
    psc_parlament_s: 45,
    ecp_parlament_s: 25,
    ...overrides,
  });
  adapter.goToScene('parlament_coalition_resolve');
  return adapter.qualities;
}

describe('Parlament coalition resolution', () => {
  it('requires separate laws for exceptional Education and Health transitions', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    for (const [sector, spending] of [['education', 'education_spending'], ['health', 'healthcare_spending']]) {
      for (const [current, target] of [[2, 3], [-3, -2], [3, 2], [-2, -3]]) {
        Object.assign(adapter.qualities, {
          parlament_coalition_leverage: 100,
          parlament_coalition_policy_spending_cost: 5,
          [spending]: current,
          [`parlament_coalition_policy_${sector}`]: false,
          [`parlament_coalition_policy_${sector}_target`]: target,
        });
        const choice = adapter.engine.game.scenes[`parlament_coalition_fiscal_policies.${sector}_policy`];
        expect(adapter.engine._runPredicate(choice.chooseIf, false)).toBe(false);
        if (current === 2 || current === 3 || current === -3) {
          expect(adapter.engine._runPredicate(choice.viewIf, true)).toBe(false);
        }
        const q = resolveCoalition({
          parlament_coalition_policy_accepted: true,
          [`parlament_coalition_policy_${sector}`]: true,
          [`parlament_coalition_policy_${sector}_target`]: target,
          [spending]: current,
          social_spending_cap: 2,
        });
        expect(q[spending]).toBe(current);
        expect(q.coalition_policy_commitments).toEqual([]);
        expect((q.active_mods as Record<string, unknown>)[`coalition_pending_${spending}`]).toBeUndefined();
      }
    }
  });

  it('derives cabinet, support, and abstention roles from the formal arrays', () => {
    const cases = [
      [['erc'], [], [], true, true, 'Generalitat'],
      [['psc'], ['erc'], [], false, true, "<span style='font-style: italic'>Government Support</span>"],
      [['psc'], [], ['erc'], false, false, "<span style='font-style: italic'>Opposition</span>"],
    ] as const;

    for (const [cabinet, support, abstain, inGovernment, supporting, position] of cases) {
      const q = resolveCoalition({
        cat_coalition: [...cabinet],
        cat_coalition_support: [...support],
        cat_coalition_abstain: [...abstain],
        parlament_coalition_proposal_president: cabinet[0],
        [`${cabinet[0]}_parlament_s`]: 70,
      });

      expect(q.erc_in_gob).toBe(inGovernment);
      expect(q.erc_supporting).toBe(supporting);
      expect(q.position_disp).toBe(position);
    }
  });

  it.each([
    { list: 'jxsi', role: 'cabinet', erc: true, cup: false, expected: [true, true, false, false] },
    { list: 'jxsi', role: 'cabinet', erc: true, cup: true, expected: [true, true, true, true] },
    { list: 'jxcat', role: 'cabinet', erc: false, cup: false, expected: [false, false, false, false] },
    { list: 'jxcat', role: 'support', erc: true, cup: true, expected: [false, true, false, true] },
    { list: 'jxcat', role: 'abstain', erc: true, cup: true, expected: [false, false, false, false] },
  ] as const)('derives constituent roles when $list is in $role', ({ list, role, erc, cup, expected }) => {
    const q = resolveCoalition({
      [`erc_in_${list}`]: erc,
      [`cup_in_${list}`]: cup,
      cat_coalition: role === 'cabinet' ? [list] : ['psc'],
      cat_coalition_support: role === 'support' ? [list] : [],
      cat_coalition_abstain: role === 'abstain' ? [list] : [],
      parlament_coalition_proposal_president: role === 'cabinet' ? list : 'psc',
      [`${list}_parlament_s`]: 62,
    });
    expect([q.erc_in_gob, q.erc_supporting, q.cup_in_gob, q.cup_supporting]).toEqual(expected);
  });

  it('registers the countdown id whose countdown value it initializes', () => {
    const q = resolveCoalition({
      jxsi_formed: true,
      jxsi_parlament_s: 62,
      countdowns: [],
    });

    expect(q.countdowns).toContain('jxcat_formation');
    expect(q.jxcat_formation_countdown).toBeGreaterThan(0);
  });

  it('starts executive and burden capacity for a player governing through JxSí', () => {
    const q = resolveCoalition({
      player_in_jxsi: true, erc_in_jxsi: true,
      cat_coalition: ['jxsi'], cat_coalition_abstain: [],
      parlament_coalition_proposal_president: 'jxsi',
      jxsi_parlament_s: 70, government_budget: 0,
    });
    expect(q.jxsi_in_gob).toBe(true);
    expect(q.position_disp).toBe('Generalitat');
    expect(q.government_budget).toBe(1);
    expect(q.burdens_start_countdown).toBe(1);
  });

  it.each(['jxsi', 'jxcat'])('describes CUP support when ERC governs through %s', (list) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      player_party: 'erc',
      [`erc_in_${list}`]: true,
      [`player_in_${list}`]: true,
      cat_coalition: [list],
      cat_coalition_support: ['cup'],
      cat_coalition_abstain: [],
      parlament_coalition_proposal_president: list,
      parlament_coalition_proposal_president_name: 'President',
      parlament_coalition_proposal_president_gender: 'm',
      speaker_party: list,
      speaker: 'Speaker',
      speaker_gender: 'f',
      [`${list}_parlament_s`]: 62,
      party_resources: 2,
      erc_dissent: 0,
      cup_relations: 50,
    });
    adapter.goToScene('parlament_coalition_resolve');
    expect(adapter.qualities.erc_in_gob).toBe(true);
    expect(adapter.qualities.erc_supporting).toBe(true);
    expect(adapter.qualities.cup_in_gob).toBe(false);
    expect(adapter.qualities.cup_supporting).toBe(true);

    const frame = adapter.goToScene('erc_inter_party.cup_relations');
    expect(frame.html.replace(/<[^>]*>/g, '')).toContain('The CUP is currently supporting our government');
    expect(adapter.qualities.cup_relations).toBe(55);
  });

  it('describes an ERC-CUP cabinet as governing together', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      player_party: 'erc',
      cat_coalition: ['erc', 'cup'],
      cat_coalition_support: [],
      cat_coalition_abstain: [],
      parlament_coalition_proposal_president: 'erc',
      parlament_coalition_proposal_president_name: 'President',
      parlament_coalition_proposal_president_gender: 'm',
      speaker_party: 'erc',
      speaker: 'Speaker',
      speaker_gender: 'f',
      erc_parlament_s: 40,
      cup_parlament_s: 30,
      party_resources: 2,
      erc_dissent: 0,
      cup_relations: 50,
    });
    adapter.goToScene('parlament_coalition_resolve');
    const frame = adapter.goToScene('erc_inter_party.cup_relations');
    expect(frame.html.replace(/<[^>]*>/g, '')).toContain('We are currently governing together with the CUP');
    expect(frame.html).not.toContain('currently supporting our government');
    expect(adapter.qualities.cup_relations).toBe(57);
  });

  it('applies the cabinet unemployment burden to ERC governing through a list', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    adapter.goToScene('root.start_menu');
    adapter.goToScene('root.start');
    Object.assign(adapter.qualities, {
      player_party: 'erc',
      erc_in_jxsi: true,
      cat_coalition: ['jxsi'],
      cat_coalition_support: ['cup'],
      cat_coalition_abstain: [],
      parlament_coalition_proposal_president: 'jxsi',
      parlament_coalition_proposal_president_name: 'President',
      parlament_coalition_proposal_president_gender: 'm',
      speaker_party: 'jxsi',
      speaker: 'Speaker',
      speaker_gender: 'f',
      jxsi_parlament_s: 62,
      month_actions: 1,
      unemployment: 25,
      erc_left_dissent: 0,
    });
    adapter.goToScene('parlament_coalition_resolve');
    const events = adapter.engine.game.scenes['post_event.events_choice'];
    const elections = adapter.engine.game.scenes['post_event.elections_choice'];
    const compile = adapter.engine._compileChoices.bind(adapter.engine);
    vi.spyOn(adapter.engine, '_compileChoices').mockImplementation((scene) =>
      scene === events || scene === elections ? [{ title: 'Continue...' }] as any : compile(scene));
    adapter.goToScene('post_event');
    expect(adapter.qualities.month).toBe(9);
    expect(adapter.qualities.erc_left_dissent).toBe(0.5);
  });

  it('applies an accepted Education policy when the shared cap allows it', () => {
    const q = resolveCoalition({
      parlament_coalition_policy_accepted: true,
      parlament_coalition_policy_education: true,
      parlament_coalition_policy_education_target: 0,
      education_spending: -1,
      social_spending_cap: 0,
      education_cuts: true,
    });

    expect(q.education_spending).toBe(0);
    expect(q.education_cuts).toBe(false);
    expect(q.coalition_policy_commitments).toContainEqual(expect.objectContaining({
      kind: 'education_spending', target: 0, scope: 'cabinet', signatories: ['psc', 'ecp'], status: 'applied',
    }));
    expect((q.active_mods as Record<string, unknown>).coalition_pending_education_spending).toBeUndefined();
  });

  it('keeps an accepted Education policy inert until the budget raises the shared cap', () => {
    const q = resolveCoalition({
      parlament_coalition_policy_accepted: true,
      parlament_coalition_policy_education: true,
      parlament_coalition_policy_education_target: 0,
      education_spending: -1,
      social_spending_cap: -1,
      education_cuts: true,
    });

    expect(q.education_spending).toBe(-1);
    expect(q.education_cuts).toBe(true);
    expect(q.coalition_policy_commitments).toContainEqual(expect.objectContaining({
      kind: 'education_spending', target: 0, scope: 'cabinet', signatories: ['psc', 'ecp'], status: 'pending_budget',
    }));
    const pending = (q.active_mods as Record<string, any>).coalition_pending_education_spending;
    expect(pending.def.colour).toBe('gray');
    expect(pending.def.targets).toEqual({});
  });
});
