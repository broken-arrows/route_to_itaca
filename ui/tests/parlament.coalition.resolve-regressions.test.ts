import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
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

  it('registers the countdown id whose countdown value it initializes', () => {
    const q = resolveCoalition({
      jxsi_formed: true,
      jxsi_parlament_s: 62,
      countdowns: [],
    });

    expect(q.countdowns).toContain('jxcat_formation');
    expect(q.jxcat_formation_countdown).toBeGreaterThan(0);
  });
});
