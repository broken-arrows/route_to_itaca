import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');

describe('CSQP in the disconnection-law vote breakdowns', () => {
  it.each([
    ['vote', 'parlament_referendum_vote_result', 'Lluís Rabell', 'abstain'],
    ['vote', 'parlament_referendum_vote_result', 'Arcadi Oliveres', 'yes'],
    ['transition_law', 'parlament_disconnection_vote_result', 'Lluís Rabell', 'no'],
    ['transition_law', 'parlament_disconnection_vote_result', 'Arcadi Oliveres', 'yes'],
  ])('%s / %s / %s lists CSQP under %s', (scene, resultKey, leader, kind) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    const q = adapter.qualities as Record<string, any>;
    for (const key of Object.keys(q)) {
      if (key.endsWith('_parlament_s')) q[key] = 0;
    }
    Object.assign(q, {
      player_party: 'cup', parlament_current_ciu: 'cdc', cdc_roadmap: 2,
      erc_roadmap: 2, jxsi_formed: true, jxsi_dissolved: false,
      jxsi_parlament_s: 74, cup_parlament_s: 10, csqp_parlament_s: 9,
      psc_parlament_s: scene === 'vote' ? 16 : 0,
      cs_parlament_s: scene === 'vote' ? 20 : 36, ppc_parlament_s: 6,
      csqp_leader: leader, psc_leader: 'Miquel Iceta',
    });

    adapter.goToScene(`parlament_referendum.${scene}`);

    const outcomes = q[resultKey].outcomes;
    expect(outcomes.reduce((sum: number, outcome: any) => sum + outcome.votes, 0)).toBe(135);
    const outcome = outcomes.find((outcome: any) => outcome.kind === kind);
    expect(outcome.parties).toContainEqual({ label: 'CSQP' });
    expect(outcomes.filter((outcome: any) => outcome.parties.some((party: any) => party.label === 'CSQP'))).toHaveLength(1);
    expect(outcomes.find((outcome: any) => outcome.kind === 'not-present').votes).toBe(42);
  });
});
