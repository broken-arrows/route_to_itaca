import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const elections = resolve(import.meta.dirname, '..', '..', 'source', 'scenes', 'events', 'elections');
const scene = (name: string) => readFileSync(resolve(elections, name), 'utf8');
const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');

function arrival(source: string, marker = '') {
  const start = marker ? source.search(new RegExp(`^${marker}\\r?$`, 'm')) : 0;
  expect(start, `missing scene marker ${marker}`).toBeGreaterThanOrEqual(0);
  const block = source.slice(start).match(/on-arrival:\s*\{!([\s\S]*?)!\}/)?.[1];
  expect(block, `missing on-arrival after ${marker || 'scene start'}`).toBeTruthy();
  return (Q: Record<string, any>) => new Function('Q', block!)(Q);
}

describe('Parlament coalition phase regressions', () => {
  it('blocks every leverage push in the final negotiation round', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    const cases = [
      ['parlament_coalition_ciu_erc_phase.push_leverage_concessions', 'ciu_erc_coalition_phase'],
      ['parlament_coalition_ciu_erc_phase.push_leverage_positions', 'ciu_erc_coalition_phase'],
      ['parlament_coalition_ciu_minority_phase.push_leverage_concessions', 'ciu_min_coalition_phase'],
      ['parlament_coalition_erc_ciu_phase.push_leverage_positions', 'erc_ciu_coalition_phase'],
      ['parlament_coalition_left_tripartite_phase.push_leverage_positions', 'left_tripartite_coalition_phase'],
      ['parlament_coalition_psc_tripartite_phase.push_leverage_positions', 'psc_tripartite_coalition_phase'],
      ['parlament_coalition_cup_broad_phase.push', 'cup_broad_coalition_phase'],
      ['parlament_coalition_erc_cup_phase_erc_player_erc_led.push', 'erc_cup_coalition_phase'],
      ['parlament_coalition_erc_cup_phase_erc_player_cup_led.push', 'erc_cup_coalition_phase'],
      ['parlament_coalition_erc_cup_phase_cup_player_erc_led.push', 'erc_cup_coalition_phase'],
      ['parlament_coalition_erc_cup_phase_cup_player_cup_led.push', 'erc_cup_coalition_phase'],
    ] as const;

    for (const [sceneId, phaseKey] of cases) {
      adapter.qualities.parlament_coalition_push = 0;
      adapter.qualities[phaseKey] = 3;
      const push = adapter.engine.game.scenes[sceneId];
      expect(push, sceneId).toBeTruthy();
      expect(adapter.engine._runPredicate(push.viewIf, true), sceneId).toBe(false);
      expect(adapter.engine._runPredicate(push.chooseIf, false), sceneId).toBe(false);
    }
  });

  it('keeps engine round controls out of player-facing wording', () => {
    const phaseFiles = readdirSync(elections).filter((name) => name.includes('_phase') && name.endsWith('.scene.dry'));
    const proceduralChoices = /- @revise:|return without sending|request another round|demand another round|ask for a further round|reject the distribution/i;

    for (const phaseFile of phaseFiles) {
      expect(scene(phaseFile), phaseFile).not.toMatch(proceduralChoices);
    }
  });

  it('does not offer cabinet ministries to an external CUP supporter', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    const ministries = adapter.engine.game.scenes['parlament_coalition_cup_broad_phase.ministries'];

    Object.assign(adapter.qualities, { player_party: 'cup', cup_broad_kind: 'republican' });
    expect(adapter.engine._runPredicate(ministries.viewIf, true)).toBe(false);

    adapter.qualities.cup_broad_kind = 'popular_unity';
    expect(adapter.engine._runPredicate(ministries.viewIf, true)).toBe(true);
  });

  it('charges the correct Left Tripartite partner and only applies the first-round ERC portfolio veto', () => {
    const setup = arrival(scene('parlament_coalition_left_tripartite.scene.dry'), '@erc');
    const phase = scene('parlament_coalition_left_tripartite_phase.scene.dry');
    const allocate = arrival(phase, '@go_back');
    const submit = arrival(phase, '@send_back');
    const q: Record<string, any> = {
      erc_parlament_s: 30,
      psc_parlament_s: 30,
      parlament_current_icv_s: 30,
      parlament_coalition_erc_psc_icv_s: 90,
      parlament_current_icv: 'icv',
      parlament_current_icv_leader: 'Joan Herrera',
      parlament_current_icv_relations: 40,
      psc_leader: 'Pere Navarro',
      psc_relations: 50,
      party_resources: 3,
      month_actions: 0,
      left_tripartite_coalition_phase: 1,
    };

    setup(q);
    expect(q.parlament_coalition_leverage).toBeCloseTo(30 * 55 / 90 - 10);

    Object.assign(q, {
      speaker_party: 'erc',
      parlament_coalition_proposal_vp: 'psc',
      parlament_coalition_proposal_economy: 'icv',
      parlament_coalition_proposal_interior: null,
      parlament_coalition_proposal_foreign: 'erc',
      parlament_coalition_proposal_health: 'erc',
      parlament_coalition_proposal_education: 'erc',
      parlament_coalition_proposal_territory: 'erc',
      parlament_coalition_proposal_bnl: 'erc',
    });
    allocate(q);
    expect(q.parlament_coalition_proposal_interior).toBe('psc');

    Object.assign(q, {
      parlament_coalition_changes: true,
      parlament_coalition_push: 0,
      parlament_coalition_proposal_vp: 'psc',
      parlament_coalition_proposal_interior: 'psc',
      parlament_coalition_proposal_education: 'icv',
    });
    submit(q);
    expect(q.parlament_coalition_proposal_accept).toBe(true);

    q.parlament_coalition_proposal_interior = 'erc';
    submit(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
  });

  it('treats an ERC Speaker as a cost in PSC-led Tripartite cabinet bargaining', () => {
    const enterPhase = arrival(scene('parlament_coalition_psc_tripartite_phase.scene.dry'));
    const base = {
      erc_parlament_s: 30,
      parlament_current_icv_s: 15,
      parlament_coalition_psc_erc_icv_s: 70,
      parlament_current_icv: 'icv',
      parlament_coalition_push: 0,
      psc_tripartite_coalition_phase: 1,
    };
    const withErcSpeaker: Record<string, any> = { ...base, speaker_party: 'erc' };
    const withoutErcSpeaker: Record<string, any> = { ...base, speaker_party: 'icv' };

    enterPhase(withErcSpeaker);
    enterPhase(withoutErcSpeaker);

    expect(withErcSpeaker.parlament_coalition_leverage)
      .toBeCloseTo(withoutErcSpeaker.parlament_coalition_leverage - 5);
  });

  it('resolves Popular Unity leadership ties and failed-talk counterparties deterministically', () => {
    const source = scene('parlament_coalition_cup_broad.scene.dry');
    const setup = arrival(source, '@setup');
    const fail = arrival(scene('parlament_coalition_cup_broad_phase.scene.dry'), '@new_elections');
    const makeQ = (erc: number, comuns: number, cup: number): Record<string, any> => ({
      cup_broad_kind: 'popular_unity',
      erc_parlament_s: erc,
      parlament_current_icv: 'icv',
      parlament_current_icv_s: comuns,
      parlament_current_icv_leader: 'Joan Herrera',
      cup_parlament_s: cup,
      year: 2015,
      parlament_coalition_push: 0,
    });

    const ercTie = makeQ(30, 30, 30);
    const comunsPlurality = makeQ(30, 31, 30);
    const cupComunsTie = makeQ(30, 31, 31);
    setup(ercTie);
    setup(comunsPlurality);
    setup(cupComunsTie);
    expect(ercTie.parlament_coalition_leader).toBe('erc');
    expect(comunsPlurality.parlament_coalition_leader).toBe('icv');
    expect(cupComunsTie.parlament_coalition_leader).toBe('cup');

    Object.assign(cupComunsTie, {
      player_party: 'cup',
      cup_broad_coalition_phase: 3,
      cup_relations: 60,
      icv_relations: 60,
    });
    fail(cupComunsTie);
    expect(cupComunsTie.snap_counterparty).not.toBe('cup');
    expect(cupComunsTie.snap_counterparty).toBe('erc');
  });
});
