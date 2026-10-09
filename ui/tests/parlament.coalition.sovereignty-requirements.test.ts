import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const directory = resolve(import.meta.dirname, '../../source/scenes/events/elections');
const files = readdirSync(directory).filter(name => /^parlament_coalition_(joint_cup|ciu_erc_cup)_phase_/.test(name));
function run(file: string, q: Record<string, any>, marker = '') {
  if (!marker && file.includes('_phase_')) marker = 'prepare';
  let source = readFileSync(resolve(directory, file), 'utf8');
  if (marker) {
    const start = source.search(new RegExp(`^@${marker}\\r?$`, 'm'));
    expect(start).toBeGreaterThanOrEqual(0);
    source = source.slice(start);
  }
  const block = source.match(/on-arrival:\s*\{!([\s\S]*?)!\}/)![1];
  new Function('Q', 'G', block)(q, { random: () => 0.8 });
}
function fixture(file: string): Record<string, any> {
  const joint = file.includes('joint_cup');
  const leader = joint ? (file.includes('jxsi') ? 'jxsi' : 'jxcat') : file.includes('_erc_led') ? 'erc' : 'ciu';
  const q: Record<string, any> = {
    player_party: file.includes('_erc_player_') ? 'erc' : 'cup',
    parlament_current_ciu: 'ciu', parlament_coalition_leader: leader,
    parlament_coalition_follower: leader === 'erc' ? 'ciu' : 'erc',
    parlament_coalition_sovereignty_kind: joint ? leader : file.includes('_government') ? 'government' : 'support',
    parlament_coalition_sovereignty_cabinet: joint ? [leader] : ['ciu', 'erc', 'cup'],
    parlament_coalition_sovereignty_phase: 3, parlament_coalition_sovereignty_mas_pressure: 0,
    parlament_coalition_support_seats: 80, erc_parlament_s: 35, cup_parlament_s: 10,
    parlament_coalition_proposal_president_name: 'Carles Puigdemont',
    parlament_coalition_leverage: 40, coalition_phase_timer: 0,
    parlament_coalition_concessions_indy: 1, parlament_coalition_policy_health: true,
    parlament_coalition_sovereignty_deadline: 2015, speaker_party: 'cup',
  };
  for (const office of ['vp', 'interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl']) {
    q[`parlament_coalition_proposal_${office}`] = joint ? leader : 'erc';
  }
  if (!joint) {
    q.parlament_coalition_proposal_economy = 'ciu';
    q.parlament_coalition_proposal_health = 'cup';
    q.parlament_coalition_proposal_education = 'cup';
  }
  return q;
}

describe('sovereignty pact requirements', () => {
  it('resolves both dispute types independently without repeating either', () => {
    const q = fixture(files[0]);
    Object.assign(q, {
      coalition_phase_timer: 1, parlament_coalition_sovereignty_phase: 2,
      parlament_coalition_sovereignty_dispute_round: 0,
      parlament_coalition_sovereignty_mediation_round: 0,
      parlament_coalition_sovereignty_feud: 1,
      cup_endavant_strength: 30, cup_poblelliure_strength: 40, cup_thirdsector_strength: 30,
      cup_endavant_dissent: 0, cup_poblelliure_dissent: 0, cup_thirdsector_dissent: 0,
      parlament_coalition_sovereignty_dispute_action: 'defend',
    });
    q.parlament_coalition_proposal_president_name = 'Artur Mas';
    const file = 'parlament_coalition_sovereignty_dispute.scene.dry';
    run('parlament_coalition_sovereignty_candidate_erc.scene.dry', q, 'defend');
    expect(q.parlament_coalition_sovereignty_dispute_round).toBe(2);
    expect(q.parlament_coalition_sovereignty_mediation_round).toBe(0);
    q.parlament_coalition_sovereignty_dispute_action = 'mediate';
    run(file, q, 'resolve');
    expect(q.parlament_coalition_sovereignty_mediation_round).toBe(2);
    const snapshot = JSON.stringify(q);
    run('parlament_coalition_sovereignty_candidate_erc.scene.dry', q, 'defend');
    expect(JSON.stringify(q)).toBe(snapshot);
    q.parlament_coalition_sovereignty_dispute_action = 'defend';
    const repeated = JSON.stringify(q);
    run(file, q, 'resolve');
    expect(JSON.stringify(q)).toBe(repeated);
  });

  it('resets negotiation state through the shared initializer in all twelve openings', () => {
    for (const file of ['parlament_coalition_joint_cup.scene.dry', 'parlament_coalition_ciu_erc_cup.scene.dry']) {
      const source = readFileSync(resolve(directory, file), 'utf8');
      const entries = [...source.matchAll(/^@(\w+_player_\w+)\r?$/gm)].map(match => match[1]);
      expect(entries.length).toBe(file.includes('joint') ? 4 : 8);
      for (const entry of entries) {
        const q = JSON.parse(JSON.stringify({
          ...fixture(files[0]), parlament_coalition_sovereignty_assembly_result: 'veto',
          parlament_coalition_sovereignty_banos_departure: true,
          parlament_coalition_sovereignty_mediation_round: 3,
          parlament_current_ciu: 'ciu', parlament_current_ciu_s: 35,
          ciu_parlament_s: 35, jxsi_parlament_s: 60, jxcat_parlament_s: 60,
          cup_endavant_strength: 30, cup_poblelliure_strength: 40, cup_thirdsector_strength: 30,
        }));
        run(file, q, entry);
        const node = source.slice(source.indexOf('@' + entry)).split(/\r?\n@/)[0];
        const called = node.match(/^call: (\w+)\.(\w+)/m)!;
        expect(called, entry).toBeTruthy();
        run(called[1] + '.scene.dry', q, called[2]);
        expect(q.parlament_coalition_sovereignty_assembly_result, entry).toBe('');
        expect(q.parlament_coalition_sovereignty_banos_departure, entry).toBe(false);
        expect(q.parlament_coalition_sovereignty_mediation_round).toBe(0);
      }
    }
  });

  it.each(files.filter(file => file.includes('_erc_player_')))('requires explicit independence even with a deadline and CUP Speaker: %s', file => {
    const q = fixture(file);
    q.parlament_coalition_concessions_indy = 0;
    run(file, q, 'submit');
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    expect(q.parlament_coalition_sovereignty_reject_reasons).toContain('explicit independence commitment');
    q.parlament_coalition_concessions_indy = 1;
    run(file, q, 'submit');
    expect(q.parlament_coalition_sovereignty_reject_reasons).toBe('');
    expect(q.parlament_coalition_proposal_accept).toBe(true);
  });

  it.each(files.filter(file => file.includes('joint_cup') || file.includes('_support')))('does not substitute the Speaker for a national commitment: %s', file => {
    const q = fixture(file);
    q.parlament_coalition_concessions_indy = 0;
    q.parlament_coalition_sovereignty_deadline = 0;
    run(file, q, 'submit');
    expect(q.parlament_coalition_proposal_accept).toBe(false);
  });

  it.each(files)('charges the Speaker recipient once per proposal, across all rounds: %s', file => {
    for (const phase of [1, 2, 3]) {
      const q = fixture(file);
      q.parlament_coalition_sovereignty_phase = phase;
      q.speaker_party = 'none';
      run(file, q);
      const baseline = q.parlament_coalition_leverage;
      q.speaker_party = file.includes('joint_cup') && q.player_party === 'erc' ? q.parlament_coalition_leader : q.player_party;
      run(file, q);
      expect(q.parlament_coalition_leverage).toBe(baseline - 5);
      run(file, q);
      expect(q.parlament_coalition_leverage).toBe(baseline - 5);
    }
  });
});
