import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

// Execute the authored arrivals directly, without relying on stale compiled output.
function arrive(path: string, scene: string, q: Record<string, any>) {
  const source = readFileSync(resolve(import.meta.dirname, '../../source/scenes/party_paths', path), 'utf8').replace(/\r\n/g, '\n');
  const body = source.split(`\n@${scene}\n`)[1].match(/on-arrival: \{!([\s\S]*?)!\}/)![1];
  const G = {
    text: (s: string) => s, literalText: (s: string) => s,
    setText: (q: Record<string, any>, key: string, value: unknown) => { q[key] = value; },
    governmentTooltipText: (_arena: unknown, _parties: unknown, _summary: unknown, label: string) => label,
  };
  new Function('Q', 'G', body).call({ state: { currentHands: {} } }, q, G);
}

function setup(joint: boolean) {
  return {
    cat_caretaker_gov: true, cat_coalition_disp: 'caretaker gov.', position_disp: 'Government Support',
    cat_coalition: [joint ? 'jxsi' : 'cdc'], cat_coalition_support: joint ? [] : ['erc'], cat_coalition_abstain: ['cup'],
    president: 'Artur Mas', president_party: joint ? 'jxsi' : 'cdc',
    erc_in_gob: joint, erc_supporting: true, cup_in_gob: false, cup_supporting: false,
    player_party: 'erc', parlament_current_ciu: 'cdc', cdc_leader: 'Artur Mas', erc_leader: 'Oriol Junqueras',
    jxsi_formed: joint, erc_in_jxsi: joint, jxsi_formation_num: 40,
    jxsi_parlament_s: joint ? 70 : 0, cdc_parlament_s: joint ? 0 : 40, erc_parlament_s: joint ? 0 : 30,
    cup_parlament_s: 10, jxcat_selected_leader: 'erc', jxcat_erc_available: true,
    jxcat_cup_join_available: true, jxcat_proposed_leader: 'Oriol Junqueras', party_resources: 10,
    next_election_time: 60, parlament_snap_election_countdown: 0,
  } as Record<string, any>;
}

const preserved = ['cat_caretaker_gov', 'cat_coalition_disp', 'position_disp', 'cat_coalition',
  'cat_coalition_support', 'cat_coalition_abstain', 'president', 'president_party',
  'erc_in_gob', 'erc_supporting', 'cup_in_gob', 'cup_supporting', 'next_election_time', 'parlament_snap_election_countdown'];
const snapshot = (q: Record<string, any>) => structuredClone(Object.fromEntries(preserved.map(k => [k, q[k]])));

describe('JxCat electoral decisions preserve the serving caretaker cabinet', () => {
  it.each(['formed', 'refuse'])('preserves the cabinet through the compiled %s route', (route) => {
    const adapter = DendryAdapter.fromJSONText(readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8'));
    adapter.beginGame([1, 2, 3, 4]);
    const q = adapter.qualities as Record<string, any>;
    Object.assign(q, setup(true), { jxcat_include_cup: true, jxcat_failure_available: true });
    const before = snapshot(q);
    adapter.goToScene(`jxcat_formation.${route}`);
    expect(snapshot(q)).toEqual(before);
    if (route === 'refuse') adapter.goToScene('jxsi_dissolution.cdc_alone');
    expect(snapshot(q)).toEqual(before);
    expect(q.jxcat_formed).toBe(true);
    expect(q.erc_in_jxcat).toBe(route === 'formed');
  });

  it('still remaps a non-caretaker cabinet when JxSí dissolves', () => {
    const q = setup(true);
    q.cat_caretaker_gov = false;
    arrive('jxsi/jxsi_dissolution.scene.dry', 'dissolve', q);
    expect(q.cat_coalition).toEqual(['cdc', 'erc']);
    expect(q.president_party).toBe('cdc');
    expect(q.erc_in_gob).toBe(true);
  });

  for (const joint of [false, true]) {
    for (const includeCup of [false, true]) {
      it(`forms with prior joint=${joint}, CUP joins=${includeCup}`, () => {
        const q = setup(joint);
        q.jxcat_include_cup = includeCup;
        const before = snapshot(q);
        arrive('jxcat/jxcat_formation.scene.dry', 'formed', q);
        expect(snapshot(q)).toEqual(before);
        expect(q.erc_in_jxcat).toBe(true);
        expect(q.cup_in_jxcat).toBe(includeCup);
        expect(q.jxcat_leader).toBe('Oriol Junqueras');
      });
    }
    it(`refuses with prior joint=${joint}`, () => {
      const q = setup(joint);
      const before = snapshot(q);
      arrive('jxcat/jxcat_formation.scene.dry', 'refuse', q);
      if (joint) {
        arrive('jxsi/jxsi_dissolution.scene.dry', 'dissolve', q);
        expect(snapshot(q)).toEqual(before);
      }
      arrive('jxcat/jxcat_formation.scene.dry', 'alone', q);
      expect(snapshot(q)).toEqual(before);
      expect(q.jxcat_formed).toBe(true);
      expect(q.erc_in_jxcat).toBe(false);
    });
  }
});
