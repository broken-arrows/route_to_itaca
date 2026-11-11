import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(
  resolve(import.meta.dirname, '..', '..', 'out', 'game.json'),
  'utf8',
);

const parties = [
  'ciu', 'cdc', 'dl', 'pdcat', 'jxsi', 'jxcat', 'junts', 'erc', 'cup',
  'psc', 'ppc', 'cs', 'icv', 'csqp', 'cecp', 'ecp', 'unio', 'si',
  'vox', 'pxc', 'fnc',
];

function singlePartyVote(
  party: string,
  leaderKey: string,
  leader: string,
  variant = 'national_scope',
) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  const q = adapter.qualities as Record<string, any>;
  for (const p of parties) q[`${p}_parlament_s`] = 0;
  q[`${party}_parlament_s`] = 10;
  q[leaderKey] = leader;
  q.parlament_digital_variant = variant;
  q.player_party = 'erc';
  q.player_in_jxsi = false;
  q.player_in_jxcat = false;
  q.coming_from_parlament = false;

  adapter.goToScene('parlament_digital.digital_vote');
  return {
    yes: q.parlament_digital_yes,
    abstain: q.parlament_digital_abst,
    no: q.parlament_digital_no,
  };
}

describe('digital-law compiled behaviour', () => {
  it.each([0, -1])('passes a player bill with %s party resources and consumes the parliamentary opportunity', (resources) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    adapter.goToScene('root.start_menu');
    adapter.goToScene('root.start');
    const q = adapter.qualities as Record<string, any>;
    for (const p of parties) q[`${p}_parlament_s`] = 0;
    Object.assign(q, {
      player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
      erc_parlament_s: 70, ppc_parlament_s: 65, year: 2014,
      cat_caretaker_gov: false, parlament_timer: 0, month_actions: 0,
      party_resources: resources,
    });
    const choose = (id: string) => {
      const choices = adapter.currentFrame().choices;
      const index = choices.findIndex(choice => choice.id === id);
      expect(index, id).toBeGreaterThanOrEqual(0);
      expect(choices[index].canChoose, id).toBe(true);
      adapter.choose(index);
    };
    adapter.goToScene('parlament_card');
    choose('parlament_card.rights');
    choose('parlament_digital');
    choose('parlament_digital.digital_public_services');
    choose('parlament_digital.digital_vote');
    expect(q.parlament_digital_passed).toBe(true);
    expect(q.party_resources).toBe(resources);
    expect(q.parlament_timer).toBe(6);
    expect(q.month_actions).toBe(1);
  });

  it('does not grant resources when a bank-tax draft is abandoned', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    adapter.goToScene('root.start_menu');
    adapter.goToScene('root.start');
    const q = adapter.qualities as Record<string, any>;
    Object.assign(q, { party_resources: 0, coming_from_parlament: true, month_actions: 1 });
    adapter.goToScene('parlament_bank_taxes.back_to_parlament');
    expect(q.party_resources).toBe(0);
    expect(adapter.engine.state.sceneId).toBe('parlament_card.parlament_menu');
  });

  it.each([
    ['Montserrat Tura', [10, 0, 0]],
    ['Àngel Ros', [0, 10, 0]],
    ['Núria Parlon', [0, 10, 0]],
    ['Miquel Iceta', [0, 0, 10]],
    ['Pere Navarro', [0, 0, 10]],
  ])('maps maximal PSC vote under %s', (leader, expected) => {
    const vote = singlePartyVote('psc', 'psc_leader', leader);
    expect([vote.yes, vote.abstain, vote.no]).toEqual(expected);
  });

  it.each([
    ['csqp', 'csqp_leader', 'Arcadi Oliveres', [10, 0, 0]],
    ['csqp', 'csqp_leader', 'Lluís Rabell', [0, 10, 0]],
    ['cecp', 'cecp_leader', 'Arcadi Oliveres', [10, 0, 0]],
    ['cecp', 'cecp_leader', 'Jaume Asens', [10, 0, 0]],
    ['cecp', 'cecp_leader', 'Xavier Domènech', [0, 10, 0]],
    ['cecp', 'cecp_leader', 'Joan Coscubiela', [0, 0, 10]],
    ['ecp', 'ecp_leader', 'Jéssica Albiach', [0, 10, 0]],
    ['icv', 'icv_leader', 'Joan Herrera', [0, 10, 0]],
  ])(
    'maps maximal federal-left vote for %s under %s',
    (party, leaderKey, leader, expected) => {
      const vote = singlePartyVote(party, leaderKey, leader);
      expect([vote.yes, vote.abstain, vote.no]).toEqual(expected);
    },
  );

  it('keeps the safe agency supported across the PSC and federal left', () => {
    for (const [party, leaderKey, leader] of [
      ['psc', 'psc_leader', 'Miquel Iceta'],
      ['icv', 'icv_leader', 'Joan Herrera'],
      ['csqp', 'csqp_leader', 'Lluís Rabell'],
      ['cecp', 'cecp_leader', 'Joan Coscubiela'],
      ['ecp', 'ecp_leader', 'Jéssica Albiach'],
    ]) {
      expect(
        singlePartyVote(party, leaderKey, leader, 'public_services'),
      ).toEqual({ yes: 10, abstain: 0, no: 0 });
    }
  });

  it('shows only the ruling responses belonging to the enacted draft', () => {
    const choicesFor = (variant: string) => {
      const adapter = DendryAdapter.fromJSONText(gameText);
      adapter.beginGame([1, 2, 3, 4]);
      const q = adapter.qualities as Record<string, any>;
      q.parlament_digital_variant = variant;
      q.constitutional_digital_trigger = 1;
      return adapter
        .goToScene('constitutional_digital')
        .choices.map((choice) => choice.id);
    };

    expect(choicesFor('public_services')).toEqual([
      'constitutional_digital.digital_ruling_safe',
    ]);
    expect(choicesFor('national_scope')).toEqual([
      'constitutional_digital.digital_ruling_accept',
      'constitutional_digital.digital_ruling_protest',
    ]);
  });
});
