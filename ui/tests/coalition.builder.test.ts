import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { mount } from '@vue/test-utils';
import { i18n } from '../src/i18n';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';
import { useGameStore } from '../src/stores/game';
import WidgetHost from '../src/components/viz/WidgetHost.vue';

const gameText = readFileSync(resolve(__dirname, '../../out/game.json'), 'utf8');
const classic = readFileSync(resolve(__dirname, '../../out/html/widgets.js'), 'utf8');

function setup(chamber: 'parlament' | 'congreso') {
  const a = DendryAdapter.fromJSONText(gameText);
  a.beginGame([1, 2, 3, 4]);
  const q = a.qualities;
  for (const key of Object.keys(q)) {
    if (key.endsWith(`_${chamber}_s`)) q[key] = 0;
  }
  const first = chamber === 'parlament' ? 'erc' : 'psoe';
  const second = chamber === 'parlament' ? 'psc' : 'pp';
  Object.assign(q, {
    player_party: 'erc', year: 2015, month: 10,
    [`${first}_${chamber}_s`]: chamber === 'parlament' ? 70 : 180,
    [`${second}_${chamber}_s`]: chamber === 'parlament' ? 65 : 170,
  });
  a.goToScene(`coalition_builder.${chamber}`);
  return { a, q, first, second };
}

beforeEach(() => setActivePinia(createPinia()));

describe('Parlament party layout and joint lists', () => {
  it('shows the fixed columns including zero-seat parties and cycles identities without changing the total', () => {
    const { q } = setup('parlament');
    const rows = (q.playground_coalitions as any).rows;
    expect([1, 2, 3].map(column => rows.filter((r: any) => r.column === column).map((r: any) => r.slot)))
      .toEqual([['ciu', 'erc', 'si', 'cup'], ['icv', 'pdcat', 'unio', 'pxc'], ['cs', 'psc', 'ppc', 'vox']]);
    for (const slot of ['cup', 'si', 'pxc', 'vox']) expect(rows.find((r: any) => r.slot === slot).hidden).not.toBe(true);
    gameLib.editCoalitionBuilder(q, 'icv', 5);
    const total = q.playground_coalitions_set_seats;
    for (const id of ['csqp', 'cecp', 'ecp', 'icv']) {
      gameLib.switchCoalitionBuilderParty(q, 'icv', 1);
      expect(rows.find((r: any) => r.slot === 'icv')).toMatchObject({ id, seats: 5 });
      expect(q.playground_coalitions_set_seats).toBe(total);
    }
    gameLib.switchCoalitionBuilderParty(q, 'pxc', 1);
    expect(rows.find((r: any) => r.slot === 'pxc').id).toBe('fnc');
  });

  it('merges a separate PDeCAT into the selected PDeCAT identity without duplicating seats', () => {
    const { q } = setup('parlament');
    gameLib.editCoalitionBuilder(q, 'ciu', 10);
    gameLib.switchCoalitionBuilderParty(q, 'joint', 1);
    gameLib.editCoalitionBuilder(q, 'pdcat', 3);
    const total = q.playground_coalitions_set_seats;
    for (let i = 0; i < 3; i++) gameLib.switchCoalitionBuilderParty(q, 'ciu', 1);
    const rows = (q.playground_coalitions as any).rows;
    expect(rows.find((r: any) => r.slot === 'ciu')).toMatchObject({ id: 'pdcat', seats: 13 });
    expect(rows.find((r: any) => r.slot === 'pdcat')).toMatchObject({ hidden: true, seats: 0 });
    gameLib.switchCoalitionBuilderParty(q, 'ciu', 1);
    expect(rows.find((r: any) => r.slot === 'pdcat').hidden).toBe(false);
    expect(q.playground_coalitions_set_seats).toBe(total);
  });

  it('initializes Junts alongside a separately seated PDeCAT', () => {
    const { a, q } = setup('parlament');
    Object.assign(q, { junts_parlament_s: 10, pdcat_parlament_s: 3, erc_parlament_s: 57, playground_coalitions: null });
    a.goToScene('coalition_builder.parlament');
    const rows = (q.playground_coalitions as any).rows;
    expect(rows.find((r: any) => r.slot === 'ciu')).toMatchObject({ id: 'junts', seats: 10 });
    expect(rows.find((r: any) => r.slot === 'pdcat')).toMatchObject({ hidden: false, seats: 3 });
  });

  it('gates PDeCAT and UDC by the selected identities and merges hidden seats without changing the total', () => {
    const { q } = setup('parlament');
    const rows = (q.playground_coalitions as any).rows;
    const slot = (name: string) => rows.find((r: any) => r.slot === name);
    expect(slot('pdcat').hidden).toBe(true);
    expect(slot('unio').hidden).toBe(true);
    expect(gameLib.editCoalitionBuilder(q, 'pdcat', 3)).toBe(false);
    expect(gameLib.editCoalitionBuilder(q, 'unio', 3)).toBe(false);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    expect(slot('ciu').member).toBe(true);
    expect(slot('unio').hidden).toBe(true);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    gameLib.switchCoalitionBuilderParty(q, 'ciu', 1);
    expect(slot('unio').hidden).toBe(false);
    gameLib.editCoalitionBuilder(q, 'unio', 3);
    const total = q.playground_coalitions_set_seats;
    gameLib.switchCoalitionBuilderParty(q, 'ciu', -1);
    expect(slot('unio')).toMatchObject({ hidden: true, seats: 0 });
    expect(slot('ciu').seats).toBe(3);
    gameLib.switchCoalitionBuilderParty(q, 'joint', 1);
    expect(slot('pdcat').hidden).toBe(false);
    gameLib.editCoalitionBuilder(q, 'pdcat', 5);
    gameLib.switchCoalitionBuilderParty(q, 'joint', -1);
    expect(slot('pdcat')).toMatchObject({ hidden: true, seats: 0 });
    expect(slot('ciu').seats).toBe(8);
    expect(q.playground_coalitions_set_seats).toBe(Number(total) + 5);
    gameLib.switchCoalitionBuilderParty(q, 'ciu', -1);
    expect(slot('ciu').id).toBe('junts');
    expect(slot('pdcat').hidden).toBe(false);
  });

  it('automatically absorbs CiU and SI, restores SI with ERC, and clears membership when the joint list is zeroed', () => {
    const { a, q } = setup('parlament');
    const rows = (q.playground_coalitions as any).rows;
    const slot = (name: string) => rows.find((r: any) => r.slot === name);
    gameLib.editCoalitionBuilder(q, 'ciu', 10);
    gameLib.editCoalitionBuilder(q, 'si', 3);
    gameLib.editCoalitionBuilder(q, 'psc', 52);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    expect(slot('joint').seats).toBe(83);
    expect(slot('ciu')).toMatchObject({ member: true, seats: 0, canJoin: true });
    expect(slot('si')).toMatchObject({ member: true, seats: 0, contribution: 3 });
    expect(q.playground_coalitions_set_seats).toBe(135);
    expect(gameLib.editCoalitionBuilder(q, 'si', 1)).toBe(false);
    expect(gameLib.toggleCoalitionBuilderMember(q, 'si')).toBe(false);
    const frame = a.refreshChoices();
    a.choose(frame.choices.findIndex(c => c.id.endsWith('run_parlament')));
    expect(q.si_parlament_s).toBe(0);
    expect(q.si_in_jxsi).toBe(true);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    expect(slot('erc').seats).toBe(70);
    expect(slot('si')).toMatchObject({ member: false, seats: 3 });
    expect(slot('joint').seats).toBe(10);
    gameLib.editCoalitionBuilder(q, 'jxsi', 0);
    expect(rows.some((r: any) => r.member)).toBe(false);
    expect(slot('joint').hidden).toBe(false);
    expect(slot('ciu')).toMatchObject({ seats: 0, contribution: 0 });
    gameLib.editCoalitionBuilder(q, 'ciu', 6);
    gameLib.editCoalitionBuilder(q, 'jxsi', 4);
    expect(slot('joint').seats).toBe(10);
    expect(slot('ciu')).toMatchObject({ member: true, seats: 0 });
  });

  it('initializes an existing joint list and its implied SI membership with no missing draft seats', () => {
    const { a, q } = setup('parlament');
    Object.assign(q, { jxcat_parlament_s: 50, ciu_parlament_s: 10, erc_parlament_s: 0,
      si_parlament_s: 3, psc_parlament_s: 72, erc_in_jxcat: true, playground_coalitions: null });
    a.goToScene('coalition_builder.parlament');
    const rows = (q.playground_coalitions as any).rows;
    expect(rows[0]).toMatchObject({ id: 'jxcat', seats: 63, hidden: false });
    expect(rows.find((r: any) => r.slot === 'si')).toMatchObject({ member: true, seats: 0 });
    expect(rows.reduce((sum: number, r: any) => sum + r.seats, 0)).toBe(135);
  });

  it('releases CiU when returning SI removes the last joint-list seats', () => {
    const { q } = setup('parlament');
    gameLib.editCoalitionBuilder(q, 'erc', 0);
    gameLib.editCoalitionBuilder(q, 'si', 5);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    const rows = (q.playground_coalitions as any).rows;
    expect(rows[0].seats).toBe(5);
    expect(rows.find((r: any) => r.slot === 'ciu').member).toBe(true);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    expect(rows[0].seats).toBe(0);
    expect(rows.find((r: any) => r.slot === 'ciu').member).toBe(false);
    expect(rows.find((r: any) => r.slot === 'si')).toMatchObject({ seats: 5, member: false });
  });

  it('preserves seats through membership changes, switches joint-list identity, and commits only on Run', () => {
    const { a, q } = setup('parlament');
    const rows = (q.playground_coalitions as any).rows;
    const joint = rows[0];
    const snapshot = JSON.stringify(q.election_simulation_parlament_hemicycle);
    expect(gameLib.toggleCoalitionBuilderMember(q, 'ciu')).toBe(true);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    gameLib.toggleCoalitionBuilderMember(q, 'cup');
    expect(joint).toMatchObject({ hidden: false, seats: 70 });
    expect(gameLib.editCoalitionBuilder(q, 'erc', 10)).toBe(false);
    expect(q.erc_parlament_s).toBe(70);
    expect(JSON.stringify(q.election_simulation_parlament_hemicycle)).toBe(snapshot);
    gameLib.switchCoalitionBuilderParty(q, 'joint', 1);
    const f = a.refreshChoices();
    a.choose(f.choices.findIndex(c => c.id.endsWith('run_parlament')));
    expect(q.jxcat_parlament_s).toBe(70);
    expect(q.erc_parlament_s).toBe(0);
    expect(q.erc_in_jxcat).toBe(true);
    expect(q.cup_in_jxcat).toBe(true);
    expect(q.player_in_jxcat).toBe(true);
    expect(q.player_in_jxsi).toBe(false);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    expect(rows.find((r: any) => r.slot === 'erc').seats).toBe(70);
    gameLib.toggleCoalitionBuilderMember(q, 'cup');
    expect(rows.find((r: any) => r.slot === 'ciu').member).toBe(true);
    expect(joint).toMatchObject({ hidden: false, seats: 0 });
    expect(q.playground_coalitions_set_seats).toBe(135);
  });

  it('caps returned contributions and leaves added joint seats with the automatic CiU member', () => {
    const { q } = setup('parlament');
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    gameLib.toggleCoalitionBuilderMember(q, 'cup');
    gameLib.editCoalitionBuilder(q, 'jxsi', 20);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    expect((q.playground_coalitions as any).rows.find((r: any) => r.id === 'erc').seats).toBe(20);
    gameLib.editCoalitionBuilder(q, 'jxsi', 5);
    gameLib.toggleCoalitionBuilderMember(q, 'cup');
    expect((q.playground_coalitions as any).rows.find((r: any) => r.id === 'cup').seats).toBe(0);
    expect((q.playground_coalitions as any).rows[0].seats).toBe(5);
    expect(q.playground_coalitions_set_seats).toBe(90);
  });
});

describe('Congreso party layout and independent joint lists', () => {
  function slot(q: Record<string, unknown>, name: string) {
    return (q.playground_coalitions as any).rows.find((r: any) => r.slot === name);
  }

  it('keeps all fixed party slots in the requested order, including GBAI and PRC', () => {
    const { q } = setup('congreso');
    const rows = (q.playground_coalitions as any).rows;
    expect([1, 2, 3].map(column => rows.filter((r: any) => r.column === column).map((r: any) => r.slot))).toEqual([
      ['iu', 'podemos', 'compromis', 'mes', 'bng', 'ciu', 'erc', 'cup', 'fr'],
      ['pnv', 'amaiur', 'gbai', 'cc', 'mpais', 'prc', 'te', 'unio', 'pdcat'],
      ['psoe', 'psc', 'pp', 'cs', 'upn', 'fac', 'upyd', 'vox'],
    ]);
    expect(rows.filter((r: any) => r.isJoint).map((r: any) => [r.id, r.position])).toEqual([['up', 1], ['jxsi', 6]]);
    expect(rows.some((r: any) => r.id === 'si' || r.id === 'sumar')).toBe(false);
    expect(slot(q, 'mpais').hidden).toBe(true);
    expect(slot(q, 'unio').hidden).toBe(true);
    expect(slot(q, 'pdcat').hidden).toBe(true);
    for (const name of ['iu', 'podemos', 'ciu']) expect(slot(q, name).canJoin).toBe(true);
    for (const name of ['bng', 'compromis', 'mes', 'erc', 'cup']) expect(slot(q, name).canJoin).toBe(true);
  });

  it.each([[2, 0, 'upn'], [0, 2, 'nsuma'], [1, 1, 'upn']] as const)(
    'combines UPN %i and NA+ %i seats in one switchable draft slot', (upnSeats, naSeats, initialId) => {
      const { a, q } = setup('congreso');
      Object.assign(q, { psoe_congreso_s: 178, upn_congreso_s: upnSeats, nsuma_congreso_s: naSeats,
        playground_coalitions: null });
      a.goToScene('coalition_builder.congreso');
      expect(slot(q, 'upn')).toMatchObject({ id: initialId, seats: 2, variants: ['upn', 'nsuma'], column: 3, position: 5 });
      expect(slot(q, 'nsuma')).toBeUndefined();
      const total = q.playground_coalitions_set_seats;
      for (const direction of [1, -1, 1]) {
        gameLib.switchCoalitionBuilderParty(q, 'upn', direction);
        expect(slot(q, 'upn').seats).toBe(2);
        expect(q.playground_coalitions_set_seats).toBe(total);
        expect(q.upn_congreso_s).toBe(upnSeats);
        expect(q.nsuma_congreso_s).toBe(naSeats);
      }
      const selectedId = slot(q, 'upn').id;
      const frame = a.refreshChoices();
      a.choose(frame.choices.findIndex(c => c.id.endsWith('run_congreso')));
      expect(q[`${selectedId}_congreso_s`]).toBe(2);
      expect(q[`${selectedId === 'upn' ? 'nsuma' : 'upn'}_congreso_s`]).toBe(0);
    });

  it('automatically includes IU and Podemos and restores optional contributions without changing totals', () => {
    const { q } = setup('congreso');
    gameLib.editCoalitionBuilder(q, 'iu', 10);
    gameLib.editCoalitionBuilder(q, 'podemos', 20);
    gameLib.editCoalitionBuilder(q, 'bng', 3);
    gameLib.editCoalitionBuilder(q, 'compromis', 4);
    gameLib.editCoalitionBuilder(q, 'mes', 2);
    const total = q.playground_coalitions_set_seats;
    gameLib.toggleCoalitionBuilderMember(q, 'bng');
    expect(slot(q, 'up').seats).toBe(33);
    expect(slot(q, 'mpais').hidden).toBe(false);
    for (const name of ['iu', 'podemos']) {
      expect(slot(q, name)).toMatchObject({ seats: 0, member: true });
      expect(gameLib.editCoalitionBuilder(q, name, 1)).toBe(false);
    }
    expect(gameLib.toggleCoalitionBuilderMember(q, 'iu')).toBe(true);
    expect(slot(q, 'up')).toMatchObject({ seats: 0, disabled: true });
    expect(slot(q, 'podemos')).toMatchObject({ seats: 20, member: false });
    expect(slot(q, 'up').seats).toBe(0);
    expect(slot(q, 'bng').seats).toBe(3);
    expect(slot(q, 'compromis').seats).toBe(4);
    expect(slot(q, 'mes').seats).toBe(2);
    expect(slot(q, 'mpais').hidden).toBe(true);
    expect(q.playground_coalitions_set_seats).toBe(total);
  });

  it('adds IU automatically when Podemos is the first UP member selected', () => {
    const { q } = setup('congreso');
    expect(gameLib.toggleCoalitionBuilderMember(q, 'podemos')).toBe(true);
    expect(slot(q, 'up')).toMatchObject({ seats: 0 });
    expect(slot(q, 'podemos')).toMatchObject({ seats: 0, member: true });
    expect(slot(q, 'iu')).toMatchObject({ seats: 0, member: true });
    gameLib.editCoalitionBuilder(q, 'up', 20);
    expect(slot(q, 'up')).toMatchObject({ seats: 20 });
  });

  it('adds IU and Podemos when a regional UP party is selected, and keeps them after removal', () => {
    const { q } = setup('congreso');
    for (const name of ['bng', 'compromis', 'mes']) {
      expect(gameLib.toggleCoalitionBuilderMember(q, name)).toBe(true);
      expect(slot(q, 'iu').member).toBe(true);
      expect(slot(q, 'podemos').member).toBe(true);
      expect(gameLib.toggleCoalitionBuilderMember(q, name)).toBe(true);
      expect(slot(q, name).member).toBe(false);
      expect(slot(q, 'iu').member).toBe(true);
      expect(slot(q, 'podemos').member).toBe(true);
    }
  });

  it('clears the entire UP coalition when IU is removed, then allows the group to be rebuilt', () => {
    const { q } = setup('congreso');
    gameLib.editCoalitionBuilder(q, 'iu', 10);
    gameLib.editCoalitionBuilder(q, 'podemos', 20);
    gameLib.editCoalitionBuilder(q, 'up', 5);
    expect(slot(q, 'iu').member).toBe(true);
    expect(slot(q, 'podemos').member).toBe(true);
    expect(slot(q, 'up').seats).toBe(35);
    const total = q.playground_coalitions_set_seats;
    expect(gameLib.toggleCoalitionBuilderMember(q, 'iu')).toBe(true);
    expect(slot(q, 'up')).toMatchObject({ seats: 0, disabled: true });
    expect(slot(q, 'iu')).toMatchObject({ seats: 10, member: false });
    expect(slot(q, 'podemos')).toMatchObject({ seats: 20, member: false });
    expect(q.playground_coalitions_set_seats).toBe(total);
    gameLib.editCoalitionBuilder(q, 'up', 4);
    expect(slot(q, 'up').disabled).toBe(false);
    expect(slot(q, 'iu').member).toBe(true);
    expect(slot(q, 'podemos').member).toBe(true);
  });

  it('clears the entire JxSí/JxCat coalition when CiU is removed', () => {
    const { q } = setup('parlament');
    gameLib.editCoalitionBuilder(q, 'erc', 10);
    gameLib.editCoalitionBuilder(q, 'cup', 5);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    gameLib.toggleCoalitionBuilderMember(q, 'cup');
    const total = q.playground_coalitions_set_seats;
    expect(gameLib.toggleCoalitionBuilderMember(q, 'ciu')).toBe(true);
    expect(q.playground_coalitions_set_seats).toBe(total);
    expect((q.playground_coalitions as any).rows.find((r: any) => r.slot === 'joint')).toMatchObject({ seats: 0, disabled: true });
    for (const name of ['ciu', 'erc', 'cup']) expect((q.playground_coalitions as any).rows.find((r: any) => r.slot === name).member).toBe(false);
    expect((q.playground_coalitions as any).rows.find((r: any) => r.slot === 'erc').seats).toBe(10);
    expect((q.playground_coalitions as any).rows.find((r: any) => r.slot === 'cup').seats).toBe(5);
  });

  it('adds ERC and CiU when CUP is selected, and keeps them after removal', () => {
    const { q } = setup('parlament');
    expect(gameLib.toggleCoalitionBuilderMember(q, 'cup')).toBe(true);
    expect(slot(q, 'cup').member).toBe(true);
    expect(slot(q, 'erc').member).toBe(true);
    expect(slot(q, 'ciu').member).toBe(true);
    expect(gameLib.toggleCoalitionBuilderMember(q, 'cup')).toBe(true);
    expect(slot(q, 'cup').member).toBe(false);
    expect(slot(q, 'erc').member).toBe(true);
    expect(slot(q, 'ciu').member).toBe(true);
  });

  it('zeroes one list without changing the other and merges newly hidden Más País seats into Podemos', () => {
    const { q } = setup('congreso');
    gameLib.editCoalitionBuilder(q, 'iu', 10);
    gameLib.editCoalitionBuilder(q, 'podemos', 20);
    gameLib.editCoalitionBuilder(q, 'up', 5);
    gameLib.editCoalitionBuilder(q, 'mpais', 3);
    gameLib.editCoalitionBuilder(q, 'ciu', 12);
    gameLib.editCoalitionBuilder(q, 'erc', 8);
    gameLib.editCoalitionBuilder(q, 'cup', 2);
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    gameLib.toggleCoalitionBuilderMember(q, 'cup');
    expect(slot(q, 'joint').seats).toBe(22);
    expect(slot(q, 'up').seats).toBe(35);
    const total = Number(q.playground_coalitions_set_seats);
    gameLib.editCoalitionBuilder(q, 'up', 0);
    expect(q.playground_coalitions_set_seats).toBe(total - 35);
    expect(slot(q, 'joint').seats).toBe(22);
    expect(slot(q, 'erc').member).toBe(true);
    expect(slot(q, 'cup').member).toBe(true);
    expect(slot(q, 'podemos')).toMatchObject({ seats: 3, member: false });
    expect(slot(q, 'mpais')).toMatchObject({ seats: 0, hidden: true });
    gameLib.editCoalitionBuilder(q, 'up', 4);
    expect(slot(q, 'up').seats).toBe(7);
    gameLib.editCoalitionBuilder(q, 'jxsi', 0);
    expect(slot(q, 'iu').member).toBe(true);
    expect(slot(q, 'podemos').member).toBe(true);
    expect(slot(q, 'up').seats).toBe(7);
    expect(slot(q, 'ciu').member).toBe(false);
    expect(slot(q, 'erc').member).toBe(false);
    expect(slot(q, 'cup').member).toBe(false);
  });

  it('cycles Amaiur/EH Bildu and gates UDC and PDeCAT independently of joint membership', () => {
    const { q } = setup('congreso');
    gameLib.editCoalitionBuilder(q, 'amaiur', 7);
    gameLib.switchCoalitionBuilderParty(q, 'amaiur', 1);
    expect(slot(q, 'amaiur')).toMatchObject({ id: 'ehbildu', seats: 7 });
    gameLib.editCoalitionBuilder(q, 'jxsi', 1);
    expect(slot(q, 'unio').hidden).toBe(true);
    gameLib.switchCoalitionBuilderParty(q, 'ciu', 1);
    expect(slot(q, 'unio').hidden).toBe(false);
    gameLib.editCoalitionBuilder(q, 'unio', 3);
    const total = q.playground_coalitions_set_seats;
    gameLib.switchCoalitionBuilderParty(q, 'ciu', -1);
    expect(slot(q, 'unio')).toMatchObject({ hidden: true, seats: 0 });
    expect(slot(q, 'joint').seats).toBe(4);
    expect(q.playground_coalitions_set_seats).toBe(total);
    gameLib.switchCoalitionBuilderParty(q, 'joint', 1);
    expect(slot(q, 'pdcat').hidden).toBe(false);
    gameLib.switchCoalitionBuilderParty(q, 'joint', -1);
    expect(slot(q, 'pdcat').hidden).toBe(true);
    gameLib.switchCoalitionBuilderParty(q, 'ciu', -1);
    expect(slot(q, 'ciu').id).toBe('junts');
    expect(slot(q, 'pdcat').hidden).toBe(false);
  });

  it('commits both lists and flags only on Run, keeps contributions on repeat Run and survives save/load', () => {
    const { a, q } = setup('congreso');
    gameLib.editCoalitionBuilder(q, 'psoe', 160);
    gameLib.editCoalitionBuilder(q, 'iu', 5);
    gameLib.editCoalitionBuilder(q, 'podemos', 5);
    gameLib.editCoalitionBuilder(q, 'bng', 2);
    gameLib.editCoalitionBuilder(q, 'ciu', 3);
    gameLib.editCoalitionBuilder(q, 'erc', 3);
    gameLib.editCoalitionBuilder(q, 'cup', 2);
    const published = JSON.stringify(q.election_simulation_congreso_hemicycle);
    const oldFlags = [q.iu_in_up, q.spa_bng_active, q.jxcat_in_congreso, q.erc_in_jxcat];
    gameLib.toggleCoalitionBuilderMember(q, 'bng');
    gameLib.toggleCoalitionBuilderMember(q, 'erc');
    gameLib.toggleCoalitionBuilderMember(q, 'cup');
    gameLib.switchCoalitionBuilderParty(q, 'joint', 1);
    expect(q.playground_coalitions_set_seats).toBe(350);
    expect(JSON.stringify(q.election_simulation_congreso_hemicycle)).toBe(published);
    expect([q.iu_in_up, q.spa_bng_active, q.jxcat_in_congreso, q.erc_in_jxcat]).toEqual(oldFlags);
    a.importStateJSON(a.exportStateJSON());
    const live = a.qualities;
    const run = () => {
      const frame = a.refreshChoices();
      a.choose(frame.choices.findIndex(c => c.id.endsWith('run_congreso')));
    };
    run();
    expect(live).toMatchObject({ up_congreso_s: 12, iu_congreso_s: 0, podemos_congreso_s: 0, bng_congreso_s: 0,
      iu_in_up: true, spa_bng_active: false, jxcat_congreso_s: 8, jxsi_congreso_s: 0,
      erc_congreso_s: 0, cup_congreso_s: 0, ciu_congreso_s: 0,
      jxcat_in_congreso: true, jxsi_in_congreso: false, erc_in_jxcat: true, cup_in_jxcat: true });
    run();
    gameLib.toggleCoalitionBuilderMember(live, 'bng');
    gameLib.toggleCoalitionBuilderMember(live, 'erc');
    expect(slot(live, 'bng').seats).toBe(2);
    expect(slot(live, 'erc').seats).toBe(3);
    expect(slot(live, 'up').seats).toBe(10);
    expect(slot(live, 'joint').seats).toBe(5);
  });

  it('initializes existing group memberships and retains an excluded Sumar seat total on Run', () => {
    const { a, q } = setup('congreso');
    Object.assign(q, { psoe_congreso_s: 140, up_congreso_s: 20, jxcat_congreso_s: 10, sumar_congreso_s: 10,
      spa_bng_active: false, spa_compromis_active: false, spa_mes_active: true, erc_in_jxcat: true,
      playground_coalitions: null });
    a.goToScene('coalition_builder.congreso');
    expect(slot(q, 'up').seats).toBe(20);
    expect(slot(q, 'bng').member).toBe(true);
    expect(slot(q, 'compromis').member).toBe(true);
    expect(slot(q, 'mes').member).toBe(false);
    expect(slot(q, 'joint')).toMatchObject({ id: 'jxcat', seats: 10 });
    expect(slot(q, 'erc').member).toBe(true);
    expect(slot(q, 'sumar')).toBeUndefined();
    const frame = a.refreshChoices();
    a.choose(frame.choices.findIndex(c => c.id.endsWith('run_congreso')));
    expect(q.sumar_congreso_s).toBe(10);
    expect(q.playground_coalitions_set_seats).toBe(350);
  });

  it('renders and edits two independent boxes and member lists in Desk', async () => {
    const { a } = setup('congreso');
    const store = useGameStore();
    store.initFromText(gameText);
    store.restoreState(a.exportStateJSON());
    const wrapper = mount(WidgetHost, { global: { plugins: [i18n] }, props: { name: 'builder-playground', props: { configFrom: 'playground_coalitions' } } });
    expect(wrapper.findAll('.builder-checkbox')).toHaveLength(8);
    expect(wrapper.find('[data-party="up"]').exists()).toBe(true);
    expect(wrapper.find('[data-party="jxsi"]').exists()).toBe(true);
    expect(wrapper.find('[data-party="mpais"]').exists()).toBe(false);
    expect(wrapper.find('[data-party="nsuma"]').exists()).toBe(false);
    await wrapper.get('[data-party="upn"] .builder-controls input').setValue('2');
    await wrapper.get('[data-party="upn"] .builder-next').trigger('click');
    expect((wrapper.get('[data-party="nsuma"] .builder-controls input').element as HTMLInputElement).value).toBe('2');
    expect(wrapper.find('[data-party="upn"]').exists()).toBe(false);
    await wrapper.get('[data-party="nsuma"] .builder-previous').trigger('click');
    expect(wrapper.find('[data-party="upn"]').exists()).toBe(true);
    // Zero-seat selections still give summaries the full member span.
    await wrapper.get('[data-party="iu"] .builder-checkbox').setValue(true);
    await wrapper.get('[data-party="mes"] .builder-checkbox').setValue(true);
    expect((wrapper.get('[data-party="up"]').element as HTMLElement).style.gridRow).toBe('1 / 5');
    await wrapper.get('[data-party="iu"] .builder-checkbox').setValue(false);
    await wrapper.get('[data-party="cup"] .builder-checkbox').setValue(true);
    expect((wrapper.get('[data-party="jxsi"]').element as HTMLElement).style.gridRow).toBe('6 / 9');
    await wrapper.get('[data-party="ciu"] .builder-checkbox').setValue(false);
    await wrapper.get('[data-party="iu"] .builder-controls input').setValue('10');
    await wrapper.get('[data-party="podemos"] .builder-controls input').setValue('20');
    await wrapper.get('[data-party="up"] .builder-controls input').setValue('1');
    expect(wrapper.find('[data-party="mpais"]').exists()).toBe(true);
    expect(wrapper.findAll('[data-party="up"] .builder-members > div').map(e => e.text())).toEqual(['IU', 'Podemos']);
    expect((wrapper.get('.builder-brace[data-group="up"]').element as HTMLElement).style.gridRow).toBe('1 / 3');
    await wrapper.get('[data-party="mes"] .builder-checkbox').setValue(true);
    expect((wrapper.get('.builder-brace[data-group="up"]').element as HTMLElement).style.gridRow).toBe('1 / 5');
    await wrapper.get('[data-party="erc"] .builder-controls input').setValue('8');
    await wrapper.get('[data-party="erc"] .builder-checkbox').setValue(true);
    await wrapper.get('[data-party="cup"] .builder-checkbox').setValue(true);
    expect(wrapper.findAll('.builder-brace')).toHaveLength(2);
    expect((wrapper.get('.builder-brace[data-group="joint"]').element as HTMLElement).style.gridRow).toBe('6 / 9');
    expect(wrapper.findAll('[data-party="jxsi"] .builder-members > div').map(e => e.text())).toEqual(['CiU', 'ERC', 'CUP']);
    expect(wrapper.findAll('[data-party="up"] .builder-members > div').map(e => e.text())).toEqual(['IU', 'Podemos', 'MÉS']);
    expect(wrapper.find('[data-party="unio"]').exists()).toBe(false);
    await wrapper.get('[data-party="jxsi"] .builder-controls input').setValue('0');
    expect(wrapper.findAll('.builder-brace')).toHaveLength(1);
    expect(wrapper.find('[data-party="iu"]').classes()).toContain('is-muted');
    wrapper.unmount();
  });

  it('renders and edits two independent boxes and member lists in Classic', () => {
    const { a, q } = setup('congreso');
    (window as any).dendryUI = { dendryEngine: a.engine };
    (window as any).applyWholesome = (name: string) => name;
    window.eval(classic);
    document.body.innerHTML = `<div data-widget="builder-playground" data-props='{"configFrom":"playground_coalitions"}'></div>`;
    (window as any).mountWidgets(document.body, q);
    function edit(id: string, value: number) {
      const input = document.querySelector(`[data-party="${id}"] .builder-controls input`) as HTMLInputElement;
      input.value = String(value);
      input.dispatchEvent(new Event('change'));
    }
    function join(id: string) { (document.querySelector(`[data-party="${id}"] .builder-checkbox`) as HTMLInputElement).click(); }
    expect(document.querySelectorAll('.builder-checkbox')).toHaveLength(8);
    expect(document.querySelector('[data-party="nsuma"]')).toBeNull();
    edit('upn', 2);
    (document.querySelector('[data-party="upn"] .builder-next') as HTMLButtonElement).click();
    expect((document.querySelector('[data-party="nsuma"] .builder-controls input') as HTMLInputElement).value).toBe('2');
    expect(document.querySelector('[data-party="upn"]')).toBeNull();
    (document.querySelector('[data-party="nsuma"] .builder-previous') as HTMLButtonElement).click();
    expect(document.querySelector('[data-party="upn"]')).not.toBeNull();
    // Zero-seat selections still give summaries the full member span.
    join('iu');
    join('mes');
    expect((document.querySelector('[data-party="up"]') as HTMLElement).style.gridRow).toBe('1 / 5');
    expect(document.querySelector('[data-party="up"]')!.classList.contains('is-joint')).toBe(true);
    join('iu');
    join('cup');
    expect((document.querySelector('[data-party="jxsi"]') as HTMLElement).style.gridRow).toBe('6 / 9');
    join('ciu');
    edit('iu', 10);
    edit('podemos', 20);
    edit('up', 1);
    expect(document.querySelector('[data-party="mpais"]')).not.toBeNull();
    expect((document.querySelector('.builder-brace[data-group="up"]') as HTMLElement).style.gridRow).toBe('1 / 3');
    join('mes');
    expect((document.querySelector('.builder-brace[data-group="up"]') as HTMLElement).style.gridRow).toBe('1 / 5');
    edit('erc', 8);
    join('erc');
    join('cup');
    expect(document.querySelectorAll('.builder-brace')).toHaveLength(2);
    expect((document.querySelector('.builder-brace[data-group="joint"]') as HTMLElement).style.gridRow).toBe('6 / 9');
    expect(Array.from(document.querySelectorAll('[data-party="jxsi"] .builder-members > div'), e => e.textContent)).toEqual(['CiU', 'ERC', 'CUP']);
    expect(Array.from(document.querySelectorAll('[data-party="up"] .builder-members > div'), e => e.textContent)).toEqual(['IU', 'Podemos', 'MES']);
    expect(document.querySelector('[data-party="unio"]')).toBeNull();
    edit('jxsi', 0);
    expect(document.querySelectorAll('.builder-brace')).toHaveLength(1);
    expect(document.querySelector('[data-party="iu"]')!.classList.contains('is-muted')).toBe(true);
  });
});

describe.each(['parlament', 'congreso'] as const)('%s coalition builder', chamber => {
  it('updates the Run gate and subtitle while freezing published seats and both charts until Run', () => {
    const { a, q, first, second } = setup(chamber);
    const initial = Number(q[`${first}_${chamber}_s`]);
    const hemicycle = JSON.stringify(q[`election_simulation_${chamber}_hemicycle`]);
    const coalition = JSON.stringify(q[`${chamber}_coalition_config`]);
    const coalitionSeats = (state: Record<string, unknown>) => (state[`${chamber}_coalition_config`] as any).coalitions.map((c: any) => state[c.seatsVar]);
    const initialCoalitionSeats = coalitionSeats(q);
    const run = () => a.currentFrame().choices.find(c => c.id.endsWith(`run_${chamber}`))!;
    expect(run().canChoose).toBe(true);
    expect(gameLib.editCoalitionBuilder(q, first, initial - 1)).toBe(true);
    a.refreshChoices();
    expect(run().canChoose).toBe(false);
    expect(JSON.stringify(run().subtitle)).toContain(String(Number(q.playground_coalitions_needed_seats) - 1));
    expect(q[`${first}_${chamber}_s`]).toBe(initial);
    expect(JSON.stringify(q[`election_simulation_${chamber}_hemicycle`])).toBe(hemicycle);
    expect(JSON.stringify(q[`${chamber}_coalition_config`])).toBe(coalition);
    // A real save/load preserves the draft and its disabled Run choice.
    a.importStateJSON(a.exportStateJSON());
    expect(run().canChoose).toBe(false);
    expect((a.qualities.playground_coalitions as any).rows.find((r: any) => r.id === first).seats).toBe(initial - 1);
    const live = a.qualities;
    gameLib.editCoalitionBuilder(live, second, Number(live[`${second}_${chamber}_s`]) + 1);
    const frame = a.refreshChoices();
    expect(run().canChoose).toBe(true);
    a.choose(frame.choices.findIndex(c => c.id.endsWith(`run_${chamber}`)));
    expect(live[`${first}_${chamber}_s`]).toBe(initial - 1);
    expect(JSON.stringify(live[`election_simulation_${chamber}_hemicycle`])).not.toBe(hemicycle);
    expect(coalitionSeats(live)).not.toEqual(initialCoalitionSeats);
  });

  it('rejects invalid input and keeps a party editable at zero', () => {
    const { q, first } = setup(chamber);
    const total = q.playground_coalitions_set_seats;
    for (const value of [-1, .5, NaN, Infinity, 9999]) {
      expect(gameLib.editCoalitionBuilder(q, first, value)).toBe(false);
    }
    expect(q.playground_coalitions_set_seats).toBe(total);
    expect(gameLib.editCoalitionBuilder(q, first, 0)).toBe(true);
    expect(gameLib.editCoalitionBuilder(q, first, 1)).toBe(true);
  });

  it('includes active zero-seat parties and independently seated DL/PSC', () => {
    const { a, q, first } = setup(chamber);
    const zero = chamber === 'parlament' ? 'si' : 'fr';
    const separate = chamber === 'parlament' ? 'dl' : 'psc';
    q[`${zero}_${chamber}_s`] = 0;
    q[`${zero}_${chamber}_pv`] = 1;
    q[`${separate}_${chamber}_s`] = 1;
    q[`${first}_${chamber}_s`] = Number(q[`${first}_${chamber}_s`]) - 1;
    q.playground_coalitions = null;
    a.goToScene(`coalition_builder.${chamber}`);
    const rows = (q.playground_coalitions as any).rows;
    expect(rows.find((r: any) => r.id === zero).seats).toBe(0);
    expect(rows.find((r: any) => r.id === separate).seats).toBe(1);
    expect(gameLib.editCoalitionBuilder(q, zero, 1)).toBe(true);
    expect(a.refreshChoices().choices.find(c => c.id.endsWith(`run_${chamber}`))!.canChoose).toBe(false);
  });

  it('edits through the Desk widget and updates the visible choices', async () => {
    const { a, first } = setup(chamber);
    const store = useGameStore();
    store.initFromText(gameText);
    store.restoreState(a.exportStateJSON());
    const wrapper = mount(WidgetHost, { global: { plugins: [i18n] }, props: { name: 'builder-playground', props: { configFrom: 'playground_coalitions' } } });
    expect(wrapper.find('input[type="number"]').exists()).toBe(false);
    expect(wrapper.find(`[data-party="${first}"] [data-term]`).exists()).toBe(true);
    const initial = Number(store.q[`${first}_${chamber}_s`]);
    await wrapper.get(`[data-party="${first}"] .builder-controls button`).trigger('click');
    expect((wrapper.get(`[data-party="${first}"] .builder-controls input`).element as HTMLInputElement).value).toBe(String(initial - 1));
    expect(store.frame!.choices.find(c => c.id.endsWith(`run_${chamber}`))!.canChoose).toBe(false);
    expect(store.q[`${first}_${chamber}_s`]).toBe(initial);
    await wrapper.get(`[data-party="${first}"] .builder-controls input`).setValue(String(initial));
    expect(store.frame!.choices.find(c => c.id.endsWith(`run_${chamber}`))!.canChoose).toBe(true);
    await wrapper.get(`[data-party="${first}"] .builder-controls input`).setValue('-1');
    expect((wrapper.get(`[data-party="${first}"] .builder-controls input`).element as HTMLInputElement).value).toBe(String(initial));
    if (chamber === 'parlament') {
      expect(wrapper.get('[data-party="jxsi"]').classes()).toContain('is-muted');
      expect(wrapper.findAll('input[type="checkbox"]')).toHaveLength(3);
      expect(wrapper.find('[data-party="ciu"] input[type="checkbox"]').exists()).toBe(true);
      expect(wrapper.find('[data-party="si"] input[type="checkbox"]').exists()).toBe(false);
      await wrapper.get('[data-party="erc"] .builder-controls input').setValue('0');
      expect(wrapper.get('[data-party="erc"]').classes()).toContain('is-muted');
      await wrapper.get('[data-party="erc"] .builder-controls input').setValue(String(initial));
      expect(wrapper.get('[data-party="erc"]').classes()).not.toContain('is-muted');
      await wrapper.get('[data-party="erc"] input[type="checkbox"]').setValue(true);
      expect(wrapper.find('[data-party="unio"]').exists()).toBe(false);
      expect(wrapper.find('.has-joint').exists()).toBe(true);
      expect(wrapper.findAll('[data-party="jxsi"] .builder-members > div').map(member => member.text())).toEqual(['CiU', 'ERC']);
      expect((wrapper.get('.builder-brace').element as HTMLElement).style.gridRow).toBe('1 / 4');
      expect(wrapper.get('[data-party="si"]').classes()).toContain('is-muted');
      expect((wrapper.get('[data-party="si"] .builder-controls input').element as HTMLInputElement).disabled).toBe(true);
      await wrapper.get('[data-party="cup"] input[type="checkbox"]').setValue(true);
      expect(wrapper.findAll('[data-party="jxsi"] .builder-members > div').map(member => member.text())).toEqual(['CiU', 'ERC', 'CUP']);
      expect((wrapper.get('.builder-brace').element as HTMLElement).style.gridRow).toBe('1 / 5');
      await wrapper.get('[data-party="cup"] input[type="checkbox"]').setValue(false);
      await wrapper.get('[data-party="jxsi"] button[aria-label="Next party for JxSí"]').trigger('click');
      expect(wrapper.find('[data-party="jxcat"]').exists()).toBe(true);
      await wrapper.get('[data-party="erc"] input[type="checkbox"]').setValue(false);
      expect(wrapper.find('.has-joint').exists()).toBe(true);
      expect(wrapper.find('[data-party="jxcat"]').exists()).toBe(true);
      expect(wrapper.find('.builder-brace').exists()).toBe(false);
    }
    wrapper.unmount();
  });

  it('edits through the classic widget without replaying scene actions or remounting charts', () => {
    const { a, q, first } = setup(chamber);
    (window as any).dendryUI = { dendryEngine: a.engine };
    const wholesome = vi.fn((name: string) => `<span data-term="${name.toLowerCase()}">${name}</span>`);
    (window as any).applyWholesome = wholesome;
    window.eval(classic);
    document.body.innerHTML = `<main><div data-widget="builder-playground" data-props='{"configFrom":"playground_coalitions"}'></div></main>`;
    const mountWidgets = (window as any).mountWidgets;
    mountWidgets(document.body, q);
    mountWidgets(document.body, q);
    expect(document.querySelector('input[type="number"]')).toBeNull();
    expect(wholesome).toHaveBeenCalledWith(chamber === 'parlament' ? 'CUP' : 'PSOE');
    const arrival = vi.spyOn(a.engine, 'goToScene');
    const initial = Number(q[`${first}_${chamber}_s`]);
    (document.querySelector(`[data-party="${first}"] .builder-controls button`) as HTMLButtonElement).click();
    expect(q.playground_coalitions_set_seats).toBe(Number(q.playground_coalitions_needed_seats) - 1);
    expect((document.querySelector(`[data-party="${first}"] .builder-controls input`) as HTMLInputElement).value).toBe(String(initial - 1));
    expect(a.currentFrame().choices.find(c => c.id.endsWith(`run_${chamber}`))!.canChoose).toBe(false);
    expect(q[`${first}_${chamber}_s`]).toBe(initial);
    expect(arrival).not.toHaveBeenCalled();
    const input = document.querySelector(`[data-party="${first}"] .builder-controls input`) as HTMLInputElement;
    input.value = String(initial);
    input.dispatchEvent(new Event('change'));
    expect(a.currentFrame().choices.find(c => c.id.endsWith(`run_${chamber}`))!.canChoose).toBe(true);
    input.value = '-1';
    input.dispatchEvent(new Event('change'));
    expect(input.value).toBe(String(initial));
    if (chamber === 'parlament') {
      expect(document.querySelector('[data-party="jxsi"]')!.classList.contains('is-muted')).toBe(true);
      expect(document.querySelectorAll('input[type="checkbox"]')).toHaveLength(3);
      expect(document.querySelector('[data-party="ciu"] input[type="checkbox"]')).not.toBeNull();
      expect(document.querySelector('[data-party="si"] input[type="checkbox"]')).toBeNull();
      input.value = '0';
      input.dispatchEvent(new Event('change'));
      expect(document.querySelector('[data-party="erc"]')!.classList.contains('is-muted')).toBe(true);
      input.value = String(initial);
      input.dispatchEvent(new Event('change'));
      expect(document.querySelector('[data-party="erc"]')!.classList.contains('is-muted')).toBe(false);
      (document.querySelector('[data-party="erc"] input[type="checkbox"]') as HTMLInputElement).click();
      expect(document.querySelector('[data-party="unio"]')).toBeNull();
      expect(document.querySelector('[data-party="jxsi"]')).not.toBeNull();
      expect((document.querySelector('[data-party="jxsi"] .builder-controls input') as HTMLInputElement).value).toBe('70');
      expect(Array.from(document.querySelectorAll('[data-party="jxsi"] .builder-members > div'), member => member.textContent)).toEqual(['CiU', 'ERC']);
      expect((document.querySelector('.builder-brace') as HTMLElement).style.gridRow).toBe('1 / 4');
      expect((document.querySelector('[data-party="si"] .builder-controls input') as HTMLInputElement).disabled).toBe(true);
      (document.querySelector('[data-party="cup"] input[type="checkbox"]') as HTMLInputElement).click();
      expect(Array.from(document.querySelectorAll('[data-party="jxsi"] .builder-members > div'), member => member.textContent)).toEqual(['CiU', 'ERC', 'CUP']);
      expect((document.querySelector('.builder-brace') as HTMLElement).style.gridRow).toBe('1 / 5');
      (document.querySelector('[data-party="cup"] input[type="checkbox"]') as HTMLInputElement).click();
      (document.querySelector('[data-party="jxsi"] button[aria-label="Next party for JxSí"]') as HTMLButtonElement).click();
      expect(document.querySelector('[data-party="jxcat"]')).not.toBeNull();
      (document.querySelector('[data-party="erc"] input[type="checkbox"]') as HTMLInputElement).click();
      expect(document.querySelector('[data-party="jxcat"]')).not.toBeNull();
      expect(document.querySelector('.builder-brace')).toBeNull();
    }
  });
});
