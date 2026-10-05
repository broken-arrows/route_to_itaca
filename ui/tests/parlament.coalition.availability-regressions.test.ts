import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const root = resolve(import.meta.dirname, '..', '..');
const gameText = readFileSync(resolve(root, 'out/game.json'), 'utf8');

function enterCoalitionMenu(overrides: Record<string, unknown>) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);

  const seatReset = Object.fromEntries(
    (adapter.qualities.parties as string[]).map((party) => [`${party}_parlament_s`, 0]),
  );
  Object.assign(adapter.qualities, seatReset, overrides);

  return { adapter, frame: adapter.goToScene('parlament_coalition') };
}

function selectableIds(frame: ReturnType<DendryAdapter['goToScene']>) {
  return frame.choices.filter((choice) => choice.canChoose).map((choice) => choice.id);
}

describe('Parlament coalition availability regressions', () => {
  it.each(['erc', 'cup'].flatMap(player =>
    [[16, 23, 'erc'], [23, 16, 'ciu'], [20, 20, 'ciu']].map(
      ([ciu, erc, leader]) => [player, ciu, erc, leader] as const,
    ),
  ))('shows distant CUP support and cabinet routes for %s with Junts %s/ERC %s (%s-led)',
    (player, ciu, erc, leader) => {
      const { adapter, frame } = enterCoalitionMenu({
        player_party: player, parlament_s_majority: 68,
        junts_parlament_s: ciu, erc_parlament_s: erc, cup_parlament_s: 3,
      });
      const q = adapter.qualities;
      const pairKey = leader === 'ciu' ? 'ciu_erc_gov' : 'erc_ciu_gov';
      const supportKey = leader === 'ciu' ? 'ciu_erc_cup_sup' : 'erc_ciu_cup_sup';
      const rows = (q.parlament_coalition_config as {
        coalitions: { seatsVar: string; members: { party: string; type: string }[] }[];
      }).coalitions;

      expect(q[`parlament_coalition_option_${pairKey}_visible`]).toBe(false);
      expect(rows.some(c => c.seatsVar === `parlament_coalition_${leader === 'ciu' ? 'ciu_erc' : 'erc_ciu'}_s`)).toBe(false);
      for (const [role, key] of [['support', supportKey], ['government', 'ciu_erc_cup_gov']]) {
        expect(frame.choices.find(c => c.id ===
          `parlament_coalition_ciu_erc_cup.${player}_player_${leader}_led_${role}`))
          .toMatchObject({ canChoose: false });
        const row = rows.find(c => c.seatsVar === `parlament_coalition_${key}_s`);
        expect(row?.members.at(-1)).toEqual({ party: 'cup', type: role });
        expect(q[`parlament_coalition_${key}_s`]).toBe(Number(ciu) + Number(erc) + 3);
      }
      expect(q.parlament_coalition_sovereignty_triple_available).toBe(false);
    },
  );

  it.each([[16, 23, 'erc'], [23, 16, 'ciu'], [24, 24, 'ciu']])(
    'keeps the pair bypass without CUP and applies near miss with CUP: Junts %s/ERC %s',
    (ciu, erc, leader) => {
      const { adapter, frame } = enterCoalitionMenu({
        player_party: 'erc', parlament_s_majority: 68,
        junts_parlament_s: ciu, erc_parlament_s: erc,
      });
      const pairId = leader === 'ciu' ? 'parlament_coalition_ciu_erc.erc' : 'parlament_coalition_erc_ciu.erc';
      expect(frame.choices.find(c => c.id === pairId)).toMatchObject({ canChoose: false });
      expect(frame.choices.some(c => c.id.startsWith('parlament_coalition_ciu_erc_cup.'))).toBe(false);

      adapter.qualities.cup_parlament_s = 3;
      const withCup = adapter.goToScene('parlament_coalition');
      expect(withCup.choices.some(c => c.id === pairId)).toBe(Number(ciu) + Number(erc) >= 48);
      expect(withCup.choices.filter(c => c.id.startsWith('parlament_coalition_ciu_erc_cup.'))).toHaveLength(2);

      adapter.qualities.cup_parlament_s = 0;
      const withoutCup = adapter.goToScene('parlament_coalition');
      expect(withoutCup.choices.find(c => c.id === pairId)).toMatchObject({ canChoose: false });
      expect(withoutCup.choices.some(c => c.id.startsWith('parlament_coalition_ciu_erc_cup.'))).toBe(false);
    },
  );

  it.each(['icv', 'csqp', 'cecp', 'ecp'])(
    'requires seated %s for the emergency bridge and clears it when those seats disappear', party => {
      const { adapter, frame } = enterCoalitionMenu({
        player_party: 'cup', parlament_s_majority: 68,
        cat_spa_relations: 30, parlament_repeat_election: true, erc_roadmap: 2,
        ciu_parlament_s: 64, ciu_roadmap: 2,
        ppc_parlament_s: 30, cs_parlament_s: 20, psc_parlament_s: 15, vox_parlament_s: 6,
      });
      const emergencyId = 'parlament_coalition_unthinkably_grand.any';
      const assertHidden = (current: typeof frame) => {
        expect(current.choices.some(c => c.id === emergencyId)).toBe(false);
        expect(adapter.qualities.parlament_emergency_psc_icv_available).toBe(false);
        expect(adapter.qualities.parlament_emergency_psc_icv_politically_available).toBe(false);
        expect(adapter.qualities.parlament_coalition_unthinkable_left_s).toBe(0);
        expect(adapter.qualities.parlament_coalition_option_unthinkable_left_visible).toBe(false);
        const rows = (adapter.qualities.parlament_coalition_config as {
          coalitions: { seatsVar: string }[];
        }).coalitions;
        expect(rows.some(c => c.seatsVar === 'parlament_coalition_unthinkable_left_s')).toBe(false);
        expect(selectableIds(current)).toContain('parlament_coalition_unionist.cs_ppc_psc_vox_abst');
      };
      assertHidden(frame);

      adapter.qualities[`${party}_parlament_s`] = 1;
      adapter.qualities.ciu_parlament_s = 63;
      const seated = adapter.goToScene('parlament_coalition');
      expect(seated.choices.find(c => c.id === emergencyId)).toMatchObject({ canChoose: true });
      expect(adapter.qualities.parlament_coalition_option_unthinkable_left_visible).toBe(true);
      expect(adapter.qualities.parlament_coalition_unthinkable_left_s).toBe(72);

      adapter.qualities[`${party}_parlament_s`] = 0;
      adapter.qualities.ciu_parlament_s = 64;
      assertHidden(adapter.goToScene('parlament_coalition'));
    },
  );

  it.each(['icv', 'csqp', 'cecp', 'ecp'].flatMap(party =>
    [18, 47, 48, 67, 68].map(seats => [party, seats] as const),
  ))('requires a near miss to display PSC + %s with %s seats', (party, seats) => {
    const { adapter, frame } = enterCoalitionMenu({
      player_party: 'cup', parlament_s_majority: 68,
      ciu_parlament_s: 135 - seats, psc_parlament_s: 12,
      [`${party}_parlament_s`]: seats - 12,
    });
    const visible = seats >= 48;
    const rows = (adapter.qualities.parlament_coalition_config as {
      coalitions: { seatsVar: string }[];
    }).coalitions;
    const choice = frame.choices.find(c => c.id === 'parlament_coalition_progressist.any');

    expect(rows.some(c => c.seatsVar === 'parlament_coalition_psc_icv_s')).toBe(visible);
    expect(adapter.qualities.parlament_coalition_option_psc_icv_visible).toBe(visible);
    expect(!!choice).toBe(visible);
    expect(choice?.canChoose ?? false).toBe(seats >= 68);
    expect(adapter.qualities.parlament_coalition_psc_icv_s).toBe(seats);
  });

  it('uses the current majority threshold and clears progressive visibility on re-entry', () => {
    const { adapter, frame } = enterCoalitionMenu({
      player_party: 'cup', parlament_s_majority: 51,
      psc_parlament_s: 20, icv_parlament_s: 11,
    });
    expect(frame.choices.find(c => c.id === 'parlament_coalition_progressist.any'))
      .toMatchObject({ canChoose: false });

    adapter.qualities.icv_parlament_s = 10;
    const next = adapter.goToScene('parlament_coalition');
    expect(next.choices.some(c => c.id === 'parlament_coalition_progressist.any')).toBe(false);
    expect(adapter.qualities.parlament_coalition_option_psc_icv_visible).toBe(false);
    expect(adapter.qualities.parlament_coalition_psc_icv_s).toBe(30);
  });

  it('selects exactly the minimum viable unionist package at each broken ladder tier', () => {
    const cases = [
      {
        seats: { cs_parlament_s: 35, ppc_parlament_s: 20, psc_parlament_s: 26 },
        expected: 'parlament_coalition_unionist.cs_ppc_psc_abst',
      },
      {
        seats: { ppc_parlament_s: 35, cs_parlament_s: 20, psc_parlament_s: 26 },
        expected: 'parlament_coalition_unionist.ppc_cs_psc_abst',
      },
      {
        seats: { ppc_parlament_s: 35, cs_parlament_s: 20, psc_parlament_s: 20 },
        expected: 'parlament_coalition_unionist.ppc_cs_psc_sup',
      },
      {
        seats: { ppc_parlament_s: 30, cs_parlament_s: 20, psc_parlament_s: 15, vox_parlament_s: 6 },
        expected: 'parlament_coalition_unionist.cs_ppc_psc_vox_abst',
      },
    ];

    for (const scenario of cases) {
      const { adapter, frame } = enterCoalitionMenu({
        player_party: 'cup',
        parlament_s_majority: 68,
        cat_spa_relations: 60,
        ...scenario.seats,
      });
      const unionistChoices = selectableIds(frame).filter((id) => id.startsWith('parlament_coalition_unionist.'));

      expect(unionistChoices, scenario.expected).toEqual([scenario.expected]);
      expect(adapter.qualities.parlament_coalition_has_working_majority).toBe(true);
      expect(adapter.qualities.parlament_coalition_no_possible_majority).toBe(false);
    }
  });

  it('auto-resolves the extended CiU-family minority with only its required external supporters', () => {
    const { adapter, frame } = enterCoalitionMenu({
      player_party: 'cup',
      year: 2018,
      parlament_s_majority: 68,
      jxcat_parlament_s: 41,
      jxcat_roadmap: 0,
      jxcat_leader: 'Carles Puigdemont',
      ppc_parlament_s: 10,
      ppc_leader: 'Xavier García Albiol',
      ppc_leader_gender: 'm',
      unio_parlament_s: 5,
      pdcat_parlament_s: 12,
      pdcat_split: true,
      cat_spa_relations: 60,
    });

    expect(adapter.qualities.parlament_route_ciu_right_available).toBe(true);
    expect(adapter.qualities.parlament_route_ciu_right_use_pdcat).toBe(true);
    expect(selectableIds(frame)).toContain('parlament_coalition_ciu_minority.ppc_any');

    adapter.goToScene('parlament_coalition_ciu_minority.ppc_any');

    expect(adapter.qualities.cat_coalition).toEqual(['jxcat']);
    expect(adapter.qualities.cat_coalition_support).toEqual(['ppc', 'unio', 'pdcat']);
    expect(adapter.qualities.cat_coalition_abstain).toEqual([]);
    expect(adapter.qualities.speaker_party).toBe('ppc');
    expect(adapter.qualities.position_disp).toContain('Opposition');
  });

  it('gives CiU-family exact ties and exposes refusal only when the player can force the election', () => {
    const tie = enterCoalitionMenu({
      player_party: 'erc',
      parlament_s_majority: 68,
      ciu_parlament_s: 34,
      ciu_roadmap: 1,
      ciu_relations: 55,
      erc_parlament_s: 34,
      erc_relations: 55,
    });

    expect(tie.adapter.qualities.parlament_coalition_ciu_erc_s).toBe(68);
    expect(tie.adapter.qualities.parlament_coalition_erc_ciu_s).toBe(0);
    expect(tie.adapter.qualities.parlament_player_can_force_repeat).toBe(true);
    expect(selectableIds(tie.frame)).toContain('parlament_coalition.refuse_negotiations');

    const bypass = enterCoalitionMenu({
      player_party: 'erc',
      parlament_s_majority: 68,
      erc_parlament_s: 5,
      psc_parlament_s: 40,
      icv_parlament_s: 28,
    });
    expect(bypass.adapter.qualities.parlament_player_can_force_repeat).toBeFalsy();
    expect(selectableIds(bypass.frame)).not.toContain('parlament_coalition.refuse_negotiations');

    const deadlock = enterCoalitionMenu({ player_party: 'cup', parlament_s_majority: 68 });
    expect(deadlock.adapter.qualities.parlament_player_can_force_repeat).toBe(false);
    expect(deadlock.adapter.qualities.parlament_coalition_no_possible_majority).toBe(true);
    expect(selectableIds(deadlock.frame)).toContain('parlament_coalition.parlament_no_majority');
  });
});
