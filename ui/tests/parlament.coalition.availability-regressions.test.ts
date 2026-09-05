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
