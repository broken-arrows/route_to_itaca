import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const root = resolve(import.meta.dirname, '..', '..');
const gameText = readFileSync(resolve(root, 'out', 'game.json'), 'utf8');

function enterCoalitionMenu(overrides: Record<string, unknown>) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);

  const seatReset = Object.fromEntries(
    Object.keys(adapter.qualities)
      .filter((key) => /^[a-z0-9]+_congreso_s$/.test(key))
      .map((key) => [key, 0]),
  );
  Object.assign(adapter.qualities, seatReset, {
    player_party: 'erc',
    congreso_s_majority: 176,
    ...overrides,
  });

  return { adapter, frame: adapter.goToScene('congreso_coalition') };
}

function selectableIds(frame: ReturnType<DendryAdapter['goToScene']>) {
  return frame.choices.filter((choice) => choice.canChoose).map((choice) => choice.id);
}

describe('Congreso necessary investiture partners', () => {
  function progressiveMembers(overrides: Record<string, unknown>) {
    const { adapter } = enterCoalitionMenu({
      parlament_current_ciu: 'pdcat',
      pdcat_roadmap: 1,
      erc_in_jxsi: false,
      erc_in_jxcat: false,
      iu_in_up: true,
      psoe_congreso_s: 100,
      up_congreso_s: 50,
      erc_congreso_s: 26,
      pdcat_congreso_s: 8,
      pp_congreso_s: 166,
      ...overrides,
    });
    const config = adapter.qualities.congreso_coalition_config as {
      coalitions: { seatsVar: string; members: { party: string; type: string; escalated?: boolean }[] }[];
    };
    return {
      adapter,
      members: config.coalitions.find((coalition) => coalition.seatsVar === 'congreso_coalition_progressive_s')!.members,
    };
  }

  it('drops PDeCAT abstention after negotiated ERC support supplies 176 yes votes', () => {
    const { adapter, members } = progressiveMembers({});

    expect(members.map((member) => member.party)).toEqual(['psoe', 'up', 'erc']);
    expect(members.find((member) => member.party === 'erc')).toMatchObject({ type: 'support', escalated: true });
    expect(adapter.qualities.congreso_coalition_progressive_reach).toBe(true);
    expect(adapter.qualities.congreso_coalition_progressive_talks).toBe(true);
  });

  it('retains PDeCAT abstention when its no vote would defeat investiture', () => {
    const { adapter, members } = progressiveMembers({
      up_congreso_s: 69,
      erc_congreso_s: 0,
      pdcat_congreso_s: 14,
      pp_congreso_s: 167,
    });

    expect(members.map((member) => member.party)).toEqual(['psoe', 'up', 'pdcat']);
    expect(members.find((member) => member.party === 'pdcat')).toMatchObject({ type: 'abstain' });
    expect(adapter.qualities.congreso_coalition_progressive_reach).toBe(true);
    expect(adapter.qualities.congreso_coalition_progressive_talks).toBe(false);
  });

  it('preserves the governing core even when one cabinet party has its own majority', () => {
    const { members } = progressiveMembers({
      psoe_congreso_s: 176,
      up_congreso_s: 50,
      erc_congreso_s: 0,
      pp_congreso_s: 116,
    });

    expect(members).toEqual([
      { party: 'psoe', type: 'government' },
      { party: 'up', type: 'government' },
    ]);
  });
});

describe('Congreso right-coalition priorities', () => {
  it('chooses a viable PP-Cs cabinet ahead of either Grand Coalition or Grand Toleration', () => {
    const { adapter, frame } = enterCoalitionMenu({
      pp_congreso_s: 130,
      cs_congreso_s: 46,
      psoe_congreso_s: 100,
      pp_leader: 'Mariano Rajoy',
      psoe_leader: 'Susana Díaz',
    });
    const selectable = selectableIds(frame);

    expect(adapter.qualities.congreso_coalition_pp_cs_reach).toBe(true);
    expect(adapter.qualities.congreso_coalition_liberal_right_preferred).toBe(true);
    expect(adapter.qualities.congreso_coalition_grand_preferred).toBe(0);
    expect(selectable).toContain('congreso_coalition_right.pp_cs');
    expect(selectable).not.toContain('congreso_coalition_grand.pp_led');
    expect(selectable).not.toContain('congreso_coalition_grand.toleration');
  });

  it('retains a grand arrangement as the fallback when PP-Cs cannot clear investiture', () => {
    const { adapter, frame } = enterCoalitionMenu({
      pp_congreso_s: 130,
      cs_congreso_s: 20,
      psoe_congreso_s: 80,
      pp_leader: 'Mariano Rajoy',
      psoe_leader: 'Susana Díaz',
    });
    const selectable = selectableIds(frame);

    expect(adapter.qualities.congreso_coalition_pp_cs_reach).toBe(false);
    expect(adapter.qualities.congreso_coalition_liberal_right_preferred).toBe(false);
    expect(adapter.qualities.congreso_coalition_grand_preferred).toBe(1);
    expect(selectable).toContain('congreso_coalition_grand.toleration');
    expect(selectable).not.toContain('congreso_coalition_right.pp_cs');
  });

  it('makes PNV a hard opponent of a Rajoy-led PP, not a negotiable supporter', () => {
    const rajoy = enterCoalitionMenu({
      pp_congreso_s: 150,
      cs_congreso_s: 20,
      pnv_congreso_s: 6,
      pp_leader: 'Mariano Rajoy',
    });
    const successor = enterCoalitionMenu({
      pp_congreso_s: 150,
      cs_congreso_s: 20,
      pnv_congreso_s: 6,
      pp_leader: 'Alberto Núñez Feijóo',
    });

    expect(rajoy.adapter.qualities.congreso_coalition_pp_cs_reach).toBe(false);
    expect(selectableIds(rajoy.frame)).not.toContain('congreso_coalition_right.pp_cs');

    expect(successor.adapter.qualities.congreso_coalition_pp_cs_reach).toBe(true);
    expect(selectableIds(successor.frame)).toContain('congreso_coalition_right.pp_cs');
  });
});
