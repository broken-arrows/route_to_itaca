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
