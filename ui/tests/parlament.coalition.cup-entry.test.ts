import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');

function enter(player: string, leader: string, overrides: Record<string, unknown> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const seatReset = Object.fromEntries(
    (adapter.qualities.parties as string[]).map(party => [`${party}_parlament_s`, 0]),
  );
  Object.assign(adapter.qualities, seatReset, {
    player_party: player,
    year: 2013,
    month: 2,
    time: 6,
    month_actions: 0,
    rubicon: false,
    countdowns: [],
    next_spanish_election_time: 100,
    next_election_time: 100,
    parlament_s_majority: 68,
    ciu_parlament_s: leader === 'ciu' ? 36 : 27,
    erc_parlament_s: leader === 'erc' ? 36 : 27,
    cup_parlament_s: 10,
    ciu_relations: 60,
    erc_relations: 60,
    cup_relations: 60,
    cup_faction: 'Poble Lliure',
    cup_endavant_strength: 30,
    cup_poblelliure_strength: 50,
    cup_thirdsector_strength: 20,
    party_resources: 3,
    ...overrides,
  });
  const frame = adapter.goToScene('parlament_coalition');
  const id = `parlament_coalition_ciu_erc_cup.${player}_player_${leader}_led_government`;
  return { adapter, frame, id };
}

describe('CUP National Concentration entry', () => {
  it.each(['cup', 'erc'].flatMap(player => ['ciu', 'erc'].map(leader => [player, leader])))
    ('deducts the advertised entry resource for player %s in a %s-led cabinet', (player, leader) => {
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      try {
        const { adapter, frame, id } = enter(player, leader);
        errors.mockClear();
        const index = frame.choices.findIndex(choice => choice.id === id);
        expect(index).toBeGreaterThanOrEqual(0);
        expect(frame.choices[index].canChoose).toBe(true);
        const speakerFrame = adapter.choose(index);
        expect(adapter.qualities.party_resources).toBe(2);
        expect(adapter.qualities.parlament_coalition_sovereignty_kind).toBe('government');
        const postSpeaker = adapter.choose(speakerFrame.choices.findIndex(choice => choice.canChoose));
        expect(adapter.qualities.party_resources).toBe(2);
        const newspaper = adapter.choose(postSpeaker.choices.findIndex(choice => choice.canChoose));
        expect(adapter.qualities.party_resources).toBe(2);
        const phaseId = `parlament_coalition_ciu_erc_cup_phase_${player}_player_${leader}_led_government`;
        const phaseIndex = newspaper.choices.findIndex(choice => choice.id === phaseId);
        expect(phaseIndex).toBeGreaterThanOrEqual(0);
        adapter.choose(phaseIndex);
        expect(adapter.qualities.party_resources).toBe(2);
        expect(errors.mock.calls).toEqual([]);
      } finally {
        errors.mockRestore();
      }
    });

  it.each(['ciu', 'erc'].flatMap(leader => [
    [55, 80, false, 0], [80, 55, false, 0],
    [56, 66, true, 1], [66, 56, true, 1],
    [65, 66, true, 1], [66, 65, true, 1], [66, 66, true, 0],
  ].map(([ciu, erc, available, cost]) => [leader, ciu, erc, available, cost] as const)))
    ('checks both partner thresholds for CUP in a %s-led cabinet: CiU %s / ERC %s',
      (leader, ciu, erc, available, cost) => {
        const { adapter, frame, id } = enter('cup', leader, {
          ciu_relations: ciu, erc_relations: erc,
        });
        const choice = frame.choices.find(choice => choice.id === id);
        expect(choice?.canChoose).toBe(available);
        if (available) {
          expect(choice?.subtitle?.includes('[-1 resources]')).toBe(cost === 1);
          expect(choice?.subtitle).not.toContain('Poble Lliure');
          adapter.choose(frame.choices.findIndex(choice => choice.id === id));
          expect(adapter.qualities.party_resources).toBe(3 - Number(cost));
        } else {
          expect(choice?.subtitle).toContain(Number(ciu) <= 55 ? 'ciu' : 'erc');
        }
      });

  it.each(['ciu', 'erc'].flatMap(leader => [0, 1].map(resources => [leader, resources] as const)))
    ('requires the one resource and spends it for CUP in a %s-led cabinet with %s resources',
      (leader, resources) => {
        const { adapter, frame, id } = enter('cup', leader, { party_resources: resources });
        const choice = frame.choices.find(choice => choice.id === id);
        expect(choice?.canChoose).toBe(resources === 1);
        if (resources === 1) {
          adapter.choose(frame.choices.findIndex(choice => choice.id === id));
          expect(adapter.qualities.party_resources).toBe(0);
        } else {
          expect(choice?.subtitle).toContain('Not enough resources');
        }
      });

  it.each(['ciu', 'erc'])('allows free CUP entry without resources in a %s-led cabinet', leader => {
    const { adapter, frame, id } = enter('cup', leader, {
      ciu_relations: 66, erc_relations: 66, party_resources: 0,
    });
    expect(frame.choices.find(choice => choice.id === id)?.canChoose).toBe(true);
    adapter.choose(frame.choices.findIndex(choice => choice.id === id));
    expect(adapter.qualities.party_resources).toBe(0);
  });

  it.each(['ciu', 'erc'].flatMap(leader =>
    ['Poble Lliure', 'Third Sector', 'Endavant'].map(faction => [leader, faction])))
    ('ignores CUP own relations and NPC faction rules for the CUP player: %s-led / %s',
      (leader, faction) => {
        const { adapter, frame, id } = enter('cup', leader, { cup_relations: 0, cup_faction: faction });
        expect(frame.choices.find(choice => choice.id === id)?.canChoose).toBe(true);
        adapter.choose(frame.choices.findIndex(choice => choice.id === id));
        expect(adapter.qualities.party_resources).toBe(2);
      });

  it.each(['ciu', 'erc'].flatMap(leader => [30, 31].map(strength => [leader, strength] as const)))
    ('blocks only cabinet entry for CUP in a %s-led cabinet at Endavant strength %s',
      (leader, strength) => {
        const { adapter, frame, id } = enter('cup', leader, {
          cup_endavant_strength: strength, ciu_relations: 80, erc_relations: 80,
        });
        const choice = frame.choices.find(choice => choice.id === id);
        expect(choice?.canChoose).toBe(strength === 30);
        if (strength > 30) expect(choice?.subtitle).toContain('Endavant');
        const supportId = id.replace('_government', '_support');
        expect(frame.choices.find(choice => choice.id === supportId)?.canChoose).toBe(true);
        adapter.choose(frame.choices.findIndex(choice => choice.id === supportId));
        expect(adapter.qualities.party_resources).toBe(3);
      });

  it.each(['ciu', 'erc'].flatMap(leader => [29, 30, 45].map(strength => [leader, strength] as const)))
    ('checks Third Sector for CUP joining a %s-led cabinet at strength %s', (leader, strength) => {
      const { adapter, frame, id } = enter('cup', leader, {
        cup_thirdsector_strength: strength, ciu_relations: 80, erc_relations: 80,
      });
      const choice = frame.choices.find(choice => choice.id === id);
      expect(choice?.canChoose).toBe(leader === 'erc' || strength < 30);
      if (leader === 'ciu' && strength >= 30) expect(choice?.subtitle).toContain('Third Sector');
      const supportId = id.replace('_government', '_support');
      expect(frame.choices.find(choice => choice.id === supportId)?.canChoose).toBe(true);
      adapter.choose(frame.choices.findIndex(choice => choice.id === supportId));
      expect(adapter.qualities.party_resources).toBe(3);
    });

  it('treats a seat tie as CiU-led for the CUP Third Sector veto', () => {
    const { frame, id } = enter('cup', 'ciu', {
      ciu_parlament_s: 30, erc_parlament_s: 30, cup_thirdsector_strength: 30,
    });
    expect(frame.choices.find(choice => choice.id === id)).toMatchObject({ canChoose: false });
  });

  it.each(['ciu', 'erc'])('keeps ordinary sovereignty support available with every CUP cabinet blocker: %s-led', leader => {
    const { adapter, frame, id } = enter('cup', leader, {
      ciu_relations: 41, erc_relations: 41, cup_relations: 0,
      cup_endavant_strength: 50, cup_thirdsector_strength: 50, party_resources: 0,
    });
    expect(frame.choices.find(choice => choice.id === id)?.canChoose).toBe(false);
    const supportId = id.replace('_government', '_support');
    expect(frame.choices.find(choice => choice.id === supportId)?.canChoose).toBe(true);
    adapter.choose(frame.choices.findIndex(choice => choice.id === supportId));
    expect(adapter.qualities.party_resources).toBe(0);
    expect(adapter.qualities.parlament_coalition_sovereignty_kind).toBe('support');
  });

  it.each([
    ['Poble Lliure', 54, 3, false, 0], ['Poble Lliure', 55, 0, false, 1],
    ['Poble Lliure', 55, 1, true, 1], ['Poble Lliure', 64, 3, true, 1],
    ['Poble Lliure', 65, 0, true, 0], ['Third Sector', 64, 3, false, 0],
    ['Third Sector', 65, 0, true, 0], ['Endavant', 80, 3, false, 0],
  ] as const)('preserves ERC/NPC CUP entry rules: %s at %s with %s resources',
    (faction, relations, resources, available, cost) => {
      const { adapter, frame, id } = enter('erc', 'ciu', {
        cup_faction: faction, cup_relations: relations, party_resources: resources,
        cup_endavant_strength: 50, cup_thirdsector_strength: 50,
      });
      const choice = frame.choices.find(choice => choice.id === id);
      expect(choice?.canChoose).toBe(available);
      if (available) {
        expect(choice?.subtitle?.includes('[-1 resources]')).toBe(cost === 1);
        adapter.choose(frame.choices.findIndex(choice => choice.id === id));
        expect(adapter.qualities.party_resources).toBe(resources - cost);
      }
    });

  it.each([55, 56, 66])('uses the live CiU-family relations for CUP: Junts %s', relations => {
    const { adapter, frame, id } = enter('cup', 'ciu', {
      ciu_parlament_s: 0, junts_parlament_s: 36,
      ciu_relations: 0, junts_relations: relations, erc_relations: 66,
    });
    expect(adapter.qualities.parlament_current_ciu).toBe('junts');
    expect(frame.choices.find(choice => choice.id === id)?.canChoose).toBe(relations > 55);
    if (relations > 55) {
      adapter.choose(frame.choices.findIndex(choice => choice.id === id));
      expect(adapter.qualities.party_resources).toBe(relations > 65 ? 3 : 2);
    }
  });

  it('recalculates CUP costs and all blockers on menu re-entry', () => {
    const { adapter, id } = enter('cup', 'ciu', {
      ciu_relations: 55, erc_relations: 55, cup_endavant_strength: 31,
      cup_thirdsector_strength: 30,
    });
    const blocked = adapter.goToScene('parlament_coalition').choices.find(choice => choice.id === id);
    expect(blocked).toMatchObject({ canChoose: false });
    for (const reason of ['ciu', 'erc', 'Endavant', 'Third Sector']) expect(blocked?.subtitle).toContain(reason);

    Object.assign(adapter.qualities, {
      ciu_relations: 56, erc_relations: 66, cup_endavant_strength: 30, cup_thirdsector_strength: 29,
    });
    const paid = adapter.goToScene('parlament_coalition').choices.find(choice => choice.id === id);
    expect(paid).toMatchObject({ canChoose: true });
    expect(paid?.subtitle).toContain('[-1 resources]');

    adapter.qualities.ciu_relations = 66;
    const free = adapter.goToScene('parlament_coalition');
    expect(free.choices.find(choice => choice.id === id)).toMatchObject({ canChoose: true });
    expect(free.choices.find(choice => choice.id === id)?.subtitle).not.toContain('[-1 resources]');
    adapter.choose(free.choices.findIndex(choice => choice.id === id));
    expect(adapter.qualities.party_resources).toBe(3);
  });
});
