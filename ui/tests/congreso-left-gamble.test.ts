import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');
type Frame = ReturnType<DendryAdapter['goToScene']>;

// Dendry logs and swallows scene errors; fail any test that triggers one.
let sceneErrors: string[] = [];
beforeEach(() => {
  sceneErrors = [];
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    const message = args.map(String).join(' ');
    if (/Scene (action|predicate|expression) error/.test(message)) sceneErrors.push(message);
  });
});
afterEach(() => {
  vi.restoreAllMocks();
  expect(sceneErrors).toEqual([]);
});

// A June 2016-style deadlock after the failed PSOE–Cs attempt: Sánchez leads,
// the PP can only govern through a PSOE abstention, and the left plus ERC add up.
const barons = {
  psoe_congreso_s: 90, up_congreso_s: 75, erc_congreso_s: 15, pp_congreso_s: 130, cs_congreso_s: 32, pdcat_congreso_s: 8,
  congreso_coalition_psoe_cs_wtf_happened: true,
};
// The December 2015 deadlock: PSOE + Cs fall in the "wtf" band, no revolt yet.
const december2015 = {
  pp_congreso_s: 123, psoe_congreso_s: 90, podemos_congreso_s: 69, cs_congreso_s: 40, erc_congreso_s: 9, dl_congreso_s: 8,
  pnv_congreso_s: 6, iu_congreso_s: 2, ehbildu_congreso_s: 2, cc_congreso_s: 1,
};

function enter(overrides: Record<string, unknown>) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  const seatReset = Object.fromEntries(Object.keys(adapter.qualities)
    .filter((key) => /^[a-z0-9]+_congreso_s$/.test(key)).map((key) => [key, 0]));
  Object.assign(adapter.qualities, seatReset, {
    player_party: 'erc', congreso_s_majority: 176, difficulty: 0,
    psoe_leader: 'Pedro Sánchez', pp_leader: 'Mariano Rajoy', iu_in_up: true,
    erc_roadmap: 2, erc_civic_strength: 0, erc_civic_dissent: 0, erc_in_jxsi: false, erc_in_jxcat: false,
    parlament_current_icv: 'cecp', cecp_leader: 'Xavier Domènech', podemos_ideology: 'Anti-austerity Left, Plurinationalism',
    parlament_current_ciu: 'pdcat', pdcat_roadmap: 3,
    psc_relations: 65, cecp_relations: 65, party_resources: 2,
    // A started game holds a Catalan Congreso abstention stock; the betrayal moves voters into it.
    abstain_congreso_catalunya_support: 30,
    ...overrides,
  });
  return { adapter, q: adapter.qualities as Record<string, any>, frame: adapter.goToScene('congreso_coalition') };
}

const choice = (frame: Frame, id: string) => frame.choices.find((c) => c.id === id);
function choose(adapter: DendryAdapter, frame: Frame, id: string) {
  const index = frame.choices.findIndex((c) => c.id === id && c.canChoose);
  if (index < 0) throw new Error(`${id} not choosable; have ${JSON.stringify(frame.choices.map((c) => [c.id, c.canChoose]))}`);
  return adapter.choose(index);
}
function gamble(overrides: Record<string, unknown>) {
  const run = enter(overrides);
  const offer = choose(run.adapter, run.frame, 'congreso_coalition.left_gamble');
  return { ...run, after: choose(run.adapter, offer, 'congreso_coalition_left_offer.erc_votes') };
}

describe('Congreso left gamble', () => {
  it('lists the gamble among the options at a barons deadlock', () => {
    const { adapter, frame } = enter(barons);
    expect(frame.sceneId).toBe('congreso_coalition.coalition_list');
    expect(choice(frame, 'congreso_coalition.left_gamble')?.canChoose).toBe(true);
    // Declining outright is letting the barons have their way.
    expect(choice(frame, 'congreso_coalition.barons_intervene')?.canChoose).toBe(true);
    expect(choose(adapter, frame, 'congreso_coalition.barons_intervene').sceneId).toBe('congreso_coalition_barons');
    // Letting the talks fall apart inside the gamble ends the same way.
    const other = enter(barons);
    const offer = choose(other.adapter, other.frame, 'congreso_coalition.left_gamble');
    expect(choose(other.adapter, offer, 'congreso_coalition_left_offer.failure').sceneId).toBe('congreso_coalition_barons');
  });

  it.each([
    ['Easy', -1, 2, 1, 55, 55],
    ['Normal', 0, 2, 2, 65, 65],
    ['Hard', 1, 1, 3, 75, 65],
  ])('%s (difficulty %i) wins at roadmap %i, %i resources, PSC %i, Comuns %i',(_, difficulty, roadmap, cost, psc, ally) => {
    const base = { ...barons, difficulty, erc_roadmap: roadmap, party_resources: cost, psc_relations: psc, cecp_relations: ally };
    const won = gamble(base);
    expect(won.q.congreso_left_agreement?.kind).toBe('erc');
    expect(won.q.party_resources).toBe(0);

    expect(gamble({ ...base, psc_relations: psc - 1 }).q.congreso_left_agreement).toBeFalsy();
    expect(gamble({ ...base, cecp_relations: ally - 1 }).q.congreso_left_agreement).toBeFalsy();
    // Without a choosable gamble the deadlock goes straight to the revolt.
    expect(enter({ ...base, erc_roadmap: roadmap + 1 }).frame.sceneId).toBe('congreso_coalition_barons');
    expect(enter({ ...base, party_resources: cost - 1 }).frame.sceneId).toBe('congreso_coalition_barons');
  });

  it('charges the betrayal on success, more heavily with strong civic ties', () => {
    expect(gamble(barons).q.erc_civic_dissent).toBe(20);
    expect(gamble({ ...barons, erc_civic_strength: 15 }).q.erc_civic_dissent).toBe(30);
  });

  it('brings the revolt when the gamble fails at a barons deadlock, costing only resources', () => {
    const { q, after } = gamble({ ...barons, psc_relations: 60 });
    expect(after.sceneId).toBe('congreso_coalition_barons');
    expect(q.psoe_barons_revolted).toBe(true);
    expect(q.party_resources).toBe(0);
    expect(q.erc_civic_dissent).toBe(0);
    expect(q.congreso_left_agreement).toBeFalsy();
  });

  it('lets an ERC inside JxSí only for the Parlament gamble', () => {
    const { q } = gamble({ ...barons, erc_in_jxsi: true, jxsi_in_congreso: false });
    expect(q.congreso_left_agreement?.kind).toBe('erc');
    expect(enter({ ...barons, erc_in_jxsi: true, jxsi_in_congreso: true }).frame.sceneId).toBe('congreso_coalition_barons');
  });

  it("sends a CUP player without the Compromís route straight to the revolt", () => {
    expect(enter({ ...barons, player_party: 'cup' }).frame.sceneId).toBe('congreso_coalition_barons');
  });

  // PSOE + Podemos clear investiture without any Catalan or Basque party.
  const mediation = { ...barons, psoe_congreso_s: 100, up_congreso_s: 80, pp_congreso_s: 120, cs_congreso_s: 30, erc_congreso_s: 12,
    player_party: 'cup', compromis_reached_out_ever: true };
  it('gives a CUP player the Compromís route when it is open', () => {
    const { adapter, q, frame } = enter(mediation);
    const offer = choose(adapter, frame, 'congreso_coalition.left_gamble');
    choose(adapter, offer, 'congreso_coalition_left_offer.compromis');
    expect(q.congreso_left_agreement?.kind).toBe('compromis');
    expect(q.party_resources).toBe(0);
    expect(q.congreso_left_gamble_tried).toBe(true);
  });

  it('holds the Compromís route to the difficulty terms', () => {
    const hard = { ...mediation, difficulty: 1, party_resources: 3, cecp_relations: 65 };
    expect(enter({ ...hard, psc_relations: 74 }).frame.sceneId).toBe('congreso_coalition_barons');
    expect(enter({ ...hard, psc_relations: 75, party_resources: 2 }).frame.sceneId).toBe('congreso_coalition_barons');
    const { adapter, q, frame } = enter({ ...hard, psc_relations: 75 });
    choose(adapter, choose(adapter, frame, 'congreso_coalition.left_gamble'), 'congreso_coalition_left_offer.compromis');
    expect(q.party_resources).toBe(0);
  });

  it('shows the gamble cost in brackets', () => {
    const { adapter, frame } = enter(barons);
    const offer = choose(adapter, frame, 'congreso_coalition.left_gamble');
    expect(choice(offer, 'congreso_coalition_left_offer.erc_votes')?.subtitle).toContain('[-2 resources]');
  });

  it('treats walking away from the gamble as giving it up', () => {
    const { adapter, q, frame } = enter(barons);
    const offer = choose(adapter, frame, 'congreso_coalition.left_gamble');
    expect(choose(adapter, offer, 'congreso_coalition_left_offer.failure').sceneId).toBe('congreso_coalition_barons');
    expect(q.congreso_left_gamble_tried).toBe(true);
    expect(q.party_resources).toBe(2);
  });

  it('falls back to the list after a failed December 2015 gamble and never offers it again', () => {
    expect(choice(enter(december2015).frame, 'congreso_coalition.barons_intervene')).toBeUndefined();
    const { adapter, q, after } = gamble({ ...december2015, psc_relations: 60 });
    // @failure has no page of its own: its go-to returns straight to the list.
    const list = after;
    expect(q.psoe_barons_revolted).toBeFalsy();
    expect(choice(list, 'congreso_coalition_psoe_cs.wtf')?.canChoose).toBe(true);
    expect(choice(list, 'congreso_coalition.left_gamble')).toBeUndefined();

    Object.assign(q, barons);
    expect(choice(adapter.goToScene('congreso_coalition'), 'congreso_coalition.left_gamble')).toBeUndefined();
  });

  it('ignores a left agreement left over from an earlier chamber', () => {
    const stale = { kind: 'erc', members: [], chamber: 99 };
    const { q, after } = gamble({ ...barons, psc_relations: 40, congreso_left_agreement: stale });
    expect(after.sceneId).toBe('congreso_coalition_barons');
    expect(q.psoe_barons_revolted).toBe(true);
  });

  it('offers nothing but the gamble or the revolt at a barons deadlock', () => {
    // PSOE–Cs reaches investiture here, which the revolt page must not offer.
    const { frame } = enter({ ...barons, psoe_congreso_s: 115, cs_congreso_s: 57, pp_congreso_s: 116, up_congreso_s: 40,
      cc_congreso_s: 2, prc_congreso_s: 2, erc_congreso_s: 10, pdcat_congreso_s: 8 });
    expect(frame.choices.filter((c) => c.canChoose).map((c) => c.id).sort())
      .toEqual(['congreso_coalition.barons_intervene', 'congreso_coalition.left_gamble']);
  });

  it('still loads a save parked on the barons page', () => {
    const { adapter } = enter(barons);
    expect(adapter.goToScene('congreso_coalition.barons_intervene').sceneId).toBe('congreso_coalition_barons');
  });

  it('does not let the player force elections instead of gambling', () => {
    // After the 2016 toleration no cabinet forms without the player, which used to unlock forced elections.
    const { q, frame } = enter({ ...barons, congreso_coalition_grand_toleration_happened: true });
    expect(q.congreso_coalition_barons_flag).toBe(1);
    expect(choice(frame, 'congreso_coalition.new_congreso_elections')?.canChoose ?? false).toBe(false);
    expect(frame.choices.filter((c) => c.canChoose).map((c) => c.id).sort())
      .toEqual(['congreso_coalition.barons_intervene', 'congreso_coalition.left_gamble']);
  });

  it('leaves the gamble as the only left route at a barons deadlock', () => {
    const { q, frame } = enter(barons);
    expect(q.congreso_coalition_progressive_reach).toBe(true);
    for (const id of ['congreso_coalition_progressive.progressive', 'congreso_coalition_progressive.psoe_minority', 'congreso_coalition_left.change']) {
      expect(choice(frame, id)?.canChoose ?? false, id).toBe(false);
    }
  });

  it('blames the barons only where they are what blocks an option', () => {
    const { q, frame } = enter({ ...barons, congreso_coalition_grand_toleration_happened: true });
    expect(q.congreso_coalition_force_elections).toBe(true);
    expect(choice(frame, 'congreso_coalition.new_congreso_elections')?.subtitle).toMatch(/barons/i);
    expect(q.congreso_coalition_pp_cs_reach).toBe(false);
    expect(choice(frame, 'congreso_coalition_right.pp_cs')?.subtitle).not.toMatch(/barons/i);
  });

  it('shows the gamble in place of the progressive government, not beside it', () => {
    expect(choice(enter(barons).frame, 'congreso_coalition_progressive.progressive')).toBeUndefined();
    expect(choice(enter(december2015).frame, 'congreso_coalition_progressive.progressive')).toBeUndefined();
    // Once the barons have revolted there is no gamble, and the ordinary option returns.
    const after = enter({ ...barons, psoe_barons_revolted: true }).frame;
    expect(choice(after, 'congreso_coalition.left_gamble')).toBeUndefined();
    expect(choice(after, 'congreso_coalition_progressive.progressive')).toBeDefined();
  });

  it('needs a PSOE leader for the PSOE–Cs investiture', () => {
    const seats = { psoe_congreso_s: 120, cs_congreso_s: 60, pp_congreso_s: 120, up_congreso_s: 50 };
    expect(choice(enter({ ...seats, psoe_leader: '???' }).frame, 'congreso_coalition_psoe_cs.government')?.canChoose).toBe(false);
    expect(choice(enter(seats).frame, 'congreso_coalition_psoe_cs.government')?.canChoose).toBe(true);
  });

  it('resolves an old save parked on the left offer', () => {
    const { adapter, q } = enter(barons);
    q.congreso_left_offer_pending = true;
    const offer = adapter.goToScene('congreso_coalition_left_offer');
    expect(choose(adapter, offer, 'congreso_coalition_left_offer.failure').sceneId).toMatch(/^congreso_coalition/);
  });
});
