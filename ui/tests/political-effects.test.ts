import { describe, it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';
const lib = gameLib as any;
const game = readFileSync('../out/game.json', 'utf8');
const clone = (x: any) => JSON.parse(JSON.stringify(x));
function setup() {
  const a = DendryAdapter.fromJSONText(game);
  a.beginGame([1,2,3,4]); a.goToScene('root.start');
  const q = a.qualities as any;
  Object.assign(q, { player_party: 'erc', game_random_state: 123, party_resources: 5,
    party_income: 0, party_resources_mult: 1, erc_left_strength: 10, erc_pragmatic_strength: 10 });
  return { a, q };
}
const transfer = { contest:'congreso', from:'erc', to:'abstain', amount:1,
  maxDonorFraction:0.5, constituencies:['catalunya'] };

describe('Persistent political reactions', () => {
  it('applies the immediate response, fades monthly, expires, and conserves transfers', () => {
    const { q } = setup();
    q.erc_congreso_catalunya_support = 20;
    const opening = q.abstain_congreso_catalunya_support;
    lib.applyPoliticalEffect(q, { id:'test', transfers:[transfer], immediate:0.5, monthly:0.1, duration:4 });
    expect(q.erc_congreso_catalunya_support).toBe(19.5);
    expect(q.abstain_congreso_catalunya_support).toBe(opening+0.5);
    let last = Infinity;
    for (let i=0;i<4;i++) {
      const control = clone(q); lib.clearPoliticalEffect(control,'test');
      lib.engineTick(control); lib.engineTick(q);
      const loss = control.erc_congreso_catalunya_support-q.erc_congreso_catalunya_support;
      expect(loss).toBeGreaterThan(0); expect(loss).toBeLessThan(last); last=loss;
      expect(q.abstain_congreso_catalunya_support-control.abstain_congreso_catalunya_support).toBeCloseTo(loss,10);
      expect(q.game_random_state).toBe(control.game_random_state);
    }
    expect(q.political_effects.test).toBeUndefined();
    const restored = clone(q); lib.engineTick(restored); lib.engineTick(q);
    expect(clone(q)).toEqual(restored);
  });

  it.each(['exponential','linear','hump'])('replays %s noise after save/load without consuming gameplay RNG', shape => {
    const { a, q } = setup();
    const rng = q.game_random_state;
    lib.applyPoliticalEffect(q, { id:'noise', transfers:[transfer], immediate:0, duration:6, shape, noise:0.25 });
    expect(q.game_random_state).toBe(rng);
    lib.engineTick(q); lib.engineTick(q);
    const save = a.exportStateJSON();
    const b = DendryAdapter.fromJSONText(game); b.beginGame([4,3,2,1]); b.importStateJSON(save);
    for(let i=0;i<4;i++) { lib.engineTick(q); lib.engineTick(b.qualities); }
    expect(clone(b.qualities)).toEqual(clone(q));
    expect(q.political_effects.noise).toBeUndefined();
  });

  it('rejects invalid definitions before changing state and replaces same-topic tails', () => {
    const { q } = setup(); const before=clone(q);
    expect(()=>lib.applyPoliticalEffect(q,{id:'bad',changes:{party_resources:3}})).toThrow();
    expect(()=>lib.applyPoliticalEffect(q,{id:'bad',transfers:[transfer,{...transfer,amount:-1}]})).toThrow();
    expect(clone(q)).toEqual(before);
    lib.applyPoliticalEffect(q,{id:'topic',transfers:[transfer],immediate:0,monthly:1});
    lib.applyPoliticalEffect(q,{id:'topic',transfers:[transfer],immediate:0,monthly:0.2});
    expect(Object.keys(q.political_effects)).toEqual(['topic']);
    expect(q.political_effects.topic.def.monthly).toBe(0.2);
  });

  it('keeps fundraising income while its reaction expires, and supports cancellation', () => {
    const { a, q }=setup(); const resources=q.party_resources;
    const errors=vi.spyOn(console,'error').mockImplementation(()=>{});
    try {
      let f=a.goToScene('erc_fundraising');
      a.choose(f.choices.findIndex(c=>c.id==='erc_fundraising.raise_rates' && c.canChoose));
      expect(q.party_resources).toBe(resources+1); expect(q.party_income).toBe(1);
      expect(q.political_effects.party_dues.def.transfers.length).toBeGreaterThan(0);
      expect(q.political_effects.party_dues.def.immediate).toBe(1);
      const save=a.exportStateJSON();
      for(let i=0;i<12;i++) lib.engineTick(q);
      expect(q.political_effects.party_dues).toBeUndefined(); expect(q.party_income).toBe(1);
      a.importStateJSON(save);
      a.goToScene('erc_fundraising.lower_rates');
      expect((a.qualities as any).political_effects.party_dues).toBeUndefined();
      expect(errors.mock.calls).toEqual([]);
    } finally { errors.mockRestore(); }
  });

  it('supersedes the subsidy reaction on reversal and guards unavailable direct entry', () => {
    const {a,q}=setup();
    a.goToScene('erc_fundraising.higher_allocation');
    expect(q.party_resources_mult).toBe(2);
    expect(q.political_effects.party_subsidies.def.changes.independence_trust).toBe(-3);
    a.goToScene('erc_fundraising.lower_allocation');
    expect(q.party_resources_mult).toBe(1);
    expect(q.political_effects.party_subsidies.def.changes).toEqual({});
    q.party_income=2; const before=clone(q);
    a.goToScene('erc_fundraising.raise_rates');
    expect(q.party_resources).toBe(before.party_resources);
    expect(q.political_effects).toEqual(before.political_effects);
  });

  it('uses the united dues route and actually lowers dues through the lower option', () => {
    const {a,q}=setup();
    Object.assign(q,{erc_in_jxsi:true,player_in_jxsi:true,jxsi_in_congreso:false,jxsi_in_local:false});
    for(const c of q.parlament_constituencies) for(const d of q.parlament_demographics) {
      q[`jxsi_parlament_${c}_${d}_support`] += q[`erc_parlament_${c}_${d}_support`];
      q[`erc_parlament_${c}_${d}_support`]=0;
    }
    const errors=vi.spyOn(console,'error').mockImplementation(()=>{});
    try {
      a.goToScene('unity_fundraising.raise_rates_erc');
      expect(q.party_income).toBe(1);
      expect(q.political_effects.party_dues.def.transfers.some((t:any)=>t.contest==='parlament' && t.amount>0)).toBe(true);
      q.party_income=2;
      a.goToScene('unity_fundraising.lower_rates_erc');
      expect(q.party_income).toBe(1); expect(q.political_effects.party_dues).toBeUndefined();
      expect(errors.mock.calls).toEqual([]);
    } finally { errors.mockRestore(); }
  });
});
