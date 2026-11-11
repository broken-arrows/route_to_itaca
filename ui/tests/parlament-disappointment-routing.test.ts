import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const key = (p: string, c = 'girona') => `${p}_parlament_${c}_middle_support`;
function state(cupShare = 8) {
  return { parties:['ciu','erc','cup','fnc','ppc','vox'], parlament_constituencies:['girona'],
    parlament_demographics:['middle'], parlament_girona_middle_pop:1000,
    parlament_current_ciu:'ciu', fnc_formed:true, pxc_dissolved:true, vox_active:true,
    independence_movement:85, independence_trust:15,
    parlament_disappointment:{version:1,frustration:1,pressure:1,cells:{}},
    ...Object.fromEntries(Object.entries({ciu:20,erc:15,cup:cupShare/2,fnc:1,ppc:9-cupShare/2,vox:5,abstain:50})
      .map(([p,n])=>[key(p),n])) } as Record<string, any>;
}
const carriers = {icr:'ciu',il:'erc',cup:'cup',fnc:'fnc',ppc:'ppc',vox:'vox',abs:'abstain'};
const profile = gameLib.getParlamentDisappointmentProfile;
const split = gameLib.splitParlamentDisappointmentGain;

describe('CUP saturation and FNC diversion', () => {
  it.each([[6,1],[8,.675],[9,.55],[10,.48]])('uses valid-vote share %s with size factor %s', (share,factor) => {
    const p = profile(state(share));
    expect(p.share).toBeCloseTo(share);
    expect(p.sizeFactor).toBeCloseTo(factor);
  });
  it('approaches a 35% size floor and weights provinces by population, excluding abstention', () => {
    const dominant = state();
    for(const p of dominant.parties) dominant[key(p)] = p === 'cup' ? 50 : 0;
    expect(profile(dominant).sizeFactor).toBeGreaterThan(.35);
    expect(profile(dominant).sizeFactor).toBeLessThan(.351);
    const q = state(6);
    q.parlament_constituencies.push('barcelona');
    q.parlament_barcelona_middle_pop = 3000;
    for(const p of [...q.parties,'abstain']) q[key(p,'barcelona')] = q[key(p)];
    q[key('cup','barcelona')] += 4;
    q[key('ciu','barcelona')] -= 4;
    expect(profile(q).share).toBeCloseTo(12);
  });
  it.each([['cat_coalition',.25],['cat_coalition_support',.6],['cat_coalition_abstain',.85]])(
    'applies %s responsibility separately from the size floor', (role, factor) => {
      const q = state(); q[role] = ['cup'];
      expect(profile(q).credibility).toBe(factor);
      expect(profile(q).cupFactor).toBeCloseTo(.675 * Number(factor));
    });
  it('uses the strongest formal role and removes CUP’s own broken-promise protest bonus until expiry', () => {
    const q = state();
    Object.assign(q,{cat_coalition:['cup'],cat_coalition_support:['cup'],cat_coalition_abstain:['cup']});
    expect(profile(q).credibility).toBe(.25);
    Object.assign(q,{player_party:'cup',broken_roadmap:1});
    expect(profile(q).cupFactor).toBe(0);
    q.broken_roadmap = 0;
    expect(profile(q).credibility).toBe(.25);
  });
  it('distinguishes donor affinity, bounds the breach multiplier and preserves every blocked voter', () => {
    const q = state();
    const ciu = split(q,'icr',100), erc = split(q,'il',100);
    expect(ciu.cup).toBeCloseTo(67.5);
    expect(ciu.fnc).toBeCloseTo(16.25);
    expect(erc.fnc).toBeCloseTo(3.25);
    q.broken_roadmap = 1;
    for(const from of ['icr','il']) {
      const normal = from==='icr' ? ciu : erc;
      const breach = split(q,from,100);
      expect(breach.cup).toBeCloseTo(normal.cup);
      expect(breach.fnc).toBeCloseTo(normal.fnc * 1.5);
      expect(breach.cup+breach.fnc+breach.retained).toBeCloseTo(100);
    }
    q.parlament_disappointment.frustration = .5;
    expect(split(q,'icr',100).fnc).toBeCloseTo(ciu.fnc * 1.5 / 4);
    q.fnc_formed = false;
    expect(split(q,'icr',100).fnc).toBe(0);
  });
  it('treats an actual CUP joint ballot as an establishment donor, ignoring obsolete list flags', () => {
    const q = state();
    q.parties.push('jxsi'); q.cup_in_jxsi = q.erc_in_jxsi = true;
    q[key('jxsi')] = 0;
    expect(profile(q).credibility).toBe(1);
    q[key('jxsi')] = 20;
    expect(profile(q).cupFactor).toBe(0);
    expect(split(q,'icr',100).fnc).toBe(50);
  });
  it('settles only realized CUP/FNC recruitment against the finite capacity and preserves it across saves', () => {
    let q = state();
    const opening = structuredClone(q);
    const p = profile(q);
    const requests = gameLib.buildParlamentDisappointmentTransfers(q, { province:'girona',demographic:'middle',
      carriers, support:Object.fromEntries([...q.parties,'abstain'].map(p=>[p,q[key(p)]])), disappointmentProfile:p });
    const capacity = 35 * .12;
    gameLib.applyParlamentTransfers(q,'girona','middle',{},requests,1,carriers);
    q = JSON.parse(JSON.stringify(q));
    const remaining = Object.values(q.parlament_disappointment.cells['girona.middle'])
      .reduce((s:number,c:any)=>s+c.remaining,0);
    const realized = q[key('cup')]-opening[key('cup')]+q[key('fnc')]-opening[key('fnc')];
    expect(capacity-remaining).toBeCloseTo(realized,10);
    expect(realized).toBeLessThan(capacity * .1);
  });
});

describe('represented FNC competes with VOX recruitment', () => {
  it.each([0,1])('redirects the original donor’s recruitment and preserves outgoing VOX voters with FNC seats=%s', seats => {
    const q = state(); q.fnc_parlament_s = seats;
    const original = structuredClone(q);
    const realized: number[] = [];
    gameLib.applyParlamentTransfers(q,'girona','middle',{},[
      {from:'ppc',to:'vox',amount:2,mechanism:'recruit',onSettled:n=>realized.push(n)},
      {from:'vox',to:'abs',amount:1,mechanism:'departure'},
    ],1,carriers);
    const gain = seats ? 1.4 : 2;
    expect(realized).toEqual([gain]);
    expect(q[key('ppc')]).toBeCloseTo(original[key('ppc')]-2);
    expect(q[key('vox')]).toBeCloseTo(original[key('vox')]+gain-1);
    expect(q[key('fnc')]).toBeCloseTo(original[key('fnc')]+(seats ? .6 : 0));
    expect(q[key('abstain')]).toBe(51);
  });
  it('balances matrix recruitment and fully preserves reverse routes', () => {
    const q = state(); q.fnc_parlament_s = 1;
    gameLib.applyParlamentTransfers(q,'girona','middle',{ppc:-2,vox:2},[],1,carriers);
    expect(q[key('vox')]).toBeCloseTo(6.4);
    expect(q[key('ppc')]).toBeCloseTo(3);
    expect(q[key('fnc')]).toBeCloseTo(1.6);
    gameLib.applyParlamentTransfers(q,'girona','middle',{ppc:2,vox:-2},[],1,carriers);
    expect(q[key('vox')]).toBeCloseTo(4.4);
    expect(q[key('ppc')]).toBeCloseTo(5);
  });
  it('also applies to Parlament card recruitment, with full departures', () => {
    const q = state(); q.fnc_parlament_s = 1;
    const options = {contest:'parlament',from:'ppc',to:'vox',amount:2,maxDonorFraction:.5,
      constituencies:'all',demographics:'all'};
    gameLib.cardSupportTransfer(q,options);
    expect(q[key('vox')]).toBeCloseTo(6.4);
    // Opening independence share is 80%: existing bloc resistance admits 10%
    // of the redirected external recruitment, retaining the rest with PPC.
    expect(q[key('fnc')]).toBeCloseTo(1.06);
    expect(q[key('ppc')]).toBeCloseTo(3.54);
    gameLib.cardSupportTransfer(q,{...options,from:'vox',to:'ppc',amount:1});
    expect(q[key('vox')]).toBeCloseTo(5.4);
  });
  it('shares one donor cap between both destinations and cancels redirection back into FNC', () => {
    const q = state(); q.fnc_parlament_s = 1;
    const original = q[key('ppc')];
    const spent = gameLib.cardSupportTransfer(q,{contest:'parlament',from:'ppc',to:'vox',amount:100,
      maxDonorFraction:.5,constituencies:'all',demographics:'all'});
    expect(spent).toBeLessThanOrEqual(original / 2);
    expect(q[key('ppc')]+q[key('vox')]+q[key('fnc')]).toBeCloseTo(original+5+1);
    const fncBefore = q[key('fnc')], voxBefore = q[key('vox')];
    gameLib.cardSupportTransfer(q,{contest:'parlament',from:'fnc',to:'vox',amount:100,
      maxDonorFraction:.5,constituencies:'all',demographics:'all'});
    expect(q[key('vox')]-voxBefore).toBeCloseTo(fncBefore * .5 * .7);
    expect(q[key('fnc')]).toBeCloseTo(fncBefore * (1-.5*.7));
  });
});

describe('authored roadmap failure routing', () => {
  const game = readFileSync('../out/game.json','utf8');
  function setup() {
    const a = DendryAdapter.fromJSONText(game);
    a.beginGame([1,2,3,4]); a.goToScene('root.start');
    return {a,q:a.qualities as Record<string,any>};
  }
  it('uses one opening national profile regardless of province settlement order', () => {
    const {q} = setup();
    Object.assign(q,{fnc_formed:true,pxc_dissolved:true,independence_movement:85,independence_trust:20,
      parlament_signal_baseline:{independence_movement:85,independence_trust:40},
      parlament_current_ciu:'ciu',cat_coalition:[],cat_coalition_support:[],cat_coalition_abstain:[],
      game_random_state:123});
    for(const c of q.parlament_constituencies) for(const d of q.parlament_demographics) {
      for(const p of q.parties) q[`${p}_parlament_${c}_${d}_support`] = 0;
      for(const [p,n] of Object.entries({ciu:25,erc:10,cup:4.5,fnc:.5,ppc:10,abstain:50}))
        q[`${p}_parlament_${c}_${d}_support`] = n;
    }
    const reverse = JSON.parse(JSON.stringify(q));
    reverse.parlament_constituencies.reverse();
    gameLib.engineTick(q); gameLib.engineTick(reverse);
    for(const c of q.parlament_constituencies) for(const d of q.parlament_demographics)
      for(const p of [...q.parties,'abstain'])
        expect(q[`${p}_parlament_${c}_${d}_support`]).toBeCloseTo(reverse[`${p}_parlament_${c}_${d}_support`],10);
  });
  it('keeps the fixed loss conserved while applying the national opening size and donor-specific diversion', () => {
    const {a,q} = setup();
    Object.assign(q,state(8),{player_party:'erc',cat_coalition:[],cat_coalition_support:[],cat_coalition_abstain:[]});
    const opening = structuredClone(JSON.parse(JSON.stringify(q)));
    a.goToScene('broken_roadmap');
    const ercLoss = 15*.12, ciuLoss = 20*.09;
    const cupGain = (ercLoss+ciuLoss)*.15*.675;
    const fncGain = (ercLoss+ciuLoss)*.05 + .15*(1-.675)*(ercLoss*.15+ciuLoss*.75);
    expect(q[key('cup')]-opening[key('cup')]).toBeCloseTo(cupGain);
    expect(q[key('fnc')]-opening[key('fnc')]).toBeCloseTo(fncGain);
    expect([...q.parties,'abstain'].reduce((s,p)=>s+q[key(p)],0)).toBeCloseTo(100);
    expect(q.independence_trust).toBe(7.5);
  });
  it('clears a CUP deadline once, registers one expiry and applies its own failure credibility', () => {
    const {a,q} = setup();
    Object.assign(q,{player_party:'cup',roadmap_year:2014,broken_roadmap_trigger:true,independence_trust:40});
    const errors = vi.spyOn(console,'error').mockImplementation(()=>{});
    try {
      a.goToScene('broken_roadmap');
      expect(q.roadmap_year).toBe(0);
      expect(q.broken_roadmap).toBe(1);
      expect(q.broken_roadmap_trigger).toBe(false);
      expect(q.broken_roadmap_expires_countdown).toBe(12);
      expect(q.countdowns.filter((c:string)=>c==='broken_roadmap_expires')).toHaveLength(1);
      expect(q.independence_trust).toBe(20);
      expect(profile(q).cupFactor).toBe(0);
      q.broken_roadmap_expires_trigger = true;
      a.goToScene('post_event');
      expect(q.broken_roadmap).toBe(0);
      expect(q.broken_roadmap_trigger).toBe(false);
      expect(errors.mock.calls.filter(c=>String(c[0]).includes('Scene action error'))).toEqual([]);
    } finally { errors.mockRestore(); }
  });
});
