import { describe, expect, it } from 'vitest';
import { gameLib } from '../src/game-bindings';

const carriers = { icr: 'ciu', il: 'erc', cup: 'cup', fnc: 'fnc', abs: 'abstain' };
const initial = { ciu: 30, erc: 15, cup: 5, fnc: 0, abstain: 50 };
const key = (p: string) => `${p}_parlament_girona_middle_support`;
function state(overrides = {}) {
  return { parties: ['ciu','erc','cup','fnc'], parlament_constituencies:['girona'], parlament_demographics:['middle'],
    parlament_current_ciu:'ciu', independence_movement:85, independence_trust:15,
    fnc_formed:true,pxc_dissolved:true,
    ...Object.fromEntries(Object.entries(initial).map(([p,v])=>[key(p),v])), ...overrides } as Record<string, any>;
}
function step(q: Record<string, any>, mapping = carriers, extra: any[] = []) {
  gameLib.advanceParlamentDisappointment(q);
  const requests = gameLib.buildParlamentDisappointmentTransfers(q, { province:'girona',demographic:'middle',carriers:mapping,
    support:Object.fromEntries(Object.keys(initial).map(p=>[p,q[key(p)]])) });
  gameLib.applyParlamentTransfers(q,'girona','middle',{},[...requests,...extra]);
  return requests;
}
describe('sustained process disappointment', () => {
  it('gives CUP the early response and FNC a slower buildup, funded by CDC', () => {
    const q=state();
    const first=step(q);
    const fnc = (requests: any[]) => requests.filter(r=>r.to==='fnc').reduce((s,r)=>s+r.amount,0);
    expect(fnc(first)).toBeLessThan(first.filter(r=>r.to==='cup').reduce((s,r)=>s+r.amount,0));
    for(let i=0;i<17;i++) step(q);
    expect(fnc(step(q))).toBeGreaterThan(fnc(first)*10);
    expect(first.filter(r=>r.to==='fnc').every(r=>r.from==='icr')).toBe(true);
  });
  it('requires strong movement and current distrust; accumulated frustration fades during recovery', () => {
    const cold=state({independence_movement:30});
    for(let i=0;i<24;i++) expect(step(cold)).toHaveLength(0);
    const hot=state();
    for(let i=0;i<24;i++) step(hot);
    const peak=hot.parlament_disappointment.frustration;
    hot.independence_trust=60;
    expect(step(hot)).toHaveLength(0);
    expect(hot.parlament_disappointment.frustration).toBeGreaterThan(0);
    for(let i=0;i<23;i++) step(hot);
    expect(hot.parlament_disappointment.frustration).toBeLessThan(peak/10);
  });
  it('preserves finite capacity across saves and campaign timing changes without draining the mainstream', () => {
    let q=state();
    for(let i=0;i<600;i++) {
      if(i===12) q=JSON.parse(JSON.stringify(q));
      q.next_election_time=i+1;
      step(q);
    }
    expect(q[key('ciu')]).toBeGreaterThanOrEqual(30*.88-1e-9);
    expect(q[key('erc')]).toBeGreaterThanOrEqual(15*.88-1e-9);
    expect(Object.keys(initial).reduce((s,p)=>s+q[key(p)],0)).toBeCloseTo(100,8);
    expect(step(q).reduce((s,r)=>s+r.amount,0)).toBeLessThan(1e-5);
    gameLib.resetParlamentParticipation(q);
    expect(q.parlament_disappointment).toBeUndefined();
  });
  it('does not create FNC before formation or send votes within a joint list', () => {
    const dormant=state({fnc_formed:false});
    expect(step(dormant).every(r=>r.to!=='fnc')).toBe(true);
    dormant.fnc_formed=true;
    expect(step(dormant).some(r=>r.to==='fnc')).toBe(true);
    const merged=state();
    step(merged,{...carriers,il:'ciu',cup:'ciu'});
    const cell=merged.parlament_disappointment.cells['girona.middle'];
    expect(cell.icr.remaining+cell.il.remaining).toBeLessThanOrEqual(30*.12);
    expect(merged[key('cup')]).toBe(5);
  });
  it('consumes capacity only for actual transfers after competing donor requests', () => {
    const q=state();
    const before=q[key('cup')]+q[key('fnc')];
    step(q,carriers,[{from:'icr',to:'abs',amount:1000,mechanism:'test.competitor'}]);
    const remaining=Object.values(q.parlament_disappointment.cells['girona.middle']).reduce((s:number,c:any)=>s+c.remaining,0);
    expect(45*.12-remaining).toBeCloseTo(q[key('cup')]+q[key('fnc')]-before,9);
    expect(q[key('ciu')]).toBeGreaterThanOrEqual(0);
  });
});
