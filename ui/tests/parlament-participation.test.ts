import { describe, expect, it } from 'vitest';
import { createRequire } from 'node:module';
import { gameLib } from '../src/game-bindings';
const { buildParlamentParticipationTransfers: build, resetParlamentParticipation: reset } =
  createRequire(import.meta.url)('../../source/lib/cat_engine.js');
const carriers = { abs:'abstain',icr:'ciu',il:'erc',cup:'cup',cs:'cs',ppc:'ppc',psc:'psc',fl:'icv',unio:'ciu',pdcat:'ciu' };
const initial = {ciu:25,erc:10,cup:5,cs:8,ppc:8,psc:8,icv:6,abstain:30};
const key = (p: string) => `${p}_parlament_girona_middle_support`;
function state(overrides = {}) {
  return { parties:Object.keys(initial).filter(p=>p!=='abstain'), parlament_constituencies:['girona'], parlament_demographics:['middle'],
    parlament_current_ciu:'ciu', parlament_current_icv:'icv', independence_movement:85,cat_spa_relations:15,time:35,next_election_time:36,
    ...Object.fromEntries(Object.entries(initial).map(([p,v])=>[key(p),v])), ...overrides } as Record<string, any>;
}
function step(q: Record<string, any>, other: any[] = [], mapping: Record<string,string> = carriers) {
  const support=Object.fromEntries([...q.parties,'abstain'].map(p=>[p,q[key(p)]]));
  const requests=build(q,{province:'girona',demographic:'middle',carriers:mapping,support});
  gameLib.applyParlamentTransfers(q,'girona','middle',{},[...requests,...other]);
  return requests;
}
function check(q: Record<string, any>) {
  const support=[...q.parties,'abstain'].map(p=>q[key(p)]);
  expect(support.reduce((a,b)=>a+b,0)).toBeCloseTo(100,8);
  expect(support.every(n=>Number.isFinite(n)&&n>=0)).toBe(true);
  for(const s of Object.values(q.parlament_participation.cells['girona.middle']) as any[])
    expect(s.active>=-1e-10 && s.inactive>=-1e-10).toBe(true);
}
describe('level-driven Parlament participation', () => {
  it('mobilizes other families under confrontation while Cs needs street momentum', () => {
    const hot=state({independence_movement:25});
    const calm=state({independence_movement:25,cat_spa_relations:65});
    for(let i=0;i<24;i++){step(hot);step(calm);}
    expect(hot[key('abstain')]).toBeLessThan(calm[key('abstain')]);
    expect(hot[key('cs')]).toBeCloseTo(calm[key('cs')], 9);
    expect(hot[key('cs')]).toBeLessThan(initial.cs);
    expect(hot[key('erc')]).toBeGreaterThan(calm[key('erc')]);
    check(hot);check(calm);
  });
  it('makes street momentum amplify independence participation alongside confrontation', () => {
    const high=state(); const low=state({independence_movement:25});
    for(let i=0;i<24;i++){step(high);step(low);}
    expect(high[key('erc')]).toBeGreaterThan(low[key('erc')]);
    expect(high[key('cup')]).toBeGreaterThan(low[key('cup')]);
  });
  it('converges under constant conditions without draining the abstention pool', () => {
    const q=state(); step(q);
    const pool=q.parlament_participation.cells['girona.middle'];
    const capacity=Object.values(pool).reduce((n:number,s:any)=>n+s.active+s.inactive,0);
    for(let i=0;i<6000;i++)step(q);
    const previous=q[key('abstain')];
    step(q);
    expect(Math.abs(q[key('abstain')]-previous)).toBeLessThan(1e-7);
    expect(q[key('abstain')]).toBeGreaterThan(0);
    expect(Object.values(pool).reduce((n:number,s:any)=>n+s.active+s.inactive,0)).toBeCloseTo(capacity,9);
    check(q);
  });
  it('releases voters gradually as conditions cool and campaign proximity falls', () => {
    const q=state(); for(let i=0;i<100;i++)step(q);
    const mobilized=q[key('abstain')];
    q.next_election_time=q.time+48;
    q.cat_spa_relations=65;q.independence_movement=25;
    step(q);
    expect(q[key('abstain')]).toBeGreaterThan(mobilized);
    expect(q[key('abstain')]-mobilized).toBeLessThan(1);
    q.next_election_time=q.time+1; q.cat_spa_relations=15;q.independence_movement=85;
    for(let i=0;i<500;i++)step(q);
    expect(q[key('abstain')]).toBeGreaterThan(0);
    check(q);
  });
  it('advances memory by realized transfers when another route exhausts the donor', () => {
    const q=state();
    const support={...initial};
    const requests=build(q,{province:'girona',demographic:'middle',carriers,support});
    const before=JSON.parse(JSON.stringify(q.parlament_participation.cells['girona.middle']));
    gameLib.applyParlamentTransfers(q,'girona','middle',{},[...requests,{mechanism:'competing',from:'abs',to:'psc',amount:300}]);
    const ercRequest=requests.find((r:any)=>r.to==='il');
    const ercRealized=q[key('erc')]-initial.erc;
    expect(ercRealized).toBeGreaterThan(0);
    expect(ercRealized).toBeLessThan(ercRequest.amount);
    expect(q.parlament_participation.cells['girona.middle'].il.active-before.il.active).toBeCloseTo(ercRealized,10);
    check(q);
  });
  it('survives JSON restore, keeps dormant families inert and preserves memory across list formation', () => {
    let q=state();step(q);
    q=JSON.parse(JSON.stringify(q));
    const untouched=JSON.parse(JSON.stringify(q.parlament_participation.cells['girona.middle'].vox));
    const before=Object.values(q.parlament_participation.cells['girona.middle']).reduce((n:number,s:any)=>n+s.active+s.inactive,0);
    q.parties.push('jxsi');q[key('jxsi')]=q[key('ciu')]+q[key('erc')];q[key('ciu')]=0;q[key('erc')]=0;q.erc_in_jxsi=true;
    for(let i=0;i<12;i++)step(q,[],{...carriers,icr:'jxsi',il:'jxsi',unio:'jxsi',pdcat:'jxsi'});
    expect(Object.values(q.parlament_participation.cells['girona.middle']).reduce((n:number,s:any)=>n+s.active+s.inactive,0)).toBeCloseTo(before,9);
    expect(q.parlament_participation.cells['girona.middle'].vox).toEqual(untouched);
    check(q);
    reset(q);expect(q.parlament_participation).toBeUndefined();
  });
});
