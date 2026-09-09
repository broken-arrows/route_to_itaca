import { describe, expect, it } from 'vitest';
import { gameLib as g } from '../src/game-bindings';
const base={independence_movement:85,cat_spa_relations:15,psc_recovery_mult:1,psc_leader:'Miquel Iceta',time:35,next_election_time:36};
const ctx={support:{psc:10,cs:8,ppc:0,fl:0},scaleCatSpa:1};
const switchRoute=(q:any,c=ctx)=>g.buildParlamentCompetitionTransfers(q,c).filter(r=>r.mechanism.includes('psc_cs')||r.mechanism.includes('cs_psc'));
const carriers={icr:'ciu',il:'erc',cup:'cup',fl:'icv',abs:'abstain',pdcat:'pdcat'};
const support={ciu:30,cdc:30,dl:30,pdcat:30,junts:30,jxsi:30,jxcat:30,erc:15,cup:5,icv:5,abstain:30};
const corrupt=(q:any,mapping=carriers)=>g.buildParlamentCorruptionTransfers({corruption_events_ciu:2,...q},{carriers:mapping,support,responsibility:{}});
const volume=(rs:any[])=>rs.reduce((s,r)=>s+r.amount,0);
describe('movement-led Cs and identity corruption',()=>{
  it('switches PSC to Cs under strong movement and reverses under weak movement despite poor relations',()=>{
    expect(switchRoute(base)[0]).toMatchObject({from:'psc',to:'cs'});
    expect(switchRoute({...base,independence_movement:25,psc_recovery_mult:0})[0]).toMatchObject({from:'cs',to:'psc'});
    const protectedPsc=switchRoute({...base,psc_leader:'Montserrat Tura',psc_recovery_mult:1.2});
    expect(volume(protectedPsc)).toBeLessThan(volume(switchRoute(base)));
    expect(volume(switchRoute(base))).toBeGreaterThan(volume(switchRoute({...base,cat_spa_relations:75})));
  });
  it('uses the broader PSC electorate and converges without a preset eligible quota',()=>{
    const context=structuredClone(ctx);
    const first=volume(switchRoute(base,context));
    for(let i=0;i<6000;i++) for(const r of switchRoute(base,context)){
      const from=r.from as 'psc'|'cs', to=r.to as 'psc'|'cs';
      context.support[from]-=r.amount;context.support[to]+=r.amount;
    }
    expect(ctx.support.psc-context.support.psc).toBeGreaterThan(ctx.support.psc*.08);
    expect(context.support.psc).toBeGreaterThanOrEqual(0);
    expect(context.support.psc+context.support.cs).toBeCloseTo(18,8);
    expect(volume(switchRoute(base,context))).toBeLessThan(first/100);
  });
  it('dampens corruption across identities, while additional scandals have diminishing effects',()=>{
    const totals=['ciu','cdc','dl','pdcat','junts'].map(p=>volume(corrupt({parlament_current_ciu:p},{...carriers,icr:p})));
    for(let i=1;i<totals.length;i++) expect(totals[i]).toBeLessThan(totals[i-1]);
    const first=volume(corrupt({corruption_events_ciu:1}));
    const second=volume(corrupt({corruption_events_ciu:2}));
    const third=volume(corrupt({corruption_events_ciu:3}));
    expect(third-second).toBeLessThan(second-first);
    expect(second-first).toBeLessThan(first);
  });
  it.each(['jxsi','jxcat'])('does not reward ERC inside active %s even if the other membership flag is false',list=>{
    const q={parlament_current_ciu:'cdc',[`erc_in_${list}`]:true,erc_in_jxsi:list==='jxsi',erc_in_jxcat:list==='jxcat'};
    const routes=corrupt(q,{...carriers,icr:list,il:list});
    expect(routes.some(r=>r.to==='il')).toBe(false);
    expect(volume(routes)).toBeLessThan(volume(corrupt({parlament_current_ciu:'cdc'},{...carriers,icr:'cdc'})));
    // An inactive list's old flag must not block a standalone ERC ballot.
    expect(corrupt(q,{...carriers,icr:'cdc'}).some(r=>r.to==='il')).toBe(true);
  });
  it('keeps split PDeCAT exposed and reduces CUP outsider advantage when responsible',()=>{
    const q={parlament_current_ciu:'junts',pdcat_split:true,corruption_events_ciu:2};
    const mapping={...carriers,icr:'junts'};
    const outside=corrupt(q,mapping);
    const inside=g.buildParlamentCorruptionTransfers(q,{carriers:mapping,support,responsibility:{cup:1}});
    expect(outside.some(r=>r.from==='pdcat')).toBe(true);
    expect(volume(inside.filter(r=>r.to==='cup'))).toBeLessThan(volume(outside.filter(r=>r.to==='cup')));
    expect(volume(inside)).toBeCloseTo(volume(outside),10);
    expect(q.corruption_events_ciu).toBe(2);
  });
});
