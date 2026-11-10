import {it,expect} from 'vitest';
import {readFileSync} from 'node:fs';
const source=readFileSync('../source/lib/cat_engine.js','utf8');
function run(stocks:Record<string,number>,flags:object={}) {
 let before:any,after:any;
 let code=source.replace('engineTick: monthPasses,','engineTick: monthPasses, testCongreso: monthPassesCongreso,');
 for(const [marker,phase]of [['      if (rates.indy_competition)',0],['      // 4g. Leadership recovery',1]] as const){
  expect(code.split(marker)).toHaveLength(2);code=code.replace(marker,`      capture(${phase}, deltas);\n${marker}`);
 }
 const module={exports:{} as any};new Function('module','capture',code)(module,(phase:number,deltas:any)=>{if(phase)after={...deltas};else before={...deltas};});
 const q:any={congreso_constituencies:['catalunya'],congreso_parties_catalunya:Object.keys(stocks).filter(p=>p!=='abstain'),
  gdp_growth:0,unemployment:20,welfare_index:50,social_dissent:80,cat_spa_relations:50,game_random_state:123,
  parlament_current_ciu:'cdc',cat_coalition:['cdc'],...flags};
 for(const [p,n]of Object.entries(stocks))q[p+'_congreso_catalunya_support']=n;
 module.exports.testCongreso(q);
 const changes=Object.fromEntries(Object.keys(after).map(p=>[p,after[p]-before[p]]));
 expect(Object.values(changes).reduce((s,n)=>s+n,0)).toBeCloseTo(0,12);
 return changes;
}
it('allows ERC and CDC to recruit back from an oversized competitor',()=>{
 expect(run({cdc:40,erc:10,abstain:50}).erc).toBeGreaterThan(0);
 expect(run({cdc:10,erc:40,abstain:50}).erc).toBeLessThan(0);
});
it('ties government dissatisfaction to actual cabinet responsibility',()=>{
 const stocks={cdc:30,erc:20,abstain:50};
 const cabinet=run(stocks),external=run(stocks,{cat_coalition:['erc'],cat_coalition_support:['cdc']}),opposition=run(stocks,{cat_coalition:['erc']});
 expect(cabinet.abstain).toBeGreaterThan(external.abstain);
 expect(external.abstain).toBeGreaterThan(opposition.abstain);
 expect(opposition.abstain).toBe(0);
});
it('retains scandal pressure outside government without inventing recipients',()=>{
 const r=run({cdc:30,erc:20,abstain:50},{cat_coalition:['erc'],corruption_events_ciu:2});
 expect(r.abstain).toBeGreaterThan(0);
 const absent=run({cdc:100},{corruption_events_ciu:2});
 expect(absent.cdc).toBe(0);expect(absent.abstain).toBeUndefined();
});
it('charges the current live successor instead of a stale joint-list residue',()=>{
 const r=run({dl:30,jxsi:1,erc:20,abstain:49},{parlament_current_ciu:'dl',cat_coalition:['dl'],corruption_events_ciu:2});
 expect(r.dl).toBeLessThan(0);expect(r.jxsi).toBe(0);
});
it('does not manufacture an ERC ballot or switch into a stale joint list',()=>{
 const r=run({dl:40,jxsi:1,abstain:59},{parlament_current_ciu:'dl',cat_coalition:['dl']});
 expect(r.erc).toBeUndefined();expect(r.jxsi).toBe(0);expect(r.dl).toBeLessThan(0);
 const joint=run({jxsi:40,abstain:60},{parlament_current_ciu:'jxsi',cat_coalition:['jxsi']});
 expect(joint.erc).toBeUndefined();expect(joint.cdc).toBeUndefined();
 expect(joint.jxsi+joint.abstain).toBeCloseTo(0,12);
});
