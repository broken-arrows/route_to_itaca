import { it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

const source=readFileSync('../source/lib/cat_engine.js','utf8');
function run(region:string, support:Record<string,number>, channeling=1, flags:object={}) {
  let before:any, after:any;
  let code=source.replace('engineTick: monthPasses,','engineTick: monthPasses, testCongreso: monthPassesCongreso,');
  for(const [marker,phase]of [['      // Regional competition is two-way:',0],['      // 4g. Leadership recovery',1]] as const) {
    expect(code.split(marker)).toHaveLength(2);
    code=code.replace(marker,`      capture(${phase}, deltas);\n${marker}`);
  }
  const module={exports:{} as any};
  new Function('module','capture',code)(module,(phase:number,deltas:any)=>{if(phase)after={...deltas};else before={...deltas};});
  const q:any={congreso_constituencies:[region],['congreso_parties_'+region]:['podemos','up','iu','ehbildu','amaiur','pnv','gbai','pp','psoe','cs'],
    gdp_growth:0,unemployment:20,welfare_index:50,social_dissent:50,cat_spa_relations:50,
    game_random_state:123,podemos_channeling:channeling,spa_ehbildu_active:true,upn_in_pp:true,...flags};
  for(const [p,n]of Object.entries(support))q[p+'_congreso_'+region+'_support']=n;
  module.exports.testCongreso(q);
  const changes=Object.fromEntries(Object.keys(after).map(p=>[p,after[p]-before[p]]));
  expect(Object.values(changes).reduce((a,b)=>a+b,0)).toBeCloseTo(0,12);
  return changes;
}
it.each(['euskadi','navarra'])('recruits from regional parties when Podemos is small in %s',region=>{
  const r=run(region,{podemos:10,ehbildu:30,pnv:30,gbai:30});
  expect(r.podemos).toBeGreaterThan(0);
  expect(r.ehbildu).toBeLessThan(0);
  expect(region==='euskadi'?r.pnv:r.gbai).toBeLessThan(0);
});
it.each(['euskadi','navarra'])('regional parties regain voters from an oversized Podemos in %s',region=>{
  const r=run(region,{podemos:80,ehbildu:5,pnv:5,gbai:5,abstain:5});
  expect(r.podemos).toBeLessThan(0);
  expect(r.ehbildu).toBeGreaterThan(0);
});
it('responds to channeling rather than applying a permanent left bonus',()=>{
  const support={podemos:20,ehbildu:20,pnv:30,abstain:30};
  expect(run('euskadi',support,1).podemos).toBeGreaterThan(0);
  expect(run('euskadi',support,0).podemos).toBeLessThan(0);
});
it('uses the live successor and legacy abertzale carriers',()=>{
  const r=run('navarra',{up:10,amaiur:30,gbai:30,abstain:30},1,{spa_ehbildu_active:false});
  expect(r.up).toBeGreaterThan(0);
  expect(r.amaiur).toBeLessThan(0);
  expect(r.podemos).toBeUndefined();
});
it.each([{ehbildu:40,pnv:30,abstain:30},{podemos:40,abstain:60}])('does not invent unavailable competitors with stocks %j',support=>{
  expect(Object.values(run('euskadi',support)).every(n=>n===0)).toBe(true);
});
it('does not apply Basque nationalist competition in Valencia',()=>{
  const r=run('valencia',{podemos:10,ehbildu:30,pnv:30,abstain:30});
  expect(r.ehbildu).toBe(0);expect(r.pnv).toBe(0);
  // Valencia has its own participation route, independently of these parties.
  expect(r.podemos).toBeGreaterThan(0);expect(r.abstain).toBeLessThan(0);
});
it.each(['euskadi','navarra'])('united competition still returns voters to depleted regional parties in %s',region=>{
 const r=run(region,{up:80,ehbildu:5,pnv:5,gbai:5,abstain:5},.8,{iu_in_up:true});
 expect(r.up).toBeLessThan(0);expect(r.ehbildu).toBeGreaterThan(0);
 expect(region==='euskadi'?r.pnv:r.gbai).toBeGreaterThan(0);
});
it('unification changes Navarra competition while continuing to respect channeling',()=>{
 const support={up:20,ehbildu:15,gbai:15,abstain:50};
 const split=run('navarra',support,.8),united=run('navarra',support,.8,{iu_in_up:true});
 expect(united.gbai).toBeLessThan(split.gbai);
 expect(run('navarra',support,0,{iu_in_up:true}).up).toBeLessThan(united.up);
});
it('lets a separate IU recover voters and Podemos compete back without changing the combined left stock',()=>{
 const recover=run('rest',{podemos:30,iu:2,abstain:68},.9);
 expect(recover.iu).toBeGreaterThan(0);expect(recover.podemos).toBeLessThan(0);
 expect(recover.iu+recover.podemos).toBeCloseTo(0,12);expect(recover.abstain).toBe(0);
 const recruit=run('rest',{podemos:5,iu:20,abstain:75},.9);
 expect(recruit.podemos).toBeGreaterThan(0);expect(recruit.iu).toBeLessThan(0);
});
it('makes IU competition respond to channeling',()=>{
 const support={podemos:20,iu:8,abstain:72};
 expect(run('rest',support,0).iu).toBeGreaterThan(run('rest',support,1).iu);
});
it('stops separate IU competition after unification and does not revive absent ballots',()=>{
 // The authored unification moves IU's stock into UP before the monthly tick.
 const united=run('rest',{up:30,iu:0,abstain:70},.8,{iu_in_up:true});
 expect(united.iu).toBeUndefined();expect(united.up).toBe(0);
 const absent=run('rest',{podemos:30,abstain:70});
 expect(absent.iu).toBeUndefined();expect(absent.podemos).toBe(0);
});
