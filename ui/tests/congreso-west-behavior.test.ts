import {it,expect} from 'vitest';
import {readFileSync} from 'node:fs';
import {DendryAdapter} from '../src/engine/adapter';

const source=readFileSync('../source/lib/cat_engine.js','utf8');
const game=readFileSync('../out/game.json','utf8');
function probe(region:string, overrides:any={}, support:any={pp:25,psoe:25,podemos:20,bng:5,abstain:25}){
 const snapshots:any={};
 let code=source.replace('engineTick: monthPasses,','engineTick: monthPasses, testCongreso: monthPassesCongreso,');
 for(const [name,marker]of Object.entries({government:'      // Government performance follows',corruption:'      // 4b. PP corruption',psoe:'      // 4f. Podemos channeling',regional:'      // Regional competition is two-way:',end:'      // 4g. Leadership recovery'})){
  expect(code.split(marker)).toHaveLength(2);
  code=code.replace(marker,`      capture(${JSON.stringify(name)}, deltas);\n${marker}`);
 }
 const module={exports:{} as any};new Function('module','capture',code)(module,(name:string,d:any)=>{snapshots[name]={...d};});
 const q:any={congreso_constituencies:[region],['congreso_parties_'+region]:['pp','psoe','podemos','up','cs','bng'],
  gdp_growth:1,_prev_spa_gdp:0,unemployment:20,welfare_index:50,social_dissent:50,cat_spa_relations:50,
  game_random_state:123,podemos_channeling:1,spa_bng_active:true,upn_in_pp:true,spanish_coalition:['pp'],...overrides};
 for(const[p,v]of Object.entries(support))q[p+'_congreso_'+region+'_support']=v;
 module.exports.testCongreso(q);
 const delta=(a:string,b:string)=>Object.fromEntries(Object.keys(snapshots[b]).map(p=>[p,snapshots[b][p]-snapshots[a][p]]));
 return {government:delta('government','corruption'),psoe:delta('psoe','regional'),regional:delta('regional','end')};
}
const conserved=(r:Record<string,number>)=>expect(Object.values(r).reduce((a,b)=>a+b,0)).toBeCloseTo(0,12);
it.each(['galicia','rest','valencia','balears'])('rewards and penalizes the current cabinet in %s',region=>{
 for(const party of ['pp','psoe']){
  const good=probe(region,{spanish_coalition:[party]}).government;
  const bad=probe(region,{spanish_coalition:[party],gdp_growth:-1}).government;
  expect(good[party]).toBeGreaterThan(0);expect(bad[party]).toBeLessThan(0);
  expect(good[party==='pp'?'psoe':'pp']).toBe(0);
  conserved(good);conserved(bad);
 }
});
it('shares the performance response across a coalition without multiplying it',()=>{
 const solo=probe('rest').government, joint=probe('rest',{spanish_coalition:['pp','psoe']}).government;
 expect(joint.pp+joint.psoe).toBeCloseTo(solo.pp,12);conserved(joint);
});
it('handles a Podemos successor government and absent abstention',()=>{
 const r=probe('rest',{spanish_coalition:['up']},{pp:25,psoe:25,up:20,abstain:30}).government;
 expect(r.up).toBeGreaterThan(0);conserved(r);
 const absent=probe('rest',{}, {pp:50,psoe:50}).government;
 expect(Object.values(absent).every(n=>n===0)).toBe(true);
});
it.each(['galicia','rest','valencia','balears'])('allows PSOE and Podemos to recruit back from oversized rivals in %s',region=>{
 const small=probe(region,{}, {pp:10,psoe:50,podemos:5,abstain:35}).psoe;
 const large=probe(region,{}, {pp:10,psoe:5,podemos:50,abstain:35}).psoe;
 expect(small.podemos).toBeGreaterThan(0);expect(large.podemos).toBeLessThan(0);
 conserved(small);conserved(large);
});
it('balances Galician participation and BNG competition without reviving a folded list',()=>{
 const r=probe('galicia').regional;conserved(r);
 expect(r.bng).toBeLessThan(0);
 const folded=probe('galicia',{spa_bng_active:false},{pp:30,psoe:20,up:20,abstain:30}).regional;
 expect(folded.bng).toBeUndefined();conserved(folded);
});
it('keeps other regions outside the added government response',()=>{
 for(const region of ['euskadi','navarra','catalunya'])expect(Object.values(probe(region).government).every(n=>n===0)).toBe(true);
});

it.each(['valencia','balears'])('mobilizes and demobilizes the left within donor limits in %s',region=>{
 const small=probe(region,{}, {pp:20,psoe:20,podemos:5,abstain:55}).regional;
 const large=probe(region,{}, {pp:20,psoe:20,up:55,abstain:5}).regional;
 expect(small.podemos).toBeGreaterThan(0);expect(small.podemos).toBeLessThanOrEqual(55*.05);
 expect(large.up).toBeLessThan(0);expect(-large.up).toBeLessThanOrEqual(55*.05);
 conserved(small);conserved(large);
 const absent=probe(region,{}, {pp:50,psoe:50}).regional;
 expect(Object.values(absent).every(n=>n===0)).toBe(true);
});

it.each([[0,false,true],[-1,false,false],[-2,false,false],[0,true,false],[2,true,true]] as const)(
 'BNG lineup with outreach %s and constitutional reform %s stays independent=%s', (outreach,reform,independent)=>{
 // Let the actual post_event month advance without unrelated pending editions.
 const compiled=JSON.parse(game);
 compiled.scenes['post_event.events_choice'].options=[];
 compiled.scenes['post_event.elections_choice'].options=[];
 const a=DendryAdapter.fromJSONText(JSON.stringify(compiled));a.beginGame([1,2,3,4]);a.goToScene('root.start');
 const q=a.qualities as any;
 Object.assign(q,{player_party:'erc',year:2016,month:4,month_actions:1,month_actions_last:0,
  podemos_surged:true,iu_in_up:true,constitution_reformed:reform,party_reaching_out_bng:outreach,
  spa_bng_active:true,spa_bng_active_applied:true,bng_congreso_galicia_support:5,
  podemos_congreso_galicia_support:0,up_congreso_galicia_support:25,psoe_congreso_galicia_support:20,
  next_spanish_election_time:q.time+3});
 a.goToScene('post_event');expect(q.spa_bng_active).toBe(independent);
 expect(q.bng_congreso_galicia_support>0).toBe(independent);
});

it('a successful June 2018 censure moves subsequent economic responsibility to PSOE',()=>{
 const a=DendryAdapter.fromJSONText(game);a.beginGame([1,2,3,4]);
 const q=a.qualities as any;
 for(const k of Object.keys(q))if(k.endsWith('_congreso_s'))q[k]=0;
 Object.assign(q,{player_party:'erc',year:2018,month:6,time:70,week:1,month_actions:0,month_actions_last:0,rubicon:false,
  next_spanish_election_time:94,congreso_seats:{total:350},congreso_s_majority:176,
  congreso_parties:['pp','psoe','up','erc','pnv'],psoe_congreso_s:100,up_congreso_s:60,erc_congreso_s:10,pnv_congreso_s:6,pp_congreso_s:174,
  spanish_pm:'Mariano Rajoy',spanish_pm_party:'pp',spanish_coalition:['pp'],spa_caretaker_gov:false,
  psoe_leader:'Pedro Sánchez',up_leader:'Pablo Iglesias',corruption_pp:55,cat_spa_relations:55,party_resources:10,
  psc_relations:80,parlament_current_icv:'icv',icv_relations:80,parlament_current_ciu:'pdcat',pdcat_roadmap:2,
  erc_in_jxsi:false,erc_in_jxcat:false,countdowns:[]});
 a.goToScene('post_event');a.goToScene('spanish_censure');
 const choices=a.currentFrame().choices;
 const index=choices.findIndex(c=>c.id==='spanish_censure.support');
 expect(index).toBeGreaterThanOrEqual(0);expect(choices[index].canChoose).toBe(true);a.choose(index);
 expect(q.censure_won).toBe(true);expect(q.spanish_pm).toBe('Pedro Sánchez');expect(q.spanish_coalition).toEqual(['psoe']);
 const good=probe('rest',{spanish_coalition:q.spanish_coalition}).government;
 const bad=probe('rest',{spanish_coalition:q.spanish_coalition,gdp_growth:-1}).government;
 expect(good.psoe).toBeGreaterThan(0);expect(bad.psoe).toBeLessThan(0);expect(good.pp).toBe(0);expect(bad.pp).toBe(0);
});
