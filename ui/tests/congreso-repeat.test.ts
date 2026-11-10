import {it,expect} from 'vitest';
import {readFileSync} from 'node:fs';
import {gameLib} from '../src/game-bindings';
import {DendryAdapter} from '../src/engine/adapter';
import {bindSceneRuntime} from './helpers/scene-runtime';
const G=gameLib as any;
const source=readFileSync('../source/lib/cat_engine.js','utf8');
const game=readFileSync('../out/game.json','utf8');
const pactActions=JSON.parse(game).scenes['congreso_coalition_psoe_cs.wtf'].onArrival
 .map((action:{$code:string})=>new Function('state','Q','G',action.$code));
function failedPact(q:any){
 const runtime=bindSceneRuntime(q);
 for(const action of pactActions)action.call(runtime.engine,runtime.state,runtime.Q,runtime.G);
}
function fixture(region='valencia',stocks:any={pp:25,psoe:20,cs:10,podemos:20,iu:5,compromis:5,abstain:15}){
 const q:any={time:40,year:2015,month:12,next_spanish_election_time:88,podemos_channeling:.9,
  congreso_constituencies:[region],['congreso_parties_'+region]:Object.keys(stocks).filter(p=>p!=='abstain')};
 for(const [p,n]of Object.entries(stocks))q[p+'_congreso_'+region+'_support']=n;
 return q;
}
const stock=(q:any,p:string,c='valencia')=>q[p+'_congreso_'+c+'_support']||0;
const total=(q:any,c='valencia')=>[...new Set([...q['congreso_parties_'+c],'abstain'])].reduce((s:number,p:any)=>s+stock(q,p,c),0);

it('preserves more Basque left participation while still reducing channeling',()=>{
 const q=fixture('euskadi',{up:30,psoe:20,pnv:25,ehbildu:15,abstain:10});
 G.callCongresoRepeatElection(q);
 expect(stock(q,'up','euskadi')).toBeCloseTo(28.2,12);
 expect(stock(q,'abstain','euskadi')).toBeCloseTo(11.8,12);
 expect(stock(q,'pnv','euskadi')).toBe(25);expect(stock(q,'ehbildu','euskadi')).toBe(15);
 expect(q.podemos_channeling).toBeCloseTo(.78,12);expect(total(q,'euskadi')).toBeCloseTo(100,12);
});

it('moves a fraction of each separate left list to abstention and lowers channeling',()=>{
 const q=fixture(), before=total(q);
 expect(G.callCongresoRepeatElection(q)).toBe(true);
 for(const [p,n]of Object.entries({podemos:20,iu:5,compromis:5}))expect(stock(q,p)).toBeCloseTo(n*.84,12);
 expect(stock(q,'abstain')).toBeCloseTo(15+30*.16,12);
 expect(q.podemos_channeling).toBeCloseTo(.78,12);expect(total(q)).toBeCloseTo(before,12);
 expect(stock(q,'pp')).toBe(25);expect(stock(q,'psoe')).toBe(20);expect(stock(q,'cs')).toBe(10);
});
it('charges a merged successor once and leaves independent nationalist parties alone',()=>{
 const q=fixture('valencia',{up:30,erc:10,bng:10,ehbildu:10,cup:10,psoe:10,abstain:20});
 q.congreso_parties_valencia.push('up','podemos','iu');
 G.callCongresoRepeatElection(q);
 expect(stock(q,'up')).toBeCloseTo(25.2,12);
 for(const p of ['erc','bng','ehbildu','cup','psoe'])expect(stock(q,p)).toBe(10);
 expect(stock(q,'podemos')).toBe(0);expect(stock(q,'iu')).toBe(0);expect(total(q)).toBeCloseTo(100,12);
});
it('does not charge the same pending repeat twice, including after save/load',()=>{
 const q=fixture();G.callCongresoRepeatElection(q);
 const loaded=JSON.parse(JSON.stringify(q));loaded.time++;
 const before=JSON.stringify(loaded);
 expect(G.callCongresoRepeatElection(loaded)).toBe(false);expect(JSON.stringify(loaded)).toBe(before);
 // Simulate the election's normal 48-month schedule advance, then a new failure.
 loaded.time=46;loaded.month=6;loaded.year=2016;loaded.next_spanish_election_time+=48;
 expect(G.callCongresoRepeatElection(loaded)).toBe(true);
 expect(loaded.next_spanish_election_month).toBe(12);expect(loaded.next_spanish_election_year).toBe(2016);
 expect(stock(loaded,'podemos')).toBeCloseTo(20*.84*.84,12);
});
it.each([1,6,7,12])('schedules a six-month repeat from month %s',month=>{
 const q=fixture();q.month=month;q.year=2019;G.callCongresoRepeatElection(q);
 expect(q.next_spanish_election_month).toBe((month+5)%12+1);
 expect(q.next_spanish_election_year).toBe(2019+Math.floor((month+5)/12));
 expect(q.next_spanish_election_time).toBe(46);
});
it('keeps small stocks nonnegative, channeling bounded, and absent parties absent',()=>{
 const q=fixture('rest',{sumar:.01,psoe:40,abstain:59.99});q.podemos_channeling=.02;
 G.callCongresoRepeatElection(q);expect(q.podemos_channeling).toBe(0);
 expect(stock(q,'sumar','rest')).toBeCloseTo(.0084,12);expect(total(q,'rest')).toBeCloseTo(100,12);
 const none=fixture('rest',{psoe:50,abstain:50});delete none.podemos_channeling;
 G.callCongresoRepeatElection(none);expect(total(none,'rest')).toBe(100);expect(stock(none,'podemos','rest')).toBe(0);
});
it.each([
 ['valencia',.10,.03],['balears',.10,.03],['galicia',.10,.06],['rest',.12,.06],
 ['euskadi',.08,.06],['navarra',.10,.06],['catalunya',.08,.06],
] as const)('scales pact losses by opening Cs support in %s', (region,toPP,toAbs)=>{
 for(const cs of [.1,10,40]){
  const q=fixture(region,{pp:20,psoe:20,cs,abstain:60-cs});failedPact(q);
  expect(stock(q,'cs',region)).toBeCloseTo(cs*(1-toPP-toAbs),12);
  expect(stock(q,'pp',region)).toBeCloseTo(20+cs*toPP,12);
  expect(stock(q,'abstain',region)).toBeCloseTo(61-cs+cs*toAbs,12);
  expect(stock(q,'psoe',region)).toBeCloseTo(19,12);
  expect(total(q,region)).toBeCloseTo(100,12);
 }
});
it('routes the pact defection to the live PP successor without creating Cs',()=>{
 const q=fixture('navarra',{nsuma:40,cs:10,abstain:50});q.spa_nsuma_formed=true;
 failedPact(q);expect(stock(q,'nsuma','navarra')).toBeCloseTo(41,12);expect(total(q,'navarra')).toBeCloseTo(100,12);
 const absent=fixture('rest',{pp:50,abstain:50});failedPact(absent);
 expect(stock(absent,'cs','rest')).toBe(0);expect(total(absent,'rest')).toBe(100);
});
it('the channeling loss weakens subsequent monthly recruitment',()=>{
 const module={exports:{} as any};new Function('module',source.replace('engineTick: monthPasses,','engineTick: monthPasses, testCongreso: monthPassesCongreso,'))(module);
 const q=fixture();G.callCongresoRepeatElection(q);
 Object.assign(q,{gdp_growth:0,unemployment:20,welfare_index:50,social_dissent:50,cat_spa_relations:50,game_random_state:123});
 const control=structuredClone(q);control.podemos_channeling=.9;
 module.exports.testCongreso(q);module.exports.testCongreso(control);
 expect(stock(q,'podemos')).toBeLessThan(stock(control,'podemos'));
});
it.each(['congreso_coalition_psoe_cs.wtf','congreso_coalition.new_congreso_elections'])('applies repeat fatigue through the authored %s route',id=>{
 const a=DendryAdapter.fromJSONText(game);a.beginGame([1,2,3,4]);a.goToScene('root.start');
 const q=a.qualities as any;
 Object.assign(q,fixture(),{player_party:'erc',rubicon:true,congreso_left_carrier:'podemos',podemos_congreso_s:50,psoe_congreso_s:100,
  congreso_coalition_left_reach:false,congreso_coalition_change_reach:false,congreso_coalition_progressive_reach:false});
 a.goToScene(id);
 expect(q.next_spanish_election_time).toBe(46);expect(q.next_spanish_election_month).toBe(6);
 expect(stock(q,'podemos')).toBeCloseTo(16.8,12);expect(q.podemos_channeling).toBeCloseTo(.78,12);
});
