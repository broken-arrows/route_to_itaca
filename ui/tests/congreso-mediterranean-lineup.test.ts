import {it,expect} from 'vitest';
import {readFileSync} from 'node:fs';
import {DendryAdapter} from '../src/engine/adapter';

const game=readFileSync('../out/game.json','utf8');
it.each([
 {name:'neutral standalone before a broader pact',outreach:0,union:false,reform:false,left:25,own:5,independent:true},
 {name:'neutral agreement with IU',outreach:0,union:true,reform:false,left:25,own:5,independent:false},
 {name:'constitutional agreement',outreach:0,union:false,reform:true,left:25,own:5,independent:false},
 {name:'no viable coalition even with IU',outreach:0,union:true,reform:false,left:5,own:8,independent:true},
 {name:'favourable outreach before IU',outreach:-1,union:false,reform:false,left:25,own:5,independent:false},
 {name:'strong preference for autonomy',outreach:2,union:true,reform:true,left:25,own:5,independent:true},
 {name:'folded organization retains bargaining strength',outreach:0,union:true,reform:false,left:10,own:0,stashed:8,independent:true},
])('MÉS: $name',v=>{
 const compiled=JSON.parse(game);
 compiled.scenes['post_event.events_choice'].options=[];
 compiled.scenes['post_event.elections_choice'].options=[];
 const a=DendryAdapter.fromJSONText(JSON.stringify(compiled));a.beginGame([1,2,3,4]);a.goToScene('root.start');
 const q=a.qualities as any;
 Object.assign(q,{player_party:'erc',year:2016,month:4,month_actions:1,month_actions_last:0,
  podemos_surged:true,iu_in_up:v.union,constitution_reformed:v.reform,party_reaching_out_mes:v.outreach,
  spa_mes_active:v.own>0,spa_mes_active_applied:v.own>0,mes_congreso_balears_support:v.own,
  mes_congreso_balears_stashed:v.stashed||0,
  podemos_congreso_balears_support:v.union?0:v.left,up_congreso_balears_support:v.union?v.left:0,
  psoe_congreso_balears_support:30,next_spanish_election_time:q.time+3});
 a.goToScene('post_event');
 expect(q.spa_mes_active).toBe(v.independent);
 expect(q.mes_congreso_balears_support>0).toBe(v.independent);
});
