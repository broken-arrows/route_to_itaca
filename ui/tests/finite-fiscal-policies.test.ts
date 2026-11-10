import { describe, it, expect } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import ts from 'typescript';
import { resolve } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const registerRuntimeLaw = (gameLib as unknown as {
  registerLaw: (q: Record<string,any>, def: Record<string,any>) => void;
}).registerLaw;

const root = resolve(__dirname, '../..');
const source = readFileSync(resolve(root, 'source/lib/cat_engine.js'), 'utf8');
const module = {exports: {} as any};
// Exercise actual settlement/lifecycle without unrelated macro drift in unit cases.
new Function('module', source.replace('    engineTick: monthPasses,',
  '    engineTick: monthPasses, applyFiscalAdjustments, advanceMods, mod,'))(module);
const laws = module.exports;
const tick = (q:any) => { laws.advanceMods(q); laws.applyFiscalAdjustments(q); };

function neutralPolicyFixture() {
  const adapter=DendryAdapter.fromJSONText(readFileSync(resolve(root,'out/game.json'),'utf8'));
  adapter.beginGame([1,2,3,4]); adapter.goToScene('root.start');
  const q=adapter.qualities as Record<string,any>;
  Object.assign(q,{education_spending:0,healthcare_spending:0,public_salary_level:0,tax_policy_level:0});
  adapter.goToScene('generalitat_setpolicy');
  q.generalitat_surplus=-2;
  return {adapter,q};
}

describe('finite fiscal policy adjustments', () => {
  it.each([[false,1.2],[true,.96]])('settles education privatization and reverses it, ruling=%s', (ruling,total) => {
    const {adapter,q}=neutralPolicyFixture();
    q.education_spending=-2; adapter.goToScene('generalitat_setpolicy');
    for(let i=0;i<24;i++) tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(-2+.72,10);
    q.education_spending=-3; q.education_privatized_ruling=ruling;
    adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBeCloseTo(-2,10);
    for(let i=0;i<36;i++) tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(-2+Number(total),10);
    q.education_spending=0; adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBeCloseTo(-2,10);
  });

  it.each([3,24,36])('reconciles curriculum focus changes at age %s without restarting benefits', age => {
    const {adapter,q}=neutralPolicyFixture();
    q.education_policy_main='language'; q.education_policy_secondary='democratic';
    adapter.goToScene('generalitat_setpolicy');
    for(let i=0;i<age;i++) tick(q);
    const maturity=age===3?.5:18.5;
    expect(q.generalitat_surplus).toBeCloseTo(-2-.015*maturity,10);
    q.active_mods=JSON.parse(JSON.stringify(q.active_mods));
    q.active_mods.education_school_policy.def.electoral_transfers=[{marker:'preserved'}];
    for(let cycle=0;cycle<3;cycle++) {
      q.education_policy_main='segregation'; q.education_policy_secondary='science';
      adapter.goToScene('generalitat_setpolicy');
      expect(q.generalitat_surplus).toBeCloseTo(-2-.04*maturity,10);
      const policy=q.active_mods.education_school_policy;
      expect(policy.ticks_active).toBe(age);
      expect(policy.def.electoral_transfers).toEqual([{marker:'preserved'}]);
      expect(policy.live_effect.welfare_index_growth).toBeCloseTo(.1*Math.min(age/12,1));
      expect(policy.live_effect.generalitat_surplus).toBe(0);
      adapter.goToScene('generalitat_setpolicy');
      expect(q.generalitat_surplus).toBeCloseTo(-2-.04*maturity,10);
      q.education_policy_main='language'; q.education_policy_secondary='democratic';
      adapter.goToScene('generalitat_setpolicy');
      expect(q.generalitat_surplus).toBeCloseTo(-2-.015*maturity,10);
    }
    for(let i=age;i<40;i++) tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(-2-.2775,10);
    expect(q.active_mods.education_school_policy.active).toBe(true);
    expect(laws.mod(q,'welfare_index_growth')).toBeCloseTo(.075,10);
    laws.deactivateLaw(q,'education_school_policy');
    expect(q.generalitat_surplus).toBeCloseTo(-2,10);
  });

  it('records only payable curriculum costs at a bound, including a subsequent cheaper switch', () => {
    const {adapter,q}=neutralPolicyFixture();
    q.generalitat_surplus=-4.9;
    q.education_policy_main='language'; q.education_policy_secondary='democratic';
    adapter.goToScene('generalitat_setpolicy');
    for(let i=0;i<30;i++) tick(q);
    expect(q.active_mods.education_school_policy.fiscal_applied).toBeCloseTo(-.1,10);
    q.education_policy_main='segregation'; q.education_policy_secondary='science';
    adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBe(-5);
    expect(q.active_mods.education_school_policy.fiscal_applied).toBeCloseTo(-.1,10);
    // Unrelated fiscal improvement does not turn the clipped amount into a refund.
    q.generalitat_surplus=-3;
    q.education_policy_main='language'; q.education_policy_secondary='democratic';
    adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBeCloseTo(-2.9-.2775,10);
    laws.deactivateLaw(q,'education_school_policy');
    expect(q.generalitat_surplus).toBeCloseTo(-2.9,10);
  });

  it('keeps legacy saved curriculum accounting when its focuses change', () => {
    const {adapter,q}=neutralPolicyFixture();
    laws.registerLaw(q,{id:'education_school_policy',main_focus:'language',secondary_focus:'democratic',
      ramp_ticks:12,targets:{generalitat_surplus:-.015}});
    q.active_mods.education_school_policy.ticks_active=30;
    q.education_policy_main='segregation'; q.education_policy_secondary='science';
    adapter.goToScene('generalitat_setpolicy');
    expect(q.active_mods.education_school_policy.def.fiscal_ticks).toBeUndefined();
    expect(q.generalitat_surplus).toBe(-2);
    expect(laws.mod(q,'generalitat_surplus')).toBeCloseTo(-.04,10);
  });

  it('bounds enacted housing expansion at one point while keeping the stock and benefits', () => {
    const {adapter,q}=neutralPolicyFixture();
    Object.assign(q,{player_party:'erc',player_in_jxsi:false,player_in_jxcat:false,
      parlament_current_icv:'icv',party_resources:20,coming_from_parlament:true});
    for(const party of q.parties) q[party+'_parlament_s']=0;
    q.erc_parlament_s=70; q.ppc_parlament_s=65;
    adapter.goToScene('parlament_housing');
    adapter.goToScene('parlament_housing.housing_public_stock');
    adapter.goToScene('parlament_housing.housing_vote');
    expect(q.parlament_housing_passed).toBe(true);
    expect(q.active_mods.housing_public_stock.def.fiscal_ticks).toBe(24);
    const opening=q.generalitat_surplus;
    for(let i=0;i<84;i++) tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(opening-1,10);
    expect(q.active_mods.housing_public_stock.active).toBe(true);
    expect(laws.mod(q,'welfare_index_growth')).toBeCloseTo(.14,10);
    expect(q.active_mods.housing_public_stock.live_effect.generalitat_surplus).toBe(0);
  });

  it('bounds successive extraordinary cuts and refunds only the round repealed through Economy', () => {
    const {adapter,q}=neutralPolicyFixture();
    Object.assign(q,{player_party:'erc',player_in_jxsi:false,player_in_jxcat:false,
      cat_coalition:['erc','ciu'],cat_caretaker_gov:false,economy_minister_party:'erc',
      government_budget:10,economy_cuts:0,coalition_affairs_concession_target:'',
      education_spending:-2,healthcare_spending:-2,public_salary_level:-2});
    adapter.goToScene('generalitat_coalition_card.concessions_ciu');
    adapter.goToScene('generalitat_coalition_card.concessions_ciu');
    expect(q.economy_cuts).toBe(2);
    // Isolate the two authored rounds from ordinary service reconciliation.
    q.active_mods=Object.fromEntries(Object.entries(q.active_mods).filter(([id])=>id.startsWith('general_cuts_')));
    q.generalitat_surplus=-2;
    for(let i=0;i<36;i++) tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(-.8,10);
    adapter.goToScene('generalitat_economy_card.revert_cuts');
    expect(q.economy_cuts).toBe(1);
    expect(q.generalitat_surplus).toBeCloseTo(-1.4,10);
    expect(q.active_mods.general_cuts_1.active).toBe(true);
    expect(q.active_mods.general_cuts_2.active).toBe(false);
  });

  it('settles and reverses every independently evaluable authored fiscal definition', () => {
    const game=JSON.parse(readFileSync(resolve(root,'out/game.json'),'utf8'));
    const inventory:any[]=[];
    for(const [sceneId,scene] of Object.entries(game.scenes) as [string,any][]) {
      for(const action of scene.onArrival||[]) {
        if(!action.$code) continue;
        const ast=ts.createSourceFile('action.js',action.$code,ts.ScriptTarget.Latest,true,ts.ScriptKind.JS);
        const visit=(node:ts.Node) => {
          if(ts.isCallExpression(node) && node.expression.getText(ast)==='G.registerLaw' && node.arguments[1]) {
            const expression=node.arguments[1].getText(ast);
            let def:any;
            try {def=new Function('protective','return ('+expression+');')(true);}
            catch {inventory.push({scene:sceneId,dynamic:true,expression});}
            if(def?.targets?.generalitat_surplus!=null) {
              const rate=def.targets.generalitat_surplus;
              const duration=def.fiscal_ticks;
              const expected=duration ? Array.from({length:duration},(_,i)=>rate*(def.ramp_ticks?Math.min((i+1)/def.ramp_ticks,1):1)).reduce((a,b)=>a+b,0):null;
              inventory.push({scene:sceneId,id:def.id,rate,ramp:def.ramp_ticks||0,duration:duration||null,total:expected});
              if(duration) {
                // Start at the opposite bound to give each definition the full seven-point room.
                const initial=rate>0?-5:2;
                const q:any={generalitat_surplus:initial,year:2012,month:8};
                laws.registerLaw(q,def);
                for(let i=0;i<duration+2;i++) tick(q);
                const realized=Math.max(-5,Math.min(2,initial+expected!))-initial;
                expect(q.generalitat_surplus,def.id).toBeCloseTo(initial+realized,10);
                expect(q.active_mods[def.id].fiscal_applied,def.id).toBeCloseTo(realized,10);
                expect(q.active_mods[def.id].live_effect.generalitat_surplus,def.id).toBe(0);
                laws.deactivateLaw(q,def.id);
                expect(q.generalitat_surplus,def.id).toBeCloseTo(initial,10);
              }
            }
          }
          ts.forEachChild(node,visit);
        };
        visit(ast);
      }
    }
    expect(inventory.filter(row=>row.duration).length).toBeGreaterThan(40);
    if(process.env.FISCAL_INVENTORY) writeFileSync(resolve(root,'docs/design/fiscal_rollout_inventory_2026_11_04.json'),JSON.stringify(inventory,null,2));
  });

  it('real railway completion retains construction costs and begins finite operating adjustment', () => {
    const adapter=DendryAdapter.fromJSONText(readFileSync(resolve(root,'out/game.json'),'utf8'));
    adapter.beginGame([1,2,3,4]); adapter.goToScene('root.start');
    const q=adapter.qualities as Record<string,any>;
    q.active_mods={}; q.generalitat_surplus=-2.06;
    registerRuntimeLaw(q,{id:'territory_rail1',targets:{generalitat_surplus:-.005}});
    q.territory_rail1_trigger=true;
    let transitionBalance:number|undefined;
    adapter.engine.setGameLib({...gameLib,registerLaw:(state:any,def:any)=>{
      if(def.id==='territory_rail1') transitionBalance=state.generalitat_surplus;
      return registerRuntimeLaw(state,def);
    }});
    adapter.goToScene('post_event');
    expect(transitionBalance).toBeCloseTo(-2.06,10);
    expect(q.active_mods.territory_rail1.def.fiscal_ticks).toBe(24);
    const isolated:any={year:2014,month:1,generalitat_surplus:transitionBalance};
    laws.registerLaw(isolated,q.active_mods.territory_rail1.def);
    for(let i=0;i<30;i++) tick(isolated);
    expect(isolated.generalitat_surplus).toBeCloseTo(-2.18,10);
    expect(isolated.active_mods.territory_rail1.active).toBe(true);
  });

  it('stops only fiscal increments, preserves the benefit, then reverses it once', () => {
    const q:any = {generalitat_surplus:-2,year:2012,month:8};
    laws.registerLaw(q,{id:'cut',fiscal_ticks:12,targets:{generalitat_surplus:.01,welfare_index_growth:-.2}});
    for(let i=0;i<15;i++) tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(-1.88,10);
    expect(q.active_mods.cut.active).toBe(true);
    expect(q.active_mods.cut.live_effect.generalitat_surplus).toBe(0);
    expect(laws.mod(q,'welfare_index_growth')).toBe(-.2);
    laws.deactivateLaw(q,'cut');
    expect(q.generalitat_surplus).toBeCloseTo(-2,10);
    laws.deactivateLaw(q,'cut');
    expect(q.generalitat_surplus).toBeCloseTo(-2,10);
  });

  it('reverses partial ramped contributions after save/load and handles replacement', () => {
    let q:any={generalitat_surplus:-2,year:2012,month:8};
    const def={id:'tax',fiscal_ticks:12,ramp_ticks:4,targets:{generalitat_surplus:.04}};
    laws.registerLaw(q,def); tick(q); tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(-1.97,10);
    q=JSON.parse(JSON.stringify(q));
    laws.registerLaw(q,def);
    expect(q.generalitat_surplus).toBeCloseTo(-2,10);
    tick(q); laws.deactivateLaw(q,'tax');
    expect(q.generalitat_surplus).toBeCloseTo(-2,10);
  });

  it.each([[1.99,.1],[-4.99,-.1]])('records only realized contributions at balance bound %s', (balance,amount) => {
    const q:any={generalitat_surplus:balance,year:2012,month:8};
    for(let cycle=0;cycle<3;cycle++) {
      laws.registerLaw(q,{id:'bounded',fiscal_ticks:12,targets:{generalitat_surplus:amount}});
      for(let i=0;i<14;i++) tick(q);
      expect(q.active_mods.bounded.fiscal_applied).toBeCloseTo(Math.sign(amount)*.01,10);
      laws.deactivateLaw(q,'bounded');
      expect(q.generalitat_surplus).toBeCloseTo(balance,10);
    }
  });

  it('retains existing behavior for laws without the opt-in', () => {
    const q:any={generalitat_surplus:-2,year:2012,month:8};
    laws.registerLaw(q,{id:'legacy',targets:{generalitat_surplus:.02}});
    for(let i=0;i<20;i++) tick(q);
    expect(laws.mod(q,'generalitat_surplus')).toBe(.02);
    laws.deactivateLaw(q,'legacy');
    expect(q.generalitat_surplus).toBe(-2);
  });

  it('authored policy reconciliation preserves completed adjustments and restoration reverses them', () => {
    const adapter=DendryAdapter.fromJSONText(readFileSync(resolve(root,'out/game.json'),'utf8'));
    adapter.beginGame([1,2,3,4]); adapter.goToScene('root.start');
    const q=adapter.qualities as Record<string,any>;
    q.public_salary_level=-1;
    adapter.goToScene('generalitat_setpolicy');
    const opening=q.generalitat_surplus;
    for(const id of ['regional_taxes_high','education_cuts','healthcare_cuts','public_salary_cuts'])
      expect(q.active_mods[id].def.fiscal_ticks).toBe(12);
    for(let i=0;i<15;i++) { tick(q); adapter.goToScene('generalitat_setpolicy'); }
    expect(q.generalitat_surplus).toBeCloseTo(opening+.6,10);
    expect(q.active_mods.regional_taxes_high.ticks_active).toBe(15);
    q.education_spending=0; q.healthcare_spending=0; q.public_salary_level=0;
    adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBeCloseTo(opening+.24,10);
    q.tax_policy_level=0;
    adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBeCloseTo(opening,10);
    // A partial cut/restore cycle cannot add permanent surplus.
    q.education_spending=-1; adapter.goToScene('generalitat_setpolicy'); tick(q);
    q.education_spending=0; adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBeCloseTo(opening,10);
    // A negative adjustment is reversed when the expansion is withdrawn too.
    q.public_salary_level=1; adapter.goToScene('generalitat_setpolicy');
    for(let i=0;i<13;i++) tick(q);
    expect(q.generalitat_surplus).toBeCloseTo(opening-.24,10);
    q.public_salary_level=0; adapter.goToScene('generalitat_setpolicy');
    expect(q.generalitat_surplus).toBeCloseTo(opening,10);
  });
});
