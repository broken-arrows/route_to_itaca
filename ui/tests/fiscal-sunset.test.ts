import { it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

const root = resolve(__dirname, '../..');
const source = readFileSync(resolve(root,'source/lib/cat_engine.js'),'utf8');
const game = readFileSync(resolve(root,'out/game.json'),'utf8');
const patch = (s:string, before:string, after:string) => {
  expect(s.split(before).length).toBe(2);
  return s.replace(before,after);
};
const load = (s:string) => {
  const module = {exports:{} as any};
  new Function('module',s)(module);
  return module.exports;
};

it('expiry retains accumulated surplus and setpolicy reinstates expired tax and salary modifiers', () => {
  const adapter = DendryAdapter.fromJSONText(game);
  adapter.beginGame([1,2,3,4]);
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string,any>;
  q.public_salary_level = -1;
  adapter.goToScene('generalitat_setpolicy');
  // Matched live engine ticks: the accumulated difference must survive expiry.
  const exposed = load(patch(source,'    engineTick: monthPasses,','    engineTick: monthPasses, advanceMods, mod,'));
  const treated = JSON.parse(JSON.stringify(q));
  const control = JSON.parse(JSON.stringify(q));
  exposed.registerLaw(treated,{id:'expiry_probe',targets:{generalitat_surplus:.1},expires:{year:q.year,month:q.month+2}});
  for (let tick=0;tick<3;tick++) {
    treated.month++; control.month++;
    exposed.engineTick(treated); exposed.engineTick(control);
    expect(treated.generalitat_surplus-control.generalitat_surplus).toBeCloseTo(.1,10);
  }
  expect(treated.active_mods.expiry_probe.active).toBe(false);
  for(const id of ['regional_taxes_high','public_salary_cuts','education_cuts','healthcare_cuts']) {
    q.active_mods[id].def.expires = {year:q.year,month:q.month};
  }
  exposed.advanceMods(q);
  for(const id of ['regional_taxes_high','public_salary_cuts','education_cuts','healthcare_cuts']) expect(q.active_mods[id].active).toBe(false);
  adapter.goToScene('generalitat_setpolicy');
  expect(q.active_mods.regional_taxes_high.active).toBe(true);
  expect(q.active_mods.public_salary_cuts.active).toBe(true);
  expect(q.active_mods.education_cuts.active).toBe(false);
  expect(q.active_mods.healthcare_cuts.active).toBe(false);
});
