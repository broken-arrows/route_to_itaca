import { it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(__dirname, '../..');
const read = (p: string) => readFileSync(resolve(root, p), 'utf8');
const source = read('source/lib/cat_engine.js');

// Exercise the real monthly function; expose it only in this in-memory test copy.
// Observe raw deltas before normalization can hide a missing debit or credit.
function phase(region: string, support: Record<string, number>, flags: object = {}) {
  const captures: Record<string, Record<string, number>> = {};
  let code = source.replace('engineTick: monthPasses,', 'engineTick: monthPasses, testCongreso: monthPassesCongreso,');
  expect(code).not.toBe(source);
  for (const name of ['4b.', '4c.', '4d.', '4e.', '4h.', '4i.']) {
    const marker = `      // ${name}`;
    expect(code.split(marker)).toHaveLength(2);
    code = code.replace(marker, `      capture(${JSON.stringify(name)}, deltas);\n${marker}`);
  }
  const module = {exports: {} as any};
  new Function('module', 'capture', code)(module, (name: string, deltas: any) => {captures[name] = {...deltas};});
  const parties = ['pp','psoe','podemos','up','cs','upn','nsuma','erc','ciu','dl','upyd'];
  const q: any = {congreso_constituencies:[region], ['congreso_parties_'+region]:parties,
    gdp_growth:0, unemployment:20, welfare_index:50, social_dissent:50,
    cat_spa_relations:49, _prev_spa_cat_spa:50, corruption_pp:100, corruption_psoe:100,
    corruption_pp_decay:1, corruption_psoe_decay:1, game_random_state:123,
    upn_in_pp:true, ...flags};
  for (const [p,v] of Object.entries(support)) q[p+'_congreso_'+region+'_support']=v;
  module.exports.testCongreso(q);
  const diff = (a: string, b: string) => Object.fromEntries([...new Set([...Object.keys(captures[a]),...Object.keys(captures[b])])]
    .map(p => [p,(captures[b][p]||0)-(captures[a][p]||0)]));
  return {pp:diff('4b.','4c.'),psoe:diff('4c.','4d.'),territory:diff('4d.','4e.'),upyd:diff('4h.','4i.')};
}
const conserved = (values: Record<string,number>) => expect(Object.values(values).reduce((a,b)=>a+b,0)).toBeCloseTo(0,12);

it('retains corruption losses intended for parties that have not emerged', () => {
  const r=phase('valencia',{pp:40,psoe:30,abstain:30});
  expect(r.pp.pp).toBeCloseTo(-.65,12);
  expect(r.psoe.psoe).toBeCloseTo(-.031,12);
  conserved(r.pp); conserved(r.psoe);
});
it('retains the abstention portion when abstention is unavailable', () => {
  const r=phase('valencia',{pp:30,psoe:30,podemos:20,cs:20});
  expect(r.pp.pp).toBeCloseTo(-.075,12);
  expect(r.psoe.psoe).toBeCloseTo(-.07,12);
  conserved(r.pp); conserved(r.psoe);
});
it('preserves full corruption transfers when every recipient exists', () => {
  const r=phase('valencia',{pp:30,psoe:25,podemos:15,cs:10,abstain:20});
  expect(r.pp).toMatchObject({pp:-.725,cs:.075,abstain:.65});
  expect(r.psoe.psoe).toBeCloseTo(-.101,12);
  conserved(r.pp); conserved(r.psoe);
});
it.each([false,true])('preserves UPN corruption routing with Cs present=%s', cs => {
  const r=phase('navarra',{pp:30,upn:20,psoe:20,abstain:cs?20:30,...(cs?{cs:10}:{})},{upn_in_pp:false});
  expect(r.pp.upn).toBeCloseTo(.041*(cs?.7:1),12);
  expect(r.pp.cs||0).toBeCloseTo(cs?.041*.3:0,12);
  conserved(r.pp);
});
it.each([{pp:40,abstain:60},{psoe:40,abstain:60},{podemos:40,abstain:60},{cs:40,abstain:60}])(
  'does not create one-sided territorial transfers with lineup %j', support => {
    const r=phase('valencia',support);
    expect(Object.values(r.territory).every(v=>Math.abs(v)<1e-12)).toBe(true);
  });
it('preserves paired territorial transfers when both endpoints exist', () => {
  const r=phase('valencia',{pp:30,psoe:25,podemos:15,cs:10,abstain:20});
  expect(r.territory.pp).toBeCloseTo(.020,12);
  expect(r.territory.psoe).toBeCloseTo(-.020,12);
  expect(r.territory.cs).toBeCloseTo(.020,12);
  expect(r.territory.podemos).toBeCloseTo(-.020,12);
  conserved(r.territory);
});
it('only charges Catalan donors for available recipients', () => {
  const r=phase('catalunya',{pp:20,psoe:20,erc:10,abstain:50});
  const gain=.02975*.65;
  expect(r.territory.erc).toBeCloseTo(gain,12);
  expect(r.territory.pp).toBeCloseTo(-gain*.4,12);
  expect(r.territory.psoe).toBeCloseTo(-gain*.6,12);
  conserved(r.territory);
});
it('only funds Catalan recipients from available donors', () => {
  const r=phase('catalunya',{psoe:30,cs:10,erc:5,ciu:5,abstain:50});
  expect(r.territory.cs).toBeCloseTo(.030*.6,12);
  expect(r.territory.erc).toBeCloseTo(.02975*.65*.6,12);
  expect(r.territory.ciu).toBeCloseTo(.02975*.35*.6,12);
  conserved(r.territory);
});
it('retains UPyD voters when neither collapse destination is available', () => {
  const r=phase('valencia',{pp:50,upyd:50});
  expect(r.upyd.upyd).toBeCloseTo(0,12);
  conserved(r.upyd);
});
it.each([
  [{pp:40,upyd:10,abstain:50},0,.35],
  [{pp:40,upyd:10,cs:50},.35,0],
  [{pp:30,upyd:10,cs:10,abstain:50},.175,.175],
] as const)('preserves the existing UPyD fallback with lineup %j', (support, cs, abstain) => {
  const r=phase('valencia',support);
  expect(r.upyd.cs||0).toBeCloseTo(cs,12);
  expect(r.upyd.abstain||0).toBeCloseTo(abstain,12);
  conserved(r.upyd);
});
it('preserves all Catalan territorial shares with all endpoints present', () => {
  const r=phase('catalunya',{pp:20,psoe:20,cs:10,erc:5,ciu:5,abstain:40});
  expect(r.territory.pp).toBeCloseTo(-(.02975+.03)*.4,12);
  expect(r.territory.psoe).toBeCloseTo(-(.02975+.03)*.6,12);
  expect(r.territory.erc).toBeCloseTo(.02975*.65,12);
  expect(r.territory.ciu).toBeCloseTo(.02975*.35,12);
  expect(r.territory.cs).toBeCloseTo(.03,12);
  conserved(r.territory);
});
