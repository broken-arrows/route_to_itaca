import { beforeAll, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { normalize, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { DendryAdapter } from '../src/engine/adapter';

const require = createRequire(import.meta.url);
const { compileGame } = require('dendrynexus-ten/lib/parsers/compiler.js');
const localization = require('dendrynexus-ten/lib/localization.js');
const repo = resolve(import.meta.dirname, '../..');
const fixtureGame = readFileSync(resolve(repo, 'out/game.json'), 'utf8');
let scenes: Record<string, any>;

beforeAll(async () => {
  const actions = [
    ['generalitat/generalitat_coalition_rupture', 'reshuffled'], ['generalitat/government_does_things', 'prepare'],
    ['events/elections/parlament_coalition_ciu_erc_cup_phase_erc_player_ciu_led_support', 'submit'],
    ['events/elections/parlament_coalition_setnames', ''], ['party_paths/ciu/pdcat_formation', ''],
    ['events/elections/parlament_coalition_resolve', ''],
    ['party_paths/ciu/junts_formation', ''],
  ];
  // Compile the actual authored actions without bringing unrelated navigation branches into these cases.
  const files = actions.map(([name, child], index) => {
    const text = readFileSync(resolve(repo, `source/scenes/${name}.scene.dry`), 'utf8');
    const sections = text.split(/^@([^\s]+)\s*$/m);
    const body = child ? sections[sections.indexOf(child) + 1] : sections[0];
    const action = body.match(/^on-arrival:\s*\{![\s\S]*?!\}/m)![0];
    return {name: normalize(`scenes/fixture_${index}.scene.dry`), contents: 'title: Fixture\n' + action + '\n\nStart.\n'};
  });
  const game = await new Promise<any>((accept, reject) => compileGame(files,
    (error: Error, result: any) => error ? reject(error) : accept(result)));
  scenes = Object.fromEntries(actions.map(([name, child], index) => [
    name.split('/').at(-1)! + (child ? '.' + child : ''), game.scenes[`fixture_${index}`],
  ]));
});

function initialized() {
  const adapter = DendryAdapter.fromJSONText(fixtureGame);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  Object.assign(adapter.qualities, { game_random_seed: 12345, game_random_state: 12345 });
  return adapter;
}

function arrival(adapter: DendryAdapter, scene: string) {
  const engine = adapter.engine as any;
  for (const action of scenes[scene].onArrival || []) action.call(engine, engine.state, adapter.qualities, engine.gameLib);
}

function recipe(adapter: DendryAdapter, field: string) {
  return (adapter.engine as any).gameLib.getText(adapter.qualities, field);
}

describe('scene-generated localization recipes', () => {
  it.each(['m', 'f'])('uses a native presidential title in English history for gender %s', gender => {
    const adapter = initialized(), q = adapter.qualities as any;
    Object.assign(q, {cat_coalition:['ciu'], cat_coalition_support:[], cat_coalition_abstain:[],
      parlament_coalition_proposal_president:'ciu', parlament_coalition_proposal_president_gender:gender,
      parlament_coalition_proposal_president_name:gender === 'f' ? 'Neus Munté' : 'Artur Mas'});
    for (const office of ['vp','economy','interior','health','education','territory','foreign','bnl']) {
      q['parlament_coalition_proposal_' + office] = 'ciu';
    }
    arrival(adapter, 'parlament_coalition_resolve');
    expect(q.parlament_coalition_resolution_errors).toEqual([]);
    const title = gender === 'f' ? 'Presidenta Neus Munté' : 'President Artur Mas';
    expect(q.historical_event).toContain(title);
    const captured = recipe(adapter, 'historical_event');
    q.president = 'Changed holder'; q.president_gender = gender === 'f' ? 'm' : 'f';
    expect(localization.render(captured, 'en', null)).toContain(title);
    expect(localization.render(captured, 'ca', {})).toContain(title);
  });

  it.each(['m', 'f'])('preserves service genders and uses the right health title for gender %s', gender => {
    const adapter = initialized(), engine = adapter.engine as any, q = adapter.qualities as any;
    Object.assign(q, {player_party:'erc', cat_coalition:['ciu','erc'], cat_caretaker_gov:false,
      npc_government_initiative:null, npc_government_response:null,
      health_minister:gender === 'f' ? 'Alba Vergés' : 'Boi Ruiz', health_minister_gender:gender,
      education_minister_gender:gender, healthcare_sexual_level:0, healthcare_spending:-1,
      ciu_ideology:'Social Democracy, Pact-Pragmatism'});
    for (const office of ['president','vicepresident','economy_minister','education_minister',
      'bnl_minister','territory_minister','foreign_minister','interior_minister']) q[office + '_party'] = 'erc';
    q.health_minister_party = 'ciu';
    const random = vi.spyOn(engine.gameLib, 'random').mockReturnValue(0);
    const text = vi.spyOn(engine.gameLib, 'text');
    try {
      arrival(adapter, 'government_does_things.prepare');
      expect(q.health_minister_gender).toBe(gender);
      expect(q.education_minister_gender).toBe(gender);
      const title = gender === 'f' ? 'Consellera' : 'Conseller';
      expect(text.mock.calls.some(([key, params]: any[]) => key === title + ' of {department}{minister}'
        && params.minister === q.health_minister && params.minister_gender === gender)).toBe(true);
      expect(q.npc_government_initiative.field).toBe('healthcare_sexual_level');
      expect(q.npc_government_reason).toContain(title + ' of Health ' + q.health_minister);
      const captured = recipe(adapter, 'npc_government_reason');
      q.health_minister = 'Changed holder'; q.health_minister_gender = gender === 'f' ? 'm' : 'f';
      expect(localization.render(captured, 'en', null)).toContain(title + ' of Health');
      expect(localization.render(captured, 'ca', {})).toContain(title + ' of Health');
    } finally {
      random.mockRestore(); text.mockRestore();
    }
  });

  it('captures neutral-English historical name/gender and a nested government independently of live Q', () => {
    const adapter = initialized(), engine = adapter.engine as any, q = adapter.qualities as any, G = engine.gameLib;
    const offices = {president:'president', vp:'vicepresident', economy:'economy_minister', interior:'interior_minister',
      health:'health_minister', education:'education_minister', territory:'territory_minister', foreign:'foreign_minister', bnl:'bnl_minister'};
    q.coalition_rupture_preserved_offices = Object.fromEntries(Object.keys(offices).map(key => [key,
      {party:'erc', name:key === 'president' ? 'Neus Munté' : 'Oriol Junqueras', gender:key === 'president' ? 'f' : 'm'}]));
    q.coalition_rupture_removed = 'ciu';
    G.setText(q, 'cat_coalition_disp', G.text('minority government'));
    arrival(adapter, 'generalitat_coalition_rupture.reshuffled');
    const captured = recipe(adapter, 'historical_event');
    expect(captured.params.president_gender).toBe('f');
    expect(q.historical_event).toBe('ciu left the Generalitat government after a coalition rupture. Neus Munté continues to lead minority government.');
    q.president = 'Artur Mas'; q.president_gender = 'm';
    const key = '{coalition_rupture_removed} left the Generalitat government after a coalition rupture. {president} continues to lead {cat_coalition_disp}.';
    const catalog = { 'minority government':'govern minoritari', [key]: {select:'president_gender', cases:{
      f:'{coalition_rupture_removed} ha marxat. La presidenta {president} encapçala el {cat_coalition_disp}.',
      other:'{coalition_rupture_removed} ha marxat. {president} encapçala el {cat_coalition_disp}.',
    }} };
    expect(localization.render(captured, 'ca', catalog)).toBe('ciu ha marxat. La presidenta Neus Munté encapçala el govern minoritari.');
    expect(q.historical_event).toContain('Neus Munté');
  });

  it('keeps native English reason dedup/eligibility while translating the retained list', () => {
    const adapter = initialized(), engine = adapter.engine as any, q = adapter.qualities as any;
    const reason = 'Some cabinet posts are still unassigned or have been given to a party outside the government.';
    const catalog = {[reason]:'Alguns ministeris encara no estan assignats.'};
    engine.setLocale('ca', catalog);
    Object.assign(q, {parlament_coalition_sovereignty_cabinet:['ciu','erc'], parlament_current_ciu:'ciu',
      player_party:'erc', coalition_phase_timer:0, parlament_coalition_leverage:0, parlament_coalition_sovereignty_phase:3});
    for (const office of ['vp','interior','economy','foreign','health','education','territory','bnl']) q['parlament_coalition_proposal_' + office] = 'outside';
    arrival(adapter, 'parlament_coalition_ciu_erc_cup_phase_erc_player_ciu_led_support.submit');
    expect(q.parlament_coalition_sovereignty_reject_reasons.split(reason)).toHaveLength(2);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    const output = localization.render(recipe(adapter, 'parlament_coalition_sovereignty_reject_reasons'), 'ca', catalog);
    expect(output).toContain('Alguns ministeris encara no estan assignats.');
    expect(q.parlament_coalition_sovereignty_reject_reasons).not.toContain('Alguns');
  });

  it('retains proper names as literals and marks only unassigned cabinet prose', () => {
    const adapter = initialized(), q = adapter.qualities as any;
    for (const office of ['vp','economy','interior','foreign','health','education','territory','bnl']) q['parlament_coalition_proposal_' + office] = null;
    q.parlament_coalition_proposal_president = 'ciu';
    arrival(adapter, 'parlament_coalition_setnames');
    expect(recipe(adapter, 'parlament_coalition_proposal_economy_name').kind).toBe('text');
    expect(recipe(adapter, 'parlament_coalition_proposal_economy_name').key).toContain('To be filled by coalition partners');
    expect(recipe(adapter, 'parlament_coalition_proposal_president_name').kind).toBe('literal');
    expect(q.parlament_coalition_proposal_president_name).toBe('Artur Mas');
  });

  it('updates current recipe parameters through exact-case rebrands without changing template keys', () => {
    const adapter = initialized(), engine = adapter.engine as any, q = adapter.qualities as any, G = engine.gameLib;
    q.cat_coalition = ['dl'];
    const key = '<span style="color: var(--{brand});">{name}</span>';
    G.setText(q, 'cat_coalition_disp', G.text(key, {brand:'dl', name:'DL dl'}));
    arrival(adapter, 'pdcat_formation');
    expect(q.cat_coalition_disp).toBe('<span style="color: var(--pdcat);">DL pdcat</span>');
    expect(recipe(adapter, 'cat_coalition_disp').key).toBe(key);
    expect(recipe(adapter, 'cat_coalition_disp').params).toEqual({brand:'pdcat', name:'DL pdcat'});
  });

  it('preserves legacy fixed-key and recipe-less English rebrand behavior', () => {
    const adapter = initialized(), engine = adapter.engine as any, q = adapter.qualities as any, G = engine.gameLib;
    q.cat_coalition = ['dl'];
    G.setText(q, 'cat_coalition_disp', G.text('<span style="color: var(--dl);">DL dl</span>'));
    arrival(adapter, 'pdcat_formation');
    expect(q.cat_coalition_disp).toBe('<span style="color: var(--pdcat);">DL pdcat</span>');
    expect(recipe(adapter, 'cat_coalition_disp').kind).toBe('literal');
    q.cat_coalition = ['jxcat']; q.cat_coalition_disp = 'JXCAT JxCat jxcat';
    arrival(adapter, 'junts_formation');
    expect(q.cat_coalition_disp).toBe('JXCAT junts junts');
  });

  it('captures selected NPC initiative context outside the persisted English candidate', () => {
    const adapter = initialized(), engine = adapter.engine as any, q = adapter.qualities as any;
    Object.assign(q, {player_party:'erc', cat_coalition:['ciu'], president_party:'ciu', cat_caretaker_gov:false,
      npc_government_initiative:null, npc_government_response:null, game_random_seed:12345, game_random_state:12345});
    for (const office of ['vicepresident','economy_minister','health_minister','education_minister','bnl_minister','territory_minister','foreign_minister','interior_minister']) q[office + '_party'] = 'ciu';
    arrival(adapter, 'government_does_things.prepare');
    const initiative = q.npc_government_initiative;
    expect(initiative).toBeTruthy();
    expect(typeof initiative.title).toBe('string');
    expect(typeof initiative.reason).toBe('string');
    expect(initiative).not.toHaveProperty('text');
    const captured = recipe(adapter, 'npc_government_reason');
    expect(captured.kind).toBe('text');
    expect(captured.params.economy_minister_gender).toBe('m');
    expect(captured.params.economy_minister).toBe('Andreu Mas-Colell');
    q.economy_minister = 'New minister'; q.economy_minister_gender = 'f';
    const qBefore = JSON.stringify(q), rng = engine.random.getState(), actions = vi.spyOn(engine, '_runActions');
    const catalog = {[captured.key]:{select:'economy_minister_gender',cases:{
      m:'El conseller {economy_minister} proposa una iniciativa.', other:'La consellera {economy_minister} proposa una iniciativa.',
    }}};
    engine.setLocale('ca', catalog);
    expect(localization.render(captured, 'ca', catalog)).toBe('El conseller Andreu Mas-Colell proposa una iniciativa.');
    expect(JSON.stringify(q)).toBe(qBefore);
    expect(engine.random.getState()).toEqual(rng);
    expect(actions).not.toHaveBeenCalled();
    expect(q.npc_government_reason).toContain('Andreu Mas-Colell');
  });
});
