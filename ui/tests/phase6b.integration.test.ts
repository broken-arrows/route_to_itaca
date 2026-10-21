import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createPinia, setActivePinia } from 'pinia';
import { mount } from '@vue/test-utils';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { DendryAdapter } from '../src/engine/adapter';
import { parseContentCatalog, clearContentCatalogCacheForTest } from '../src/locales/content';
import { useGameStore } from '../src/stores/game';
import Coalitions from '../src/components/viz/Coalitions.vue';
import GlossaryTerm from '../src/components/GlossaryTerm.vue';
import { i18n } from '../src/i18n';
import { extractFiles } from 'dendrynexus-ten/lib/localization-extract.js';

const rootDir = resolve(__dirname, '../..');
const source = (path: string) => readFileSync(join(rootDir, 'source', path), 'utf8');
const declarations = JSON.parse(source('data/localization.json'));
// Test the localization contract with stable wording; editorial catalogs change independently.
const catalan = parseContentCatalog(JSON.parse(readFileSync(
  join(import.meta.dirname, 'fixtures/localization-contract.ca.json'), 'utf8',
)));
const root = `title: Test start
role: setup
new-page: true
on-arrival: {!
Object.assign(Q, {
  president: "Artur Mas", president_gender: "m", president_party: "ciu", player_party: "erc", erc_in_gob: true,
  month: 12, year: 2012, time: 0, week: 0, parlament_snap_election: 1,
  countdown_parlament_snap_election_trigger: true, cat_coalition: ["ciu", "erc"],
  ciu_parlament_s: 50, erc_parlament_s: 21, parlament_s_majority: 68,
  coalition_policy_commitments: [], countdowns: ["burdens_start"], snap_budget_distrust: true,
  budget_failed_times: 2, arrivals: 0
});
G.setText(Q, "cat_coalition_disp", "{party} {government}", {
  party: "CiU", government: G.governmentTooltipText("generalitat", Q.cat_coalition,
    G.text("Minority government - {seats}/{majority}", {seats:50,majority:68}),
    G.text("<span style='color: var(--ciu);'>minority government</span>"))
});
!}

Earlier setup prose.

- @snap_election: Announce
`;
const files = [
  { name: join('scenes', 'root.scene.dry'), contents: root },
  { name: join('scenes', 'snap_election.scene.dry'), contents: source('scenes/events/elections/snap_election.scene.dry') },
  { name: join('qdisplays', 'month.qdisplay.dry'), contents: source('qdisplays/month.qdisplay.dry') },
  { name: join('scenes', 'post_event.scene.dry'), contents: 'title: Continue\n\nFinished.\n' },
  { name: join('scenes', 'library.scene.dry'), contents: 'title: Library\nis-special: true\nnew-page: true\n\nLibrary content.\n\n- @backSpecialScene: Return\n' },
];
function compile(filesToCompile = files): Promise<string> {
  return new Promise((accept, reject) => compileGame(filesToCompile, (error, game) => {
    if (error) return reject(error);
    game.info = { ifid: 'phase6b-integration', version: '1', languages: ['en', 'ca'] };
    game.data = {
      localization: declarations,
      achievements: { achievements: [{id:'complete',name:'El Procés',description:'Finish the game.',image:'img/flags/estelada.png',stars:1}] },
      glossary: { terms: [{id:'test',match:['Test'],display:'Test',colour:'erc',tooltip:{title:'Test',q:{ideology:'test_ideology'}}}] },
    };
    convertGameToJSON(game, 0, (serializeError, text) => serializeError ? reject(serializeError) : accept(text));
  }));
}
beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.clear();
  clearContentCatalogCacheForTest();
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Phase 6B real content and Desk integration', () => {
  it('extracts every literal glossary ideology without collecting quality references', async () => {
    const glossary = JSON.parse(source('data/glossary.json'));
    const result = await extractFiles([
      { name: join('source', 'data', 'glossary.json'), contents: JSON.stringify(glossary) },
    ], {
      declarations: { registryFields: declarations.registryFields.filter((entry: { registry: string }) => entry.registry === 'glossary') },
      catalogs: { en: {}, ca: {} },
    });
    expect(result.errors).toEqual([]);
    const keys = new Set(result.units.map((unit: { key: string }) => unit.key));
    expect(keys.has('Anti-austerity Left, Plurinationalism')).toBe(true);
    for (const term of glossary.terms) {
      if (term.tooltip?.ideology) expect(keys.has(term.tooltip.ideology), term.id).toBe(true);
      if (term.tooltip?.q?.ideology) {
        expect(term.tooltip.q.ideology, term.id).toMatch(/^[a-z][a-z0-9_]*_ideology$/);
        expect(keys.has(term.tooltip.q.ideology), term.id).toBe(false);
      }
    }
  });

  it('renders fixed and changing glossary ideologies across locale switches without changing state', async () => {
    const compiled = JSON.parse(await compile());
    compiled.data.glossary = JSON.parse(source('data/glossary.json'));
    const game = useGameStore();
    game.initFromText(JSON.stringify(compiled));
    game.newGame();
    game.adapter!.qualities.cup_ideology = 'Anticapitalism, Independence';
    const translations = {
      'Anti-austerity Left, Plurinationalism': 'Esquerra antiausteritat, Plurinacionalisme',
      'Anticapitalism, Independence': 'Anticapitalisme, Independència',
      'Social Democracy, Independence': 'Socialdemocràcia, Independència',
    };
    vi.stubGlobal('fetch', vi.fn(async (url: string) => ({
      ok: true, json: async () => url.includes('/ca/') ? translations : {},
    })));
    const wrapper = mount(GlossaryTerm, {
      props: { termId: 'ada_colau', anchor: null }, global: { plugins: [i18n] },
    });
    const text = () => document.querySelector('.popover-ideology')?.textContent;
    try {
      expect(text()).toBe('Anti-austerity Left, Plurinationalism');
      const state = game.adapter!.exportStateJSON();
      const rng = game.adapter!.engine.random.getState();
      await game.setContentLanguage('ca');
      expect(text()).toBe(translations['Anti-austerity Left, Plurinationalism']);
      expect(game.adapter!.exportStateJSON()).toBe(state);
      expect(game.adapter!.engine.random.getState()).toEqual(rng);
      await wrapper.setProps({ termId: 'cup' });
      expect(text()).toBe(translations['Anticapitalism, Independence']);
      game.adapter!.qualities.cup_ideology = 'Social Democracy, Independence';
      await game.setContentLanguage('ca');
      expect(text()).toBe(translations['Social Democracy, Independence']);
      await game.setContentLanguage('en');
      expect(text()).toBe('Social Democracy, Independence');
      delete game.adapter!.qualities.cup_ideology;
      await game.setContentLanguage('en');
      expect(text()).toBeUndefined();
      await wrapper.setProps({ termId: 'ada_colau' });
      expect(text()).toBe('Anti-austerity Left, Plurinationalism');
    } finally {
      wrapper.unmount();
    }
  });

  it('retains selector trees and safely ignores unsupported catalog entries', () => {
    expect(parseContentCatalog({
      sentence: {select:'gender',cases:{m:'He',f:{plural:'count',cases:{one:'One',other:'Many'}},other:'They'}},
      bad: {select:'gender',cases:{m:'He'}}, array: [], number: 4,
    })).toEqual({sentence:{select:'gender',cases:{m:'He',f:{plural:'count',cases:{one:'One',other:'Many'}},other:'They'}}});
  });

  it('translates the real announcement and original gender branches while keeping English state', async () => {
    const adapter = DendryAdapter.fromJSONText(await compile());
    adapter.beginGame([11,22,33,44,55]);
    Object.assign(adapter.qualities, {president:'Neus Munté',president_gender:'f'});
    const english = adapter.goToScene('snap_election');
    const q = JSON.stringify(adapter.qualities);
    const rng = adapter.engine.random.getState();
    const actions = vi.spyOn(adapter.engine, '_runActions');
    const translated = adapter.refreshLocale('ca', catalan)!;
    expect(translated.title).toBe('Es convoquen noves eleccions al Parlament!');
    expect(translated.html).toContain('La presidenta Neus Munté');
    expect(translated.html).toContain('El Parlament ha estat dissolt');
    expect(translated.html).toContain('febrer');
    expect(translated.html).toContain('2013');
    expect(translated.html).toContain('pèrdua de confiança');
    expect(translated.html).toContain('Govern majoritari - 71/68');
    expect(translated.html).toContain('data-parties="ciu erc"');
    expect(translated.html).toContain('govern en funcions');
    expect(translated.choices[0].title).toBe('Que aquesta sigui la nostra oportunitat!');
    expect(JSON.stringify(adapter.qualities)).toBe(q);
    expect(adapter.engine.random.getState()).toEqual(rng);
    expect(actions).not.toHaveBeenCalled();
    expect(JSON.stringify(adapter.engine.state.currentContent)).not.toContain('govern en funcions');
    expect(adapter.refreshLocale('en', {})!.html).toBe(english.html);
  });

  it('captures names, genders, predicates and face-image boundaries through Library and JSON saves', async () => {
    const adapter = DendryAdapter.fromJSONText(await compile());
    adapter.beginGame();
    const english = adapter.goToScene('snap_election');
    Object.assign(adapter.qualities, {president:'Neus Munté',president_gender:'f',snap_budget_distrust:false,budget_failed_times:0});
    const translated = adapter.refreshLocale('ca', catalan)!;
    expect(translated.html).toContain('El president Artur Mas');
    expect(translated.html).toContain('pèrdua de confiança');
    adapter.goToScene('library');
    const returned = adapter.choose(0);
    expect(returned.html).toBe(translated.html);
    expect(returned.faceImageContent).toEqual(translated.faceImageContent);
    const restored = DendryAdapter.fromJSONText(await compile());
    restored.setLocale('ca', catalan);
    const loaded = restored.importStateJSON(adapter.exportStateJSON());
    expect(loaded.html).toBe(translated.html);
    expect(loaded.faceImageContent).toEqual(translated.faceImageContent);
    expect(restored.refreshLocale('en', {})!.html).toBe(english.html);
  });

  it('relocalizes the same chosen cards and deck availability without advancing the LIVE RNG', async () => {
    const handFiles = [
      {name:join('scenes','root.scene.dry'),contents:'title: Hand\nrole: desk\nis-hand: true\n\nYour hand.\n\n- @deck\n'},
      {name:join('scenes','deck.scene.dry'),contents:'title: Policies\nis-deck: true\n\n- @card\n'},
      {name:join('scenes','card.scene.dry'),contents:'title: A policy\nis-card: true\n\nPolicy content.\n'},
    ];
    const adapter = DendryAdapter.fromJSONText(await compile(handFiles));
    adapter.beginGame([11,22,33,44,55]);
    adapter.drawCard('deck');
    const state = adapter.exportStateJSON();
    const rng = adapter.engine.random.getState();
    const frame = adapter.refreshLocale('ca', {'Hand':'Mà','Your hand.':'La teva mà.','Policies':'Polítiques','A policy':'Una política'})!;
    expect(frame.hand.map(card=>card.id)).toEqual(['card']);
    expect(frame.hand[0].title).toBe('Una política');
    expect(frame.decks[0].title).toBe('Polítiques');
    expect(adapter.engine.random.getState()).toEqual(rng);
    expect(adapter.exportStateJSON()).toBe(state);
    adapter.refreshLocale('en', {});
    expect(adapter.engine.random.getState()).toEqual(rng);
  });

  it('keeps declared registry translations reactive without changing lookup identifiers', async () => {
    const game = useGameStore();
    game.initFromText(await compile());
    game.newGame();
    expect(game.achievements[0].description).toBe('Finish the game.');
    vi.stubGlobal('fetch', vi.fn(async()=>({ok:true,json:async()=>catalan})));
    await game.setContentLanguage('ca');
    expect(game.achievements[0].description).toBe('Acaba la partida.');
    expect(game.glossary[0].id).toBe('test');
    expect(game.glossary[0].match).toEqual(['Test']);
    expect(game.glossary[0].tooltip?.q?.ideology).toBe('test_ideology');
    expect(game.achievements[0].image).toBe('img/flags/estelada.png');
  });

  it('lets the latest requested language win when catalog responses arrive out of order', async () => {
    const game = useGameStore();
    game.initFromText(await compile()); game.newGame();
    const responses = new Map<string, (value: unknown)=>void>();
    vi.stubGlobal('fetch', vi.fn((url:string)=>new Promise(accept=>responses.set(url,accept))));
    const earlier = game.setContentLanguage('ca');
    const later = game.setContentLanguage('en');
    responses.get('/locales/en/content.json')!({ok:true,json:async()=>({'Test start':'English current'})});
    await later;
    responses.get('/locales/ca/content.json')!({ok:true,json:async()=>({'Test start':'Catalan stale'})});
    await earlier;
    expect(game.frame?.title).toBe('English current');
  });

  it('renders translated widget labels and plural seat tooltips without translating party keys', async () => {
    const game = useGameStore(); game.initFromText(await compile()); game.newGame();
    game.adapter!.refreshLocale('ca', catalan);
    const wrapper = mount(Coalitions, {props:{totalSeats:135,majoritySeats:68,
      q:{first:1,second:70,erc_s:1,ciu_s:69},
      coalitions:[{label:'Minority government - 1/68',seatsVar:'first',members:[{party:'erc',type:'support'}]},
        {label:'Second',seatsVar:'second',members:[{party:'ciu'}]}],
    }});
    expect(wrapper.text()).toContain('(suport)');
    expect(wrapper.find('.coalition-segment').attributes('title')).toBe('erc: 1 escó');
    expect(wrapper.findAll('.coalition-segment')[1].attributes('title')).toBe('ciu: 69 escons');
    expect(wrapper.find('.coalition-track').attributes('aria-label')).toBe('1 de 68 escons');
    expect(wrapper.find('.coalition-segment').attributes('style')).toContain('var(--erc)');
    expect(game.adapter!.translate('{seats} seats',{seats:0})).toBe('Cap escó (0)');
    wrapper.unmount();
  });
});
