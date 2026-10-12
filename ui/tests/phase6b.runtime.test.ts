import { describe, expect, it } from 'vitest';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { DendryEngine, UserInterface, convertJSONToGame, makeFunctionFromSource } = require('dendrynexus-ten/lib/engine.js');
const { compileGame, convertGameToJSON } = require('dendrynexus-ten/lib/parsers/compiler.js');
const L = require('dendrynexus-ten/lib/localization.js');
const HTML = require('dendrynexus-ten/lib/ui/content/html.js');

class Output extends UserInterface {
  paragraphs: unknown[] = [];
  choices: any[] = [];
  decks: any[] = [];
  hand: any[] = [];
  pinned: any[] = [];
  signals: unknown[] = [];
  image: string | undefined;
  offset: number | undefined;
  newPage() { this.paragraphs = []; this.image = undefined; this.offset = undefined; }
  displayContent(paragraphs: unknown[], image?: string, offset?: number) {
    if (image) { this.image = image; this.offset = offset ?? this.paragraphs.length; }
    this.paragraphs.push(...paragraphs);
  }
  removeChoices() { this.choices = []; this.decks = []; this.pinned = []; }
  displayChoices(choices: any[]) { this.choices = choices; }
  displayDecks(decks: any[]) { this.decks = decks; }
  displayHand(hand: any[]) { this.hand = hand; }
  displayPinnedCards(pinned: any[]) { this.pinned = pinned; }
  signal(data: unknown) { this.signals.push(data); }
  get html() { return HTML.convert(this.paragraphs); }
}

async function compiled(files: Record<string, string>, data: object = {}) {
  const sources = Object.entries(files).map(([name, contents]) => ({ name: join('scenes', name + '.scene.dry'), contents }));
  const game = await new Promise<any>((resolve, reject) => compileGame(sources, (error: Error, result: any) => error ? reject(error) : resolve(result)));
  game.data = data;
  const json = await new Promise<string>((resolve, reject) => convertGameToJSON(game, 0, (error: Error, result: string) => error ? reject(error) : resolve(result)));
  return new Promise<any>((resolve, reject) => convertJSONToGame(json, (error: Error, result: any) => error ? reject(error) : resolve(result)));
}
function start(game: any, catalog?: object) {
  const ui = new Output();
  const engine = new DendryEngine(ui, game);
  engine.setGameLib({ stub: true });
  if (catalog) engine.setLocale('ca', catalog);
  engine.beginGame(['6b-runtime']);
  return { engine, ui };
}
function state(engine: any) { return JSON.parse(JSON.stringify(engine.getExportableState())); }

describe('6B shared English-string recipe lifecycle', () => {
  it('materializes captured English once for native lists regardless of the active locale', async () => {
    const { engine } = start(await compiled({ root: 'title: Root\n\nStart.\n' }), {
      'government': 'govern', '{name} leads the {label}.': '{name} dirigeix el {label}.',
    });
    const G = engine.gameLib;
    let evaluations = 0;
    const recipe = G.text('{name} leads the {label}.', {
      name: (() => { evaluations++; return 'Neus'; })(), label: G.text('government'),
    });
    const before = state(engine), rng = engine.random.getState(), captured = JSON.stringify(recipe);
    const reasons = [G.englishText(recipe)];
    expect(reasons.indexOf(G.englishText(recipe))).toBe(0);
    expect(reasons).toEqual(['Neus leads the government.']);
    expect(evaluations).toBe(1);
    expect(L.render(recipe, 'ca', engine.catalog)).toBe('Neus dirigeix el govern.');
    expect(JSON.stringify(recipe)).toBe(captured);
    expect(state(engine)).toEqual(before);
    expect(engine.random.getState()).toEqual(rng);
  });

  it('captures nested text and copies without replacing string qualities or sharing mutable parameters', async () => {
    const game = await compiled({ root: 'title: Root\n\nStart.\n' });
    const { engine } = start(game);
    const Q = engine.state.qualities, G = engine.gameLib;
    const nested = G.text('government');
    const parameters = { name: 'Neus', gender: 'f', label: nested };
    G.setText(Q, 'description', '{name} leads the {label}.', parameters);
    G.copyText(Q, 'previous', 'description');
    parameters.name = 'Artur'; nested.key = 'different';
    G.updateText(Q, 'description', { name: 'Artur', gender: 'm' });
    expect(Q.description).toBe('Artur leads the government.');
    expect(Q.previous).toBe('Neus leads the government.');
    expect(typeof Q.description).toBe('string');
    expect(engine.state.presentation.qualities.previous.recipe.params.gender).toBe('f');
    const oldRecipe = G.getText(Q, 'previous');
    oldRecipe.params.name = 'Mutated outside';
    expect(G.getText(Q, 'previous').params.name).toBe('Neus');
    expect(G.joinText([], ', ')).toEqual({ kind: 'join', parts: [], separator: ', ' });
    G.setText(Q, 'empty', G.joinText([], ', '));
    expect(Q.empty).toBe('');
  });

  it('invalidates same-value assignments, Object.assign, deletion/recreation, redefinition and ordinary replacement', async () => {
    const { engine } = start(await compiled({ root: 'title: Root\n\nStart.\n' }));
    const Q = engine.state.qualities, G = engine.gameLib;
    const marked = () => G.setText(Q, 'label', 'The president', { gender: 'f' });
    marked(); Q.label = Q.label;
    expect(engine.state.presentation.qualities.label).toBeUndefined();
    marked(); Object.assign(Q, { label: Q.label });
    expect(engine.state.presentation.qualities.label).toBeUndefined();
    marked(); delete Q.label; Q.label = 'The president';
    expect(engine.state.presentation.qualities.label).toBeUndefined();
    marked(); Object.defineProperty(Q, 'label', { value: 'The president', writable: true, enumerable: true, configurable: true });
    expect(engine.state.presentation.qualities.label).toBeUndefined();
    marked(); Q.label = Q.label.replaceAll('president', 'speaker');
    expect(G.getText(Q, 'label')).toEqual({ kind: 'literal', value: 'The speaker' });
    G.copyText(Q, 'unsetCopy', 'notSet');
    expect(Q.unsetCopy).toBeUndefined();
    Q.history = 0; G.copyText(Q, 'lastHistory', 'history');
    expect(Q.lastHistory).toBe(0);
  });

  it('preserves bounds, validity and quality-change signals and does not attach rejected recipes', async () => {
    const game = await compiled({ root: 'title: Root\n\nStart.\n' });
    game.qualities = {
      number: { initial: 5, min: 0, max: 10, signal: 'number-signal' },
      label: { initial: 'Original', isValid: makeFunctionFromSource('return Q.label !== "Rejected";'), signal: 'label-signal' },
    };
    const { engine, ui } = start(game);
    const Q = engine.state.qualities, G = engine.gameLib;
    Q.number = 30; expect(Q.number).toBe(10);
    Q.number = -10; expect(Q.number).toBe(0);
    G.setText(Q, 'label', 'Accepted');
    expect(Q.label).toBe('Accepted');
    G.setText(Q, 'label', 'Rejected');
    expect(Q.label).toBe('Accepted');
    expect(engine.state.presentation.qualities.label).toBeUndefined();
    const changes: any[] = ui.signals.filter((signal: any) => signal.event === 'quality-change');
    expect(changes.map(signal => signal.now)).toEqual([5, 'Original', 10, 0, 'Accepted']);
    expect(state(engine).qualities).toEqual({ number: 0, label: 'Accepted' });
  });

  it('rehydrates recipes/accessors and rejects helpers used with another engine or an old run', async () => {
    const game = await compiled({ root: 'title: Root\n\nStart.\n' });
    const a = start(game), b = start(game);
    const oldQ = a.engine.state.qualities;
    a.engine.gameLib.setText(oldQ, 'label', 'Ready {name}', { name: 'Neus' });
    const checkpoint = state(a.engine);
    b.engine.setState(checkpoint);
    expect(b.engine.gameLib.getText(b.engine.state.qualities, 'label').params.name).toBe('Neus');
    expect(() => a.engine.gameLib.setText(b.engine.state.qualities, 'label', 'Wrong')).toThrow(/another engine/);
    b.engine.state.qualities.label = b.engine.state.qualities.label;
    expect(b.engine.state.presentation.qualities.label).toBeUndefined();
    a.engine.beginGame(['new-run']);
    expect(() => a.engine.gameLib.getText(oldQ, 'label')).toThrow(/another engine/);
    expect(a.engine.state.presentation.qualities).toEqual({});
  });
});

describe('6B captured rendering and grammatical catalog entries', () => {
  it('preserves ordinary array/object insert coercion and captures expressions once', async () => {
    const game = await compiled({
      root: 'title: Root\non-arrival: {! Q.items=["a","b"]; Q.object={}; Q.calls=0; !}\n\n[+ items +] / [+ {! Q.calls++; return Q.object; !} +]\n',
    });
    const { engine, ui } = start(game);
    expect(ui.html).toContain('a,b / [object Object]');
    expect(engine.state.qualities.calls).toBe(1);
    const before = state(engine);
    engine.setLocale('ca', {}); engine.refreshPresentation();
    expect(state(engine)).toEqual(before);
    expect(engine.state.qualities.calls).toBe(1);
  });

  it('repaints a limited random choice selection without reevaluating candidates', async () => {
    const game = await compiled({
      root: 'title: Root\nmax-choices: 1\n\nPick.\n\n- @a\n- @b\n- @c\n',
      a: 'title: Alpha\n\nAlpha.\n', b: 'title: Beta\n\nBeta.\n', c: 'title: Gamma\n\nGamma.\n',
    });
    const { engine, ui } = start(game);
    const selected = engine.choiceCache.map((choice: any) => choice.id);
    const rng = engine.random.getState(), before = state(engine);
    expect(selected).toHaveLength(1);
    engine.setLocale('ca', { Alpha: 'Alfa', Beta: 'Beta', Gamma: 'Gamma', 'Pick.': 'Tria.' });
    engine.refreshPresentation();
    expect(ui.choices.map(choice => choice.id)).toEqual(selected);
    expect(engine.random.getState()).toEqual(rng);
    expect(state(engine)).toEqual(before);
  });
  it('protects literal widget JSON and permits a count consumed by a plural selector', () => {
    const widget = '<span data-props=\'{"parties":["erc"],"seats":21}\'>Ready</span>';
    expect(L.validateTemplate(widget, [])).toEqual([]);
    expect(L.render(L.text(widget), 'en', null)).toBe(widget);
    const entry = { plural: 'seats', cases: { '=0': 'Cap escó', one: '{seats} escó', other: '{seats} escons' } };
    expect(L.validateEntry('{seats} seats', entry, ['seats'])).toEqual([]);
    expect(L.render(L.text('{seats} seats', { seats: 0 }), 'ca', { '{seats} seats': entry })).toBe('Cap escó');
    expect(L.validateEntry('{name} has {seats} seats', entry, ['name', 'seats'])).toContain('Missing placeholder {name}.');
  });
  const files = {
    root: 'title: Root\non-arrival: {! Q.president="Neus"; Q.president_gender="f"; Q.arrivals=0; Q.displays=0; G.setText(Q,"government","Minority government - {seats}/{majority}",{seats:21,majority:68}); !}\n\nStart.\n\n- @announcement\n',
    announcement: 'title: President [+ president +] is ready.\nnew-page: true\nface-image: img/president.png\non-arrival: {! Q.arrivals++; !}\non-display: {! Q.displays++; !}\n\nPresident [+ president +] is ready. [? if president_gender="f": She leads [+ government +]. ?]\n\n- @continuation: Continue\n- @library: Library\n',
    continuation: 'title: Continuation\n\nEarlier president: [+ president +].\n\n- @library: Library\n',
    library: 'title: Library\nis-special: true\nnew-page: true\n\nLibrary content.\n\n- @backSpecialScene: Return\n',
  };
  const catalog = {
    'President {president} is ready.': { select: 'president_gender', cases: { m: 'El president {president} està preparat.', f: 'La presidenta {president} està preparada.', other: '{president} està a punt.' } },
    'President {president} is ready. {clause1}': { select: 'president_gender', cases: { m: 'El president {president} està preparat. {clause1}', f: 'La presidenta {president} està preparada. {clause1}', other: '{president} està a punt. {clause1}' } },
    'She leads {government}. ': 'Dirigeix {government}. ',
    'Minority government - {seats}/{majority}': 'Govern en minoria - {seats}/{majority}',
    'Earlier president: {president}.': 'President anterior: {president}.',
    Continue: 'Continua', Library: 'Biblioteca', Return: 'Torna',
  };

  it('shares canonical named units with the parser/extractor, including contexts absent from English', async () => {
    const game = await compiled(files, { localization: { grammarBindings: { president: 'president_gender' } } });
    const deps = game.scenes.announcement.content.stateDependencies;
    expect(deps.find((def: any) => def.quality === 'president')).toMatchObject({ language: 'logic', source: 'president' });
    const units = L.collectUnits(game.scenes.announcement.content, { grammarBindings: { president: 'president_gender' } });
    expect(units.find((unit: any) => unit.key.startsWith('President'))).toMatchObject({ key: 'President {president} is ready. {clause1}', context: ['president_gender'] });
    expect(L.validateEntry('President {president} is ready.', catalog['President {president} is ready.'], ['president', 'president_gender'])).toEqual([]);
  });

  it('keeps Q/page/choice fallback English, freezes context and refreshes without any evaluation or RNG', async () => {
    const game = await compiled(files, { localization: { grammarBindings: { president: 'president_gender' } } });
    const { engine, ui } = start(game, catalog);
    engine.goToScene('announcement');
    expect(ui.html).toContain('La presidenta Neus està preparada.');
    expect(ui.html).toContain('Govern en minoria - 21/68');
    expect(engine.getCurrentTitle()).toBe('La presidenta Neus està preparada.');
    expect(ui.choices[0].title).toEqual(['Continua']);
    expect(JSON.stringify(state(engine).currentContent)).toContain('President Neus is ready.');
    expect(JSON.stringify(state(engine).currentContent)).not.toContain('presidenta');
    const Q = engine.state.qualities;
    Q.president = 'Artur'; Q.president_gender = 'm';
    engine.goToScene('continuation');
    const before = state(engine), rng = engine.random.getState();
    const signals = ui.signals.length;
    engine.setLocale(null, null); engine.refreshPresentation();
    expect(ui.html).toContain('President Neus is ready.');
    expect(ui.html).toContain('Earlier president: Artur.');
    engine.setLocale('ca', catalog); engine.refreshPresentation();
    expect(ui.html).toContain('La presidenta Neus està preparada.');
    expect(ui.html).toContain('President anterior: Artur.');
    expect(state(engine)).toEqual(before);
    expect(engine.random.getState()).toEqual(rng);
    expect(ui.signals.length).toBe(signals);
    expect(ui.image).toBe('img/president.png'); expect(ui.offset).toBe(0);
  });

  it('restores accumulated/special pages across locale and saves without arrival replay, retaining load onDisplay behavior', async () => {
    const game = await compiled(files, { localization: { grammarBindings: { president: 'president_gender' } } });
    const { engine, ui } = start(game, catalog);
    engine.goToScene('announcement'); engine.goToScene('continuation');
    const expected = ui.html;
    engine.goToScene('library');
    const saved = state(engine);
    const classic = start(game);
    classic.engine.setState(saved);
    classic.engine.goToScene('backSpecialScene');
    expect(classic.ui.html).toContain('President Neus is ready.');
    const restored = state(classic.engine);
    engine.setState(restored);
    expect(ui.html).toBe(expected);
    expect(engine.state.qualities.arrivals).toBe(1);
    expect(engine.getCurrentTitle()).toBe('Continuation');
    expect(ui.image).toBe('img/president.png');
  });

  it('evaluates insert and conditional dependencies once and preserves captured results through locale refresh', async () => {
    const game = await compiled({
      root: 'title: Root\non-arrival: {! Q.reads=0; Q.predicates=0; !}\n\nStart.\n\n- @sample\n',
      sample: 'title: Sample\nnew-page: true\n\nValue [+ {! Q.reads++; return this.random.uint32(); !} +]. [? if {! Q.predicates++; return true; !}: Active. ?]\n',
    });
    const { engine, ui } = start(game);
    engine.goToScene('sample');
    expect(engine.state.qualities.reads).toBe(1);
    expect(engine.state.qualities.predicates).toBe(1);
    const rng = engine.random.getState(), before = state(engine);
    engine.setLocale('ca', { 'Value {value1}. {clause1}': 'Valor {value1}. {clause1}', 'Active. ': 'Actiu. ' });
    engine.refreshPresentation();
    expect(ui.html).toContain('Valor');
    expect(engine.state.qualities.reads).toBe(1);
    expect(state(engine)).toEqual(before);
    expect(engine.random.getState()).toEqual(rng);
  });

  it('uses nested select/plural entries with English fallback and protects markup and technical attributes', async () => {
    const { engine } = start(await compiled({ root: 'title: Root\n\nStart.\n' }));
    const G = engine.gameLib, Q = engine.state.qualities;
    const ca = {
      '{count} votes': { plural: 'count', cases: { '=0': 'Cap vot ({count})', one: '{count} vot', other: '{count} vots' } },
      Ready: { select: 'gender', cases: { f: 'Preparada', m: 'Preparat', other: 'A punt' } },
    };
    engine.setLocale('ca', ca);
    expect(engine.translate('{count} votes', { count: 0 })).toBe('Cap vot (0)');
    expect(engine.translate('{count} votes', { count: 1 })).toBe('1 vot');
    expect(engine.translate('{count} votes', { count: 2 })).toBe('2 vots');
    expect(engine.translate('Ready', { gender: 'f' })).toBe('Preparada');
    const markup = G.markupText('span', { class: 'government', 'data-parties': 'cdc-erc', 'data-summary': G.text('{count} votes', { count: 2 }) }, G.text('Ready', { gender: 'f' }));
    G.setText(Q, 'label', markup);
    expect(Q.label).toBe('<span class="government" data-parties="cdc-erc" data-summary="2 votes">Ready</span>');
    expect(L.render(G.getText(Q, 'label'), 'ca', ca)).toBe('<span class="government" data-parties="cdc-erc" data-summary="2 vots">Preparada</span>');
    expect(L.validateEntry('Strong <em>{word}</em>', 'Fort <a>{word}</a>', ['word'])).toContain('Protected HTML markup changed.');
    expect(L.render(G.text('Ready', { gender: 'f' }), 'ca', { Ready: { select: 'gender', cases: { f: 'Preparada' } } })).toBe('Ready');
  });

  it('loads an old recipe-less page safely and retains it as English when a continuation is appended', async () => {
    const game = await compiled(files);
    const { engine, ui } = start(game);
    engine.goToScene('announcement');
    const saved = state(engine); delete saved.presentation;
    engine.setLocale('ca', catalog); engine.setState(saved);
    expect(ui.html).toContain('President Neus is ready.');
    engine.goToScene('continuation'); engine.refreshPresentation();
    expect(ui.html).toContain('President Neus is ready.');
    expect(ui.html).toContain('President anterior: Neus.');
  });
});
