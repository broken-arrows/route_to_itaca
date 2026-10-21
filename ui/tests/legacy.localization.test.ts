import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const w = window as any;
const catalog = (language: string, kind: string) => JSON.parse(read(`source/locales/${language}/${kind}.json`));

async function boot() {
  document.body.innerHTML = read('out/html/index.html').split('<body>')[1]!.split('</body>')[0]!;
  new Function(read('out/html/jquery-1.11.1.min.js'))();
  (globalThis as any).jQuery = w.jQuery;
  (globalThis as any).$ = w.jQuery;
  const BrowserUI = require('../../vendor/dendrynexus-ten/lib/ui/browser.js');
  const { convertJSONToGame } = require('../../vendor/dendrynexus-ten/lib/engine.js');
  const game = await new Promise<any>((accept, reject) => convertJSONToGame(read('out/game.json'), (error: any, value: any) => error ? reject(error) : accept(value)));
  w.mountWidgets = vi.fn();
  w.RTI_GAME_LIB = require('../../source/lib/index.js');
  w.dendryUI = new BrowserUI(game, w.$('#content'));
  new Function(read('out/html/localization.js'))();
  new Function(read('out/html/game.js'))();
  w.onload = null;
  expect(w.dendryModifyUI(w.dendryUI)).toBe(true);
  await w.classicLocaleReady;
  w.dendryUI.animate = false;
  return w.dendryUI;
}

describe('Classic localization', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const [, language, kind] = /locales\/(\w+)\/(\w+)\.json/.exec(url)!;
      const data = catalog(language!, kind!);
      if (language === 'ca' && kind === 'content') Object.assign(data, { 'New Game': 'Nova partida', Status: 'Estat' });
      return { ok: true, json: async () => data };
    }));
  });
  afterEach(() => { w.onload = null; vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it('shows the language switch only on the title menu and follows the selected language', async () => {
    const ui = await boot();
    const footer = document.getElementById('menu-footer') as HTMLElement;
    const switchButton = document.getElementById('menu-language-switch') as HTMLButtonElement;
    expect(ui.dendryEngine.state.sceneId).toBe('root.start_menu');
    expect(footer.hidden).toBe(false);
    expect(switchButton.textContent).toBe('També en Català!');
    expect(switchButton.value).toBe('ca');

    await w.ClassicI18n.setLocale('ca');
    expect(footer.hidden).toBe(false);
    expect(switchButton.textContent).toBe('Also in English');
    expect(switchButton.value).toBe('en');
    expect((document.getElementById('classic-language-ca') as HTMLInputElement).checked).toBe(true);

    ui.dendryEngine.goToScene('about');
    expect(footer.hidden).toBe(true);
    ui.dendryEngine.goToScene('root.start_menu');
    expect(footer.hidden).toBe(false);

    await w.ClassicI18n.setLocale('en');
    expect(switchButton.textContent).toBe('També en Català!');
    expect(switchButton.value).toBe('ca');
    expect((document.getElementById('classic-language-en') as HTMLInputElement).checked).toBe(true);
  });

  it('translates content and sidebar without actions, saves or RNG changes, including after save restoration', async () => {
    const ui = await boot(), engine = ui.dendryEngine;
    const state = JSON.stringify(engine.getExportableState());
    const rng = JSON.stringify(engine.random.getState());
    const arrivals = vi.spyOn(engine, '_runActions');
    const autosave = vi.spyOn(ui, 'autosave');
    await w.ClassicI18n.setLocale('ca');
    expect(document.documentElement.lang).toBe('ca');
    expect((document.getElementById('classic-language-ca') as HTMLInputElement).checked).toBe(true);
    expect(document.querySelector('#content')!.textContent).toContain('Nova partida');
    expect(document.querySelector('#qualities')!.textContent).toContain('Estat');
    expect(document.querySelector('#stats-link')!.textContent).toBe('Biblioteca');
    expect(JSON.stringify(engine.getExportableState())).toBe(state);
    expect(JSON.stringify(engine.random.getState())).toBe(rng);
    expect(arrivals).not.toHaveBeenCalled();
    expect(autosave).not.toHaveBeenCalled();
    ui.saveSlot(0);
    engine.setState(ui.saveStore.read('manual-1').record.state);
    expect(document.querySelector('#content')!.textContent).toContain('Nova partida');
    await w.ClassicI18n.setLocale('en');
    expect(document.querySelector('#content')!.textContent).toContain('New Game');
    expect(document.querySelector('#politics_tab')!.getAttribute('data-tooltip')).toBe('Politics');
    engine.state.gameOver = true;
    await w.ClassicI18n.setLocale('ca');
    expect(document.querySelectorAll('.game-over')).toHaveLength(1);
    await w.ClassicI18n.setLocale('en');
    expect(document.querySelectorAll('.game-over')).toHaveLength(1);
  });

  it('supports named parameters and fallback, and restores the saved language', async () => {
    const ui = await boot();
    await w.ClassicI18n.setLocale('ca');
    expect(w.ClassicI18n.text('Majority: {count} seats', { count: 68 })).toBe('Majority: 68 seats');
    expect(w.ClassicI18n.text('Not catalogued')).toBe('Not catalogued');
    expect(localStorage.getItem(ui.settings_key + ':language')).toBe('ca');
    new Function(read('out/html/localization.js'))();
    await w.ClassicI18n.init(ui);
    expect(w.ClassicI18n.locale).toBe('ca');
  });

  it('loads party forms and captures a real authored party insert', async () => {
    const ui = await boot(), engine = ui.dendryEngine;
    await w.ClassicI18n.setLocale('ca');
    engine.state.qualities.parlament_current_ciu = 'pdcat';
    const scene = engine.game.scenes['parlament_climate_change.climate_change_approach_convergence'];
    expect(scene).toBeDefined();
    const snapshot = engine._captureDisplayContent(scene.unavailableSubtitle, false);
    engine.state.qualities.parlament_current_ciu = 'junts';
    const html = require('dendrynexus-ten/lib/ui/content/html.js');
    expect(html.convertLine(engine._renderSnapshot(snapshot, false))).toBe('No tenim prou bones relacions amb el PDeCAT.');
    expect(html.convertLine(engine._renderSnapshot(snapshot, true))).toBe("We don't have good enough relations with pdcat");
  });

  it.each([false, true])('localizes captured cooldown captions without changing native Q (weekly=%s)', async (weekly) => {
    const ui = await boot(), engine = ui.dendryEngine;
    Object.assign(engine.state.qualities, { year: 2013, rubicon: weekly, advisor_action_timer: 3, parlament_timer: 3 });
    engine._runActions(engine.game.scenes.main.onArrival);
    const unit = weekly ? 'weeks' : 'months';
    expect(engine.state.qualities.pinnedCardsDescription).toBe(`Advisors - next action available in 3 ${unit}.`);
    engine.setLocale('ca', { [`Advisors - next action available in {count} ${unit}.`]: 'Assessors: {count}' });
    expect(ui._description('pinnedCardsDescription', '')).toBe('Assessors: 3');
    expect(engine.state.qualities.pinnedCardsDescription).toBe(`Advisors - next action available in 3 ${unit}.`);
  });

  it('keeps the latest language request and retains English on initial fetch failure', async () => {
    await boot();
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      await gate;
      const [, language, kind] = /locales\/(\w+)\/(\w+)\.json/.exec(url)!;
      return { ok: true, json: async () => catalog(language!, kind!) };
    }));
    const earlier = w.ClassicI18n.setLocale('ca');
    await w.ClassicI18n.setLocale('en');
    release();
    expect(await earlier).toBe(false);
    expect(w.ClassicI18n.locale).toBe('en');
    new Function(read('out/html/localization.js'))();
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    (document.getElementById('classic-language-ca') as HTMLInputElement).checked = true;
    expect(await w.ClassicI18n.setLocale('ca')).toBe(false);
    expect(w.ClassicI18n.locale).toBe('en');
    expect((document.getElementById('classic-language-en') as HTMLInputElement).checked).toBe(true);
    expect((document.getElementById('menu-language-switch') as HTMLButtonElement).textContent).toBe('També en Català!');
  });
});
