import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import OptionsPane from '../src/components/menu/OptionsPane.vue';
import { clearContentCatalogCacheForTest } from '../src/locales/content';
import { useGameStore } from '../src/stores/game';
import { useSettingsStore } from '../src/stores/settings';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { join } from 'node:path';

let pinia: ReturnType<typeof createPinia>;

beforeEach(() => {
  localStorage.clear();
  i18n.global.locale.value = 'en';
  clearContentCatalogCacheForTest();
  pinia = createPinia();
  setActivePinia(pinia);
  vi.restoreAllMocks();
});

function mountPane() {
  return mount(OptionsPane, { global: { plugins: [pinia, i18n] } });
}

function compileText(files: { name: string; contents: string }[]): Promise<string> {
  return new Promise((resolve, reject) => {
    compileGame(files, (error, game) => {
      if (error) return reject(error);
      convertGameToJSON(game, 0, (jsonError, json) => jsonError ? reject(jsonError) : resolve(json));
    });
  });
}

describe('OptionsPane', () => {
  it('installs a persisted content locale before beginGame without inventing engine state', async () => {
    const text = await compileText([
      { name: 'info.dry', contents: 'title: Test\nauthor: Test\nstorage-id: rti\nlanguages: en ca\n' },
      { name: join('scenes', 'root.scene.dry'), contents: 'title: Root\n\nHello\n' },
    ]);
    localStorage.setItem('rti:settings', JSON.stringify({ language: 'ca', animations: true, eventImages: true }));
    vi.stubGlobal('fetch', vi.fn(async (url: string) => url.endsWith('game.en.json')
      ? { ok: true, text: async () => text }
      : { ok: true, json: async () => ({ Hello: 'Hola' }) }));
    const game = useGameStore();

    await game.initFromUrl('/game.en.json');

    expect(game.ready).toBe(true);
    expect(game.frame).toBeNull();
    expect(game.q).toEqual({});
    expect(game.loadError).toBe(false);
    expect(game.newGame()).toBe(true);
    expect(game.frame!.html).toContain('Hola');
  });

  it('offers only EN and CA, persists the working toggles, and leaves Music disabled as WIP', async () => {
    const settings = useSettingsStore();
    settings.configure('rti');
    const wrapper = mountPane();

    expect(wrapper.findAll('input[name="language"]')).toHaveLength(2);
    expect(wrapper.find('[value="es"]').exists()).toBe(false);
    expect(wrapper.get('[data-test="setting-music"]').attributes('disabled')).toBeDefined();
    expect(wrapper.text()).toContain(i18n.global.t('shell.options.wip'));

    await wrapper.get('[data-test="setting-animations"]').setValue(false);
    await wrapper.get('[data-test="setting-event-images"]').setValue(false);
    expect(settings.animations).toBe(false);
    expect(settings.eventImages).toBe(false);
    expect(JSON.parse(localStorage.getItem('rti:settings')!)).toMatchObject({
      animations: false,
      eventImages: false,
    });
  });

  it('switches Vue and current engine content live without navigating or replaying arrival', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ Hello: 'Hola', Continue: 'Continua ara' }),
    }));
    const game = useGameStore();
    const text = await compileText([
      { name: 'info.dry', contents: 'title: Test\nauthor: Test\nstorage-id: rti\nlanguages: en ca\n' },
      { name: join('scenes', 'root.scene.dry'), contents: 'title: Root\non-arrival: {!\n  Q.arrivals = (Q.arrivals || 0) + 1;\n!}\n\nHello\n\n- @next: Continue\n' },
      { name: join('scenes', 'next.scene.dry'), contents: 'title: Continue\n\nDone.\n' },
    ]);
    game.initFromText(text);
    game.newGame();
    const beforeScene = game.frame!.sceneId;
    const wrapper = mountPane();

    await wrapper.get('[data-test="setting-language-ca"]').setValue(true);
    await flushPromises();

    expect(i18n.global.locale.value).toBe('ca');
    expect(game.frame!.sceneId).toBe(beforeScene);
    expect(game.q.arrivals).toBe(1);
    expect(game.frame!.html).toContain('Hola');
    expect(game.frame!.choices[0].title).toBe('Continua ara');
    expect(JSON.parse(localStorage.getItem('rti:settings')!).language).toBe('ca');
  });

  it('preserves an accumulated page while translating it without lifecycle replay', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ Hello: 'Hola', Second: 'Segon', Continue: 'Continua' }),
    }));
    const game = useGameStore();
    const text = await compileText([
      { name: 'info.dry', contents: 'title: Test\nauthor: Test\nstorage-id: rti\nlanguages: en ca\n' },
      { name: join('scenes', 'root.scene.dry'), contents: 'title: Root\non-arrival: {! Q.rootArrivals = (Q.rootArrivals || 0) + 1; !}\n\nHello\n\n- @next: Continue\n' },
      { name: join('scenes', 'next.scene.dry'), contents: 'title: Next\non-arrival: {! Q.nextArrivals = (Q.nextArrivals || 0) + 1; !}\n\nSecond\n' },
    ]);
    game.initFromText(text);
    game.newGame();
    game.choose(0);
    expect(game.frame!.html).toContain('Hello');
    expect(game.frame!.html).toContain('Second');
    const wrapper = mountPane();

    await wrapper.get('[data-test="setting-language-ca"]').setValue(true);
    await flushPromises();

    expect(game.frame!.html).toContain('Hola');
    expect(game.frame!.html).toContain('Segon');
    expect(game.q.rootArrivals).toBe(1);
    expect(game.q.nextArrivals).toBe(1);
  });
});
