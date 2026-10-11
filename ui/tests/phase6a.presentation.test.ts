import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { flushPromises, mount } from '@vue/test-utils';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { i18n } from '../src/i18n';
import GameView from '../src/views/GameView.vue';
import { useGameStore } from '../src/stores/game';
import { useSettingsStore } from '../src/stores/settings';
import { useShellStore } from '../src/stores/shell';
import { DendryAdapter } from '../src/engine/adapter';

const paragraph = (text: string) => [{ type: 'paragraph', content: [text] }];
const fixture = {
  info: { title: 'Presentation test', ifid: 'phase6a-test', version: '1' },
  scenes: {
    root: { id: 'root', type: 'scene', goTo: [{ id: 'menu' }], onArrival: [{ $code: 'Q.boots = (Q.boots || 0) + 1;' }] },
    menu: { id: 'menu', type: 'scene', title: 'Title', role: 'title-hub', newPage: true, options: [{ id: '@setup' }] },
    setup: { id: 'setup', type: 'scene', title: 'New Game', role: 'setup', tags: ['starts-game'], newPage: true,
      setBg: 'img/background.jpg', content: paragraph('Choose your party.'), options: [{ id: '@intro' }],
      onArrival: [{ $code: 'Q.setupEntries = (Q.setupEntries || 0) + 1;' }] },
    intro: { id: 'intro', type: 'scene', title: 'Introduction', role: 'intro', newPage: true,
      content: paragraph('Read the introduction.'), options: [{ id: '@desk' }],
      onArrival: [{ $code: 'Q.introEntries = (Q.introEntries || 0) + 1;' }] },
    desk: { id: 'desk', type: 'scene', title: 'Desk', role: 'desk', newPage: true, content: paragraph('At the desk.'),
      onArrival: [{ $code: 'Q.turns = (Q.turns || 0) + 1;' }] },
    ending: { id: 'ending', type: 'scene', title: 'Ending', role: 'ending', newPage: true,
      content: paragraph('Review the outcome.'), options: [{ id: '@terminal' }],
      onArrival: [{ $code: 'Q.endingEntries = (Q.endingEntries || 0) + 1;' }] },
    terminal: { id: 'terminal', type: 'scene', title: 'Finished', role: 'ending', gameOver: true },
  }, qualities: {}, qdisplays: {}, tagLookup: {},
};
let pinia: ReturnType<typeof createPinia>;
const wrappers: Array<{ unmount(): void }> = [];
beforeEach(() => {
  localStorage.clear();
  pinia = createPinia();
  setActivePinia(pinia);
  useSettingsStore().setAnimations(false);
});
afterEach(() => {
  wrappers.splice(0).forEach((wrapper) => wrapper.unmount());
  vi.useRealTimers();
});
async function boot() {
  const game = useGameStore();
  game.initFromText(JSON.stringify(fixture));
  const wrapper = mount(GameView, { attachTo: document.body, global: { plugins: [pinia, i18n] } });
  wrappers.push(wrapper);
  await flushPromises();
  return { game, wrapper, shell: useShellStore(), settings: useSettingsStore() };
}
async function start(wrapper: ReturnType<typeof mount>) {
  await wrapper.get('[data-test="ribbon-authored:setup"]').trigger('click');
}

describe('Phase 6A presentation navigation', () => {
  it('runs setup, intro and gameplay actions once with animations disabled', async () => {
    const { game, wrapper } = await boot();
    await start(wrapper);
    expect(game.q.setupEntries).toBe(1);
    expect(wrapper.get('[data-test="presentation-background"]').attributes('src')).toBe('/img/background.jpg');
    expect(wrapper.find('.clipboard-frame').exists()).toBe(false);
    await wrapper.get('[data-test="presentation-choice-0"]').trigger('click');
    expect(game.q.introEntries).toBe(1);
    expect(game.frame?.effectiveRole).toBe('intro');
    await wrapper.get('[data-test="presentation-choice-0"]').trigger('click');
    expect(game.frame?.effectiveRole).toBe('desk');
    expect(game.q.turns).toBe(1);
    expect(wrapper.find('[data-test="surface-fade"]').exists()).toBe(false);
  });

  it('blocks duplicate navigation and pause during a fade without replaying actions', async () => {
    const { game, wrapper, shell, settings } = await boot();
    vi.useFakeTimers();
    settings.setAnimations(true);
    await start(wrapper);
    expect(shell.transitioning).toBe(true);
    expect(game.q.setupEntries).toBe(1);
    game.choose(0);
    game.chooseFromShell(0);
    game.goToScene('intro');
    expect(game.frame?.sceneId).toBe('setup');
    expect(shell.openPause('page', 'setup', false)).toBe(false);
    await vi.advanceTimersByTimeAsync(400);
    expect(shell.transitioning).toBe(false);
    await wrapper.get('[data-test="presentation-choice-0"]').trigger('click');
    expect(game.q.introEntries).toBe(1);
    expect(shell.transitioning).toBe(true);
    await vi.advanceTimersByTimeAsync(400);
    expect(game.q.introEntries).toBe(1);
    expect(game.q.turns).toBeUndefined();
  });

  it('preserves stable-page pause and resumes the same setup', async () => {
    const { game, wrapper, shell } = await boot();
    await start(wrapper);
    const region = wrapper.get('[data-test="presentation-region"]').element;
    region.scrollTop = 160;
    await wrapper.get('[data-test="pause-button"]').trigger('click');
    await flushPromises();
    expect(wrapper.get('[data-test="paused-surface-snapshot"] [data-test="presentation-region"]').element.scrollTop).toBe(160);
    const before = game.captureState();
    expect(shell.paused).toBe(true);
    game.choose(0);
    expect(game.captureState()).toBe(before);
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await flushPromises();
    expect(shell.paused).toBe(false);
    expect(game.frame?.sceneId).toBe('setup');
    expect(region.scrollTop).toBe(160);
  });

  it('finishes a pending fade immediately when animations are disabled', async () => {
    const { game, wrapper, shell, settings } = await boot();
    vi.useFakeTimers();
    settings.setAnimations(true);
    await start(wrapper);
    expect(shell.transitioning).toBe(true);
    settings.setAnimations(false);
    await wrapper.vm.$nextTick();
    expect(shell.transitioning).toBe(false);
    await wrapper.get('[data-test="presentation-choice-0"]').trigger('click');
    await vi.advanceTimersByTimeAsync(500);
    expect(game.frame?.sceneId).toBe('intro');
    expect(game.q.introEntries).toBe(1);
  });

  it('keeps missing-art fallback usable and returns from a terminal ending to title', async () => {
    const { game, wrapper, shell } = await boot();
    await start(wrapper);
    await wrapper.get('[data-test="presentation-background"]').trigger('error');
    expect(wrapper.find('[data-test="presentation-background"]').exists()).toBe(false);
    expect(wrapper.get('[data-test="presentation-choice-0"]').attributes('disabled')).toBeUndefined();
    game.goToScene('ending');
    await flushPromises();
    expect(wrapper.find('[data-test="pause-button"]').exists()).toBe(true);
    await wrapper.get('[data-test="presentation-choice-0"]').trigger('click');
    expect(game.frame?.gameOver).toBe(true);
    expect(wrapper.find('[data-test="pause-button"]').exists()).toBe(false);
    await wrapper.get('[data-test="return-to-title"]').trigger('click');
    expect(shell.mode).toBe('title');
    expect(game.frame?.sceneId).toBe('menu');
    expect(wrapper.find('[data-test="title-shell"]').exists()).toBe(true);
  });
});

describe('Phase 6A real authored routes and restoration', () => {
  const compiled = readFileSync(join(__dirname, '../../out/game.json'), 'utf8');
  it('awards Lucky one only on the reserved wish scene, never on ordinary setup', () => {
    const adapter = DendryAdapter.fromJSONText(compiled);
    adapter.beginGame([1, 2, 3, 4]);
    adapter.goToScene('root.start');
    expect(adapter.qualities.achievement_made_a_wish).not.toBeTruthy();
    expect(adapter.qualities.game_achievement_made_a_wish).not.toBeTruthy();
    adapter.goToScene('root.esquerra_2');
    adapter.goToScene('post_event.make_a_wish');
    expect(adapter.qualities.achievement_made_a_wish).toBeTruthy();
    expect(adapter.qualities.game_achievement_made_a_wish).toBeTruthy();
  });

  it('places the party logo with the party text after the earlier setup prose', async () => {
    const game = useGameStore();
    game.initFromText(compiled);
    game.newGame();
    const shell = useShellStore();
    const hub = game.roleHub('title-hub');
    if (hub) shell.enterTitle(hub);
    const wrapper = mount(GameView, { attachTo: document.body, global: { plugins: [pinia, i18n] } });
    wrappers.push(wrapper);
    await flushPromises();
    await wrapper.get('[data-test="ribbon-authored:root.start"]').trigger('click');
    game.goToScene('root.normal_mode');
    game.goToScene('root.esquerra');
    await flushPromises();
    const nodes = Array.from(wrapper.get('[data-test="presentation-page"]').element.querySelectorAll('img, h1, p'));
    const difficulty = nodes.findIndex((node) => node.textContent?.includes('Normal difficulty'));
    const image = nodes.findIndex((node) => node.getAttribute('data-test') === 'presentation-face-image');
    const party = nodes.findIndex((node) => node.tagName === 'P' && node.textContent?.includes('Republican Left of Catalonia - ERC'));
    expect(image).toBeGreaterThan(difficulty);
    expect(image).toBeLessThan(party);
  });

  it('keeps the image beside its scene text through save/load, locale refresh and Library return', () => {
    const adapter = DendryAdapter.fromJSONText(compiled);
    adapter.beginGame([1, 2, 3, 4]);
    adapter.goToScene('root.start');
    adapter.goToScene('root.normal_mode');
    const detail = adapter.goToScene('root.esquerra');
    expect(detail.faceImageContent?.beforeHtml).toContain('difficulty');
    expect(detail.faceImageContent?.afterHtml).toContain('Republican Left of Catalonia');
    expect(detail.faceImageContent?.afterHtml).not.toContain('difficulty');
    const restored = DendryAdapter.fromJSONText(compiled);
    const loaded = restored.importStateJSON(adapter.exportStateJSON());
    expect(loaded.faceImageContent).toEqual(detail.faceImageContent);
    expect(restored.refreshLocale('ca', {})?.faceImageContent).toEqual(detail.faceImageContent);
    restored.goToScene('library');
    expect(restored.goToScene('backSpecialScene').faceImageContent).toEqual(detail.faceImageContent);
  });

  it.each([
    ['root.start', 'setup'], ['root.party_selection', 'setup'], ['root.esquerra', 'setup'],
    ['intros', 'intro'], ['game_over', 'ending'], ['game_over.end_game', 'ending'],
  ])('restores %s as %s without repeating arrival actions', (scene, role) => {
    const first = DendryAdapter.fromJSONText(compiled);
    first.beginGame([1, 2, 3, 4]);
    first.goToScene('root.start');
    first.goToScene(scene);
    const state = first.exportStateJSON();
    const restored = DendryAdapter.fromJSONText(compiled);
    expect(restored.importStateJSON(state).effectiveRole).toBe(role);
    expect(restored.qualities).toEqual(JSON.parse(JSON.stringify(first.qualities)));
  });

  it('reads each introduction branch once and retains the authored skip/continue path', () => {
    const adapter = DendryAdapter.fromJSONText(compiled);
    adapter.beginGame([1, 2, 3, 4]);
    adapter.goToScene('root.start');
    let frame = adapter.goToScene('root.esquerra_2');
    for (const scene of ['charter_of_autonomy', 'dire_economics', 'current_status']) {
      const choice = frame.choices.findIndex((item) => item.id === `intros.${scene}`);
      expect(choice).toBeGreaterThanOrEqual(0);
      frame = adapter.choose(choice);
      expect(frame.effectiveRole).toBe('intro');
    }
    expect(adapter.qualities.intros_seen).toBe(3);
    expect(frame.choices.map((item) => item.id)).toEqual(['intros.what_now']);
    expect(adapter.choose(0).effectiveRole).not.toBe('intro');
  });

  it.each([
    ['http://localhost:5173/', true], ['http://127.0.0.1:5173/', true],
    ['file:///game/index.html', true], ['https://broken-arrows.github.io/route_to_itaca/desk/', false],
    ['https://localhost.example.com/', false],
  ])('retains the New Game origin gate at %s', (url, allowed) => {
    const game = JSON.parse(compiled);
    const arrival = game.scenes['root.start_menu'].onArrival[0].$code as string;
    const origin = arrival.slice(arrival.indexOf('const valid_urls'), arrival.indexOf('// setup'));
    const qualities: Record<string, unknown> = {};
    const log = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      new Function('window', 'Q', origin)({ location: new URL(url) }, qualities);
      expect(qualities.url_valid).toBe(allowed);
      expect(log).toHaveBeenCalledTimes(allowed ? 0 : 1);
    } finally { log.mockRestore(); }
  });

  it('keeps Historical difficulty and its save restriction when returning from party details', () => {
    const adapter = DendryAdapter.fromJSONText(compiled);
    adapter.beginGame([1, 2, 3, 4]);
    adapter.goToScene('root.start');
    adapter.goToScene('root.historical_mode');
    const detail = adapter.goToScene('root.esquerra');
    const back = detail.choices.findIndex((choice) => choice.id === 'root.back_to_party_selection');
    expect(back).toBeGreaterThanOrEqual(0);
    const selection = adapter.choose(back);
    expect(selection.effectiveRole).toBe('setup');
    expect(selection.html).toContain('Historical');
    expect(selection.html).not.toContain('>Hard</span> difficulty');
    expect(adapter.qualities.historical_mode).toBe(1);
    expect(adapter.engine.state.disableSaves).toBe(true);
  });
});
