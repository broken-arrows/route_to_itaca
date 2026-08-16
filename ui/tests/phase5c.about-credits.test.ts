import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { i18n, setLocale } from '../src/i18n';
import GameView from '../src/views/GameView.vue';
import { useGameStore } from '../src/stores/game';
import { setAnimationsForTest } from '../src/stores/desk';

const gameData = {
  info: { title: 'Route to Itaca', storageId: 'phase5c-about', version: '1.0.0' },
  scenes: {
    root: { id: 'root', type: 'scene', newPage: true, goTo: [{ id: 'menu' }] },
    menu: {
      id: 'menu', type: 'scene', title: 'Route', newPage: true, role: 'title-hub',
      options: [{ id: '@about' }],
    },
    about: {
      id: 'about', type: 'scene', title: 'About', role: 'main-menu-item',
      content: [{ type: 'heading', content: ['About Route to Itaca'] }],
      options: [{ id: '@credits' }, { id: '@menu', tags: ['shell-return'] }],
    },
    credits: {
      id: 'credits', type: 'scene', title: 'Credits',
      content: [{ type: 'heading', content: ['Credits'] }, { type: 'paragraph', content: ['Source links remain live.'] }],
      options: [{ id: '@about' }],
    },
  },
  qualities: {}, qdisplays: {}, tagLookup: {},
};

let wrapper: VueWrapper | null = null;
let pinia: ReturnType<typeof createPinia>;

beforeEach(() => {
  localStorage.clear();
  setLocale('en');
  pinia = createPinia();
  setActivePinia(pinia);
  setAnimationsForTest(false);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

function button(text: string) {
  const found = wrapper!.findAll('button').find((node) => node.text().includes(text));
  if (!found) throw new Error(`missing button ${text}`);
  return found;
}

describe('Phase 5C About and Credits', () => {
  it('keeps Credits as live internal About navigation while the title ribbon remains visible', async () => {
    const game = useGameStore();
    game.initFromText(JSON.stringify(gameData));
    wrapper = mount(GameView, { attachTo: document.body, global: { plugins: [pinia, i18n] } });
    await flushPromises();

    expect(wrapper.findAll('[data-test="ribbon-stack"] button').map((node) => node.text()))
      .toEqual(expect.arrayContaining([expect.stringContaining('About')]));
    expect(wrapper.text()).not.toContain('Source links remain live.');

    await button('About').trigger('click');
    expect(game.frame?.sceneId).toBe('about');
    expect(wrapper.get('[data-test="pane-heading"]').text()).toBe('About Route to Itaca');

    await button('Credits').trigger('click');
    expect(game.frame?.sceneId).toBe('credits');
    expect(wrapper.text()).toContain('Source links remain live.');
    expect(wrapper.find('[data-test="ribbon-stack"]').exists()).toBe(true);

    await button('About').trigger('click');
    expect(game.frame?.sceneId).toBe('about');
    expect(wrapper.get('[data-test="pane-heading"]').text()).toBe('About Route to Itaca');
  });
});
