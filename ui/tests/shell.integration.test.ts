import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { nextTick } from 'vue';
import { i18n } from '../src/i18n';
import GameView from '../src/views/GameView.vue';
import { useGameStore } from '../src/stores/game';
import { useShellStore } from '../src/stores/shell';
import { setAnimationsForTest } from '../src/stores/desk';

const shellGame = {
  info: { title: 'Route to Itaca - An Alternate History', storageId: 'phase5a-test', version: '1.0.0' },
  scenes: {
    root: {
      id: 'root', type: 'scene', newPage: true,
      onArrival: [{ $code: 'Q.initialized = (Q.initialized || 0) + 1; Q.simulated = 0;' }],
      goTo: [{ id: 'menu' }],
    },
    menu: {
      id: 'menu', type: 'scene', title: 'Route', newPage: true, role: 'title-hub',
      options: [{ id: '@start' }, { id: '@achievements' }, { id: '@simulator' }, { id: '@about' }],
    },
    start: {
      id: 'start', type: 'scene', title: 'New Game', role: 'main-menu-item', tags: ['starts-game'],
      onArrival: [{ $code: 'Q.started = true;' }], content: [{ type: 'paragraph', content: ['Setup'] }],
      options: [],
    },
    achievements: {
      id: 'achievements', type: 'scene', title: 'Achievements', newPage: true, role: 'main-menu-item', tags: ['shell-achievements'],
      content: [
        { type: 'heading', content: ['Filed Achievements'] },
        { type: 'paragraph', content: ['Ledger'] },
      ], options: [],
    },
    simulator: {
      id: 'simulator', type: 'scene', title: 'Election Simulator', role: 'main-menu-item',
      onArrival: [{ $code: 'Q.simulated = 99;' }], content: [{ type: 'paragraph', content: ['Simulator'] }],
      options: [],
    },
    about: {
      id: 'about', type: 'scene', title: 'About', role: 'main-menu-item',
      content: [{ type: 'paragraph', content: ['About body'] }], options: [],
    },
  },
  qualities: {}, qdisplays: {}, tagLookup: {},
};

let pinia: ReturnType<typeof createPinia>;
const mounted: Array<{ unmount(): void }> = [];
beforeEach(() => {
  localStorage.clear();
  pinia = createPinia();
  setActivePinia(pinia);
  setAnimationsForTest(false);
});
afterEach(() => {
  for (const wrapper of mounted.splice(0)) wrapper.unmount();
});

async function mountTitle() {
  const game = useGameStore();
  game.initFromText(JSON.stringify(shellGame));
  const wrapper = mount(GameView, { attachTo: document.body, global: { plugins: [pinia, i18n] } });
  mounted.push(wrapper);
  await flushPromises();
  return { game, shell: useShellStore(), wrapper };
}

function ribbon(wrapper: ReturnType<typeof mount>, text: string) {
  const button = wrapper.findAll('[data-test="ribbon-stack"] button').find((node) => node.text().includes(text));
  if (!button) throw new Error(`missing ribbon ${text}`);
  return button;
}

describe('Phase 5A shell integration', () => {
  it('preserves authored order between injected Load Game and Options commands', async () => {
    const { shell, wrapper } = await mountTitle();
    expect(shell.mode).toBe('title');
    expect(wrapper.findAll('[data-test="ribbon-stack"] button').map((node) => node.text())).toEqual([
      expect.stringContaining('Load Game'),
      expect.stringContaining('New Game'),
      expect.stringContaining('Achievements'),
      expect.stringContaining('Election Simulator'),
      expect.stringContaining('About'),
      expect.stringContaining('Options'),
    ]);
    expect(wrapper.find('.title-identity').text()).toContain('Route to Itaca');
    expect(document.activeElement?.textContent).toContain('Load Game');
  });

  it('opens the title Save/Load manager without pause-only save commands', async () => {
    const { wrapper } = await mountTitle();
    await ribbon(wrapper, 'Load Game').trigger('click');
    expect(wrapper.find('[data-test="pane-heading"]').text()).toBe('Load Game');
    expect(wrapper.find('[data-test="save-load-pane"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="save-new"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="save-overwrite"]').exists()).toBe(false);
  });

  it('reuses the working Options pane from the title shell', async () => {
    const { wrapper } = await mountTitle();
    await ribbon(wrapper, 'Options').trigger('click');
    expect(wrapper.find('[data-test="options-pane"]').exists()).toBe(true);
    expect(wrapper.findAll('input[name="language"]')).toHaveLength(2);
    expect(wrapper.find('[data-test="setting-music"]').attributes('disabled')).toBeDefined();
  });

  it('moves focus into a pane and returns it to the invoking ribbon with animations off', async () => {
    const { wrapper } = await mountTitle();
    expect(wrapper.find('[data-test="ribbon-stack"]').classes()).not.toContain('is-animated');
    await ribbon(wrapper, 'About').trigger('click');
    await nextTick();
    expect(document.activeElement).toBe(wrapper.find('[data-test="pane-heading"]').element);
    expect(wrapper.find('[data-test="step-aside-pane"]').classes()).not.toContain('is-animated');
    await wrapper.find('[data-test="pane-close"]').trigger('click');
    await nextTick();
    expect(document.activeElement?.textContent).toContain('About');
  });

  it('switches title panes directly, then New Game fully resets simulator mutations', async () => {
    const { game, shell, wrapper } = await mountTitle();
    await ribbon(wrapper, 'Election Simulator').trigger('click');
    expect(game.q.simulated).toBe(99);
    expect(shell.mode).toBe('title');
    expect(wrapper.text()).toContain('Simulator');

    await ribbon(wrapper, 'About').trigger('click');
    expect(wrapper.text()).toContain('About body');
    expect(game.q.initialized).toBe(1);

    await ribbon(wrapper, 'New Game').trigger('click');
    expect(shell.mode).toBe('playing');
    expect(game.frame?.sceneId).toBe('start');
    expect(game.q.simulated).toBe(0);
    expect(game.q.initialized).toBe(1);
    expect(game.q.started).toBe(true);
  });

  it('pauses only a stable playing surface, keeps it mounted and blocks engine choices', async () => {
    const { game, shell, wrapper } = await mountTitle();
    await ribbon(wrapper, 'New Game').trigger('click');
    const scene = game.frame?.sceneId;
    expect(wrapper.find('[data-test="pause-button"]').exists()).toBe(true);
    await wrapper.find('[data-test="pause-button"]').trigger('click');
    expect(shell.paused).toBe(true);
    expect(wrapper.find('.pause-heading').text()).toContain('Paused');
    expect(wrapper.find('[data-test="playing-surface"]').attributes('inert')).toBeDefined();
    game.choose(0);
    expect(game.frame?.sceneId).toBe(scene);

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await flushPromises();
    expect(shell.overlay).toBe('closed');
    expect(document.activeElement).toBe(wrapper.find('[data-test="pause-button"]').element);
  });

  it('restores the exact captured scene state after pause Achievements and ESC', async () => {
    const { game, shell, wrapper } = await mountTitle();
    await ribbon(wrapper, 'New Game').trigger('click');
    await wrapper.find('[data-test="pause-button"]').trigger('click');
    await ribbon(wrapper, 'Achievements').trigger('click');
    expect(game.frame?.sceneId).toBe('achievements');
    expect(wrapper.text()).toContain('Ledger');
    expect(wrapper.get('[data-test="paused-surface-snapshot"]').text()).toContain('Setup');
    expect(wrapper.get('[data-test="pane-heading"]').text()).toBe('Filed Achievements');
    expect(wrapper.findAll('[data-test="step-aside-pane"] h1')).toHaveLength(1);

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await nextTick();
    expect(shell.overlay).toBe('pause-menu');
    expect(game.frame?.sceneId).toBe('start');
    expect(game.q.started).toBe(true);
  });

  it('opens the pause Save/Load and Settings panes through the shared components', async () => {
    const { wrapper } = await mountTitle();
    await ribbon(wrapper, 'New Game').trigger('click');
    await wrapper.find('[data-test="pause-button"]').trigger('click');

    await ribbon(wrapper, 'Save & Load').trigger('click');
    expect(wrapper.find('[data-test="save-load-pane"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="save-new"]').exists()).toBe(true);
    await ribbon(wrapper, 'Settings').trigger('click');
    expect(wrapper.find('[data-test="options-pane"]').exists()).toBe(true);
  });

  it('disables the pause Save & Load ribbon for an active ironman state', async () => {
    const { game, wrapper } = await mountTitle();
    await ribbon(wrapper, 'New Game').trigger('click');
    game.adapter!.engine.state.disableSaves = true;
    await wrapper.find('[data-test="pause-button"]').trigger('click');

    const saveRibbon = wrapper.get('[data-test="ribbon-load"]');
    expect(saveRibbon.attributes('disabled')).toBeDefined();
    expect(saveRibbon.attributes('title')).toContain('latest recovery autosave');
  });

  it('uses an in-shell exit confirmation and returns through fresh root initialization', async () => {
    const { game, shell, wrapper } = await mountTitle();
    await ribbon(wrapper, 'New Game').trigger('click');
    await wrapper.find('[data-test="pause-button"]').trigger('click');
    await ribbon(wrapper, 'Exit to Main Menu').trigger('click');
    expect(shell.overlay).toBe('exit-confirm');
    expect(wrapper.find('[data-test="step-aside-pane"]').text()).toContain('Exit to Main Menu?');
    const confirm = wrapper.findAll('[data-test="step-aside-pane"] button').find((node) => node.text() === 'Exit to Main Menu');
    expect(confirm).toBeDefined();
    await confirm!.trigger('click');
    expect(shell.mode).toBe('title');
    expect(game.frame?.sceneId).toBe('menu');
    expect(game.q.started).toBeUndefined();
  });
});
