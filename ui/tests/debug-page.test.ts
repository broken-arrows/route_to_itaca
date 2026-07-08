import { describe, it, expect, beforeEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import DebugPage from '../src/views/DebugPage.vue';
import { useGameStore } from '../src/stores/game';
import { i18n } from '../src/i18n';
import { miniGameText } from './fixtures/mini-game';

describe('DebugPage', () => {
  let pinia: ReturnType<typeof createPinia>;
  beforeEach(() => {
    pinia = createPinia();
    setActivePinia(pinia);
  });

  function mountPage() {
    const store = useGameStore();
    store.initFromText(miniGameText);
    store.newGame();
    const wrapper = mount(DebugPage, { global: { plugins: [pinia, i18n] } });
    return { store, wrapper };
  }

  it('renders scene prose and choices', () => {
    const { wrapper } = mountPage();
    expect(wrapper.html()).toContain('Welcome to the mini game.');
    const buttons = wrapper.findAll('[data-test="choice"]');
    expect(buttons).toHaveLength(1);
    expect(buttons[0].text()).toContain('The Desk');
  });

  it('clicking a choice advances the game and shows hand surfaces', async () => {
    const { wrapper } = mountPage();
    await wrapper.find('[data-test="choice"]').trigger('click');
    expect(wrapper.find('[data-test="deck"]').text()).toContain('Government');
    expect(wrapper.find('[data-test="pinned"]').text()).toContain('Advisor');
  });

  it('drawing and playing a card works end to end', async () => {
    const { wrapper } = mountPage();
    await wrapper.find('[data-test="choice"]').trigger('click');
    await wrapper.find('[data-test="deck"]').trigger('click');
    const card = wrapper.find('[data-test="hand-card"]');
    expect(card.exists()).toBe(true);
    await card.trigger('click');
    expect(wrapper.findAll('[data-test="choice"]').length).toBeGreaterThan(0);
  });

  it('Q inspector filters qualities', async () => {
    const { wrapper } = mountPage();
    await wrapper.find('[data-test="q-filter"]').setValue('gold');
    const rows = wrapper.findAll('[data-test="q-row"]');
    expect(rows).toHaveLength(1);
    expect(rows[0].text()).toContain('gold');
    expect(rows[0].text()).toContain('2');
  });
});
