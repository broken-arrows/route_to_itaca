import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import DebugPage from '../src/views/DebugPage.vue';
import { useGameStore } from '../src/stores/game';
import { i18n } from '../src/i18n';
import { miniGame, miniGameText } from './fixtures/mini-game';
import { markGlossary, type GlossaryTerm } from '../src/glossary/mark';

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

// REGRESSION (Task 5, same class as OpenDossier's cover-title fix): decks/
// hand cards/pinned cards all reach DebugPage via CaptureUI's
// normalizeCard/normalizeChoice -> convertLine, exactly like choice
// title/subtitle above, so their title can arrive already glossary-marked.
// Installs the real window.displayText hook (main.ts's own wiring) so this
// exercises the actual marking path, not a stand-in.
describe('DebugPage — a pinned card title the engine itself marked', () => {
  const TERMS: GlossaryTerm[] = [
    { id: 'advisor', match: ['The Advisor'], display: 'The Advisor', colour: 'erc' },
  ];
  const markedGame = {
    ...miniGame,
    scenes: {
      ...miniGame.scenes,
      advisor_note: { ...miniGame.scenes.advisor_note, title: 'The Advisor' },
    },
    data: { glossary: { terms: TERMS } },
  };

  beforeEach(() => {
    window.displayText = (text: string) => markGlossary(text, TERMS);
  });
  afterEach(() => {
    delete (window as { displayText?: unknown }).displayText;
  });

  it('renders the marked pinned-card title as an element, not literal tag text', async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const store = useGameStore();
    store.initFromText(JSON.stringify(markedGame));
    store.newGame();
    const wrapper = mount(DebugPage, { global: { plugins: [pinia, i18n] } });
    await wrapper.find('[data-test="choice"]').trigger('click'); // root -> desk

    const pinned = wrapper.get('[data-test="pinned"]');
    expect(pinned.text()).toBe('The Advisor');
    expect(pinned.find('[data-term="advisor"]').exists()).toBe(true);
    expect(wrapper.html()).not.toContain('&lt;span');
  });
});
