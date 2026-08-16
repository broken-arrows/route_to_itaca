import { describe, it, expect, beforeEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createI18n } from 'vue-i18n';
import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import Clipboard from '../src/components/brief/Clipboard.vue';
import { useGameStore } from '../src/stores/game';
import uiEn from '../../source/locales/en/ui.json';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');

// The tab labels/context strings are GAME chrome (source/locales), merged
// over the bundled defaults at boot by initGameLocale(); tests merge the
// catalog directly, same as the old ClipboardFrame test did.
function mountClipboard() {
  const i18n = createI18n({ legacy: false, locale: 'en', messages: { en: uiEn as never } });
  return mount(Clipboard, { global: { plugins: [i18n] } });
}

describe('Clipboard (live, phase 3b)', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const g = useGameStore();
    g.initFromText(readFileSync(GAME, 'utf8'));
    g.newGame();
  });

  it('renders the authored Library entry as the gold seventh tab', () => {
    const tabs = mountClipboard().findAll('[data-test="brief-tab"]');
    expect(tabs).toHaveLength(7);
    expect(tabs.map((t) => t.text())).toEqual([
      'Overview', 'Party', 'Chamber', 'Economy', 'World', 'Polls', '▤ LIBRARY',
    ]);
    expect(tabs[6].classes()).toContain('tab-gold');
  });

  it('defaults to OVERVIEW active, with real sheet values rendered through Prose', () => {
    const w = mountClipboard();
    const tabs = w.findAll('[data-test="brief-tab"]');
    expect(tabs[0].classes()).toContain('tab-active');
    expect(w.find('h2').text()).toBe('Overview');
    // Real content from renderView('status_new.overview'), not a placeholder --
    // proves the sheet is wired to the live engine, not stubbed text.
    expect(w.text()).toMatch(/Party resources/);
    // Prose, not a bare v-html: it marks glossary terms and mounts widgets;
    // a bare v-html would still show the text but silently drop both -- so
    // assert on the actual component tree, not just the rendered string.
    expect(w.findComponent({ name: 'Prose' }).exists()).toBe(true);
  });

  it('clicking a tab adds the active class and swaps the rendered sheet', async () => {
    const w = mountClipboard();
    const tabs = w.findAll('[data-test="brief-tab"]');
    await tabs[2].trigger('click'); // Chamber
    expect(tabs[2].classes()).toContain('tab-active');
    expect(tabs[0].classes()).not.toContain('tab-active');
    expect(w.find('h2').text()).toBe('Chamber');
    expect(w.text()).toMatch(/Speaker of the House/);
  });

  it('the Library tab enters the live special scene and renders its authored index', async () => {
    const w = mountClipboard();
    const tabs = w.findAll('[data-test="brief-tab"]');
    const library = tabs[6];
    await library.trigger('click');

    expect(useGameStore().effectiveRole).toBe('library-item');
    expect(library.classes()).toContain('tab-active');
    expect(w.find('[data-test="library-index"]').exists()).toBe(true);
    expect(w.text()).toMatch(/Catalan Political System/);
  });

  it('multi-root safety: asserts on its own root node, not wrapper.element', () => {
    // Fragment roots break wrapper.classes() -- LEARNINGS 2026-07-17 §2.
    expect(mountClipboard().find('.clipboard-frame').classes()).toContain('clipboard-frame');
  });
});
