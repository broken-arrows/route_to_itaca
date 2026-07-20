import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createI18n } from 'vue-i18n';
import ClipboardFrame from '../src/components/brief/ClipboardFrame.vue';
import uiEn from '../../source/locales/en/ui.json';

// The tab labels are GAME chrome (source/locales), merged over the bundled
// defaults at boot by initGameLocale(); tests merge the catalog directly.
function mountFrame() {
  const i18n = createI18n({ legacy: false, locale: 'en', messages: { en: uiEn as never } });
  return mount(ClipboardFrame, { global: { plugins: [i18n] } });
}

describe('ClipboardFrame (inert, phase 3a)', () => {
  it('renders the seven tabs in order, LIBRARY last and gold', () => {
    const tabs = mountFrame().findAll('[data-test="brief-tab"]');
    expect(tabs.map((t) => t.text())).toEqual([
      'OVERVIEW', 'PARTY', 'CHAMBER', 'ECONOMY', 'WORLD', 'POLLS', '▤ LIBRARY',
    ]);
    expect(tabs[6].classes()).toContain('tab-gold');
  });

  it('is inert: no buttons, no click targets', () => {
    const w = mountFrame();
    expect(w.find('button').exists()).toBe(false);
    expect(w.find('[role="button"]').exists()).toBe(false);
  });

  it('multi-root safety: asserts on its own root node, not wrapper.element', () => {
    // Single-root component — this locks that in (fragment roots break
    // wrapper.classes(); see LEARNINGS 2026-07-17 §2).
    expect(mountFrame().find('.clipboard-frame').classes()).toContain('clipboard-frame');
  });
});
