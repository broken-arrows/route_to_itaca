import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { defineComponent, nextTick } from 'vue';
import WidgetHost from '../src/components/viz/WidgetHost.vue';

function mountHost(props: { name: string; props?: Record<string, unknown> }) {
  const pinia = createPinia();
  setActivePinia(pinia);
  return mount(WidgetHost, { props, global: { plugins: [pinia] } });
}

describe('WidgetHost', () => {
  it('mounts the registered component and passes parsed props', () => {
    const w = mountHost({ name: 'hemicycle', props: { seatsKey: 'x' } });
    expect(w.findComponent({ name: 'Hemicycle' }).exists()).toBe(true);
  });

  it('renders the striped placeholder for an unknown widget — never a broken div', () => {
    const w = mountHost({ name: 'nope', props: {} });
    expect(w.find('.widget-placeholder').exists()).toBe(true);
  });

  describe('a widget that throws', () => {
    afterEach(() => {
      vi.doUnmock('../src/components/viz/registry');
      vi.resetModules();
    });

    it('renders the placeholder and does not blank the sheet', async () => {
      // Swap the registry for one whose only entry throws during setup, so
      // this exercises WidgetHost's REAL onErrorCaptured wiring — not a
      // hand-rolled replica of it. vi.resetModules() + a fresh dynamic
      // import is required (not a static import) because WidgetHost.vue's
      // own `import { WIDGETS } from './registry'` was already resolved
      // against the real module by the two tests above in this same file;
      // only a fresh module graph picks up the mock.
      vi.resetModules();
      vi.doMock('../src/components/viz/registry', () => ({
        // A `<script setup>` SFC compiles to `setup()` RETURNING the render
        // function — so a throw before that return, with no separate
        // `render` option, leaves the component with nothing to render at
        // all. That is what makes this a faithful stand-in for a real widget
        // SFC that throws, unlike a setup+separate-render mix (which would
        // still render via the untouched `render` option even after
        // errorCaptured swallows the setup error).
        WIDGETS: {
          throws: defineComponent({
            name: 'ThrowsInSetup',
            setup() {
              throw new Error('boom');
            },
          }),
        },
      }));
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const { default: FreshWidgetHost } = await import('../src/components/viz/WidgetHost.vue');

      const pinia = createPinia();
      setActivePinia(pinia);
      const w = mount(FreshWidgetHost, {
        props: { name: 'throws', props: {} },
        global: { plugins: [pinia] },
      });
      await nextTick();

      expect(w.find('.widget-placeholder').exists()).toBe(true);
      expect(w.find('[data-widget-missing="throws"]').exists()).toBe(true);
      expect(warn).toHaveBeenCalled();
      warn.mockRestore();
    });
  });
});
