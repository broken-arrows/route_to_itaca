import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { defineComponent, h, nextTick } from 'vue';
import WidgetHost from '../src/components/viz/WidgetHost.vue';

// Stub widget for the deriveFrom tests below (`test-rows` is not a real
// WidgetName — it never ships — it just needs to exist in the registry so
// WidgetHost has something to mount). Merged with the REAL registry via
// `importOriginal` rather than replacing WIDGETS outright: `vi.mock` swaps
// the module for the WHOLE file, not just this describe block, and the
// 'hemicycle' test above mounts the real registry entry — replacing it
// wholesale would break that test for an unrelated reason.
vi.mock('../src/components/viz/registry', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/components/viz/registry')>();
  const TestRows = defineComponent({
    name: 'TestRows',
    props: { rows: { type: Array, default: () => [] } },
    setup: (p) => () => h('div', { 'data-rows': p.rows.length }),
  });
  return { ...actual, WIDGETS: { ...actual.WIDGETS, 'test-rows': TestRows } };
});

// Stubs `G.brief` (source/lib/brief.js, Wave 2 — not written yet) so the
// first deriveFrom test doesn't need a booted engine just to prove the
// plumbing calls through to the game lib. `broken` is added alongside
// `control` for the "derivation throws" test below — it needs a builder
// function that actually throws, not a hand-rolled substitute for one.
vi.mock('../src/game-bindings', () => ({
  gameLib: {
    brief: {
      control: () => [{ id: 'airports', label: 'Airports', value: 0 }],
      broken: () => {
        throw new Error('derivation boom');
      },
    },
  },
}));

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

describe('WidgetHost deriveFrom', () => {
  it('resolves deriveFrom by calling the game lib and passes rows', async () => {
    const w = mountHost({ name: 'test-rows', props: { deriveFrom: 'control' } });
    await nextTick();
    const child = w.findComponent({ name: 'TestRows' });
    const rows = child.props('rows') as unknown[];
    expect(Array.isArray(rows)).toBe(true);
    // Assert on the actual derived content, not just "is an array" — `rows`
    // defaults to `[]` on TestRows even with zero deriveFrom handling, so a
    // bare Array.isArray check passes vacuously before this feature exists.
    // This is what makes the test fail for the right reason pre-implementation.
    expect(rows).toEqual([{ id: 'airports', label: 'Airports', value: 0 }]);
    // deriveFrom itself must NOT reach the component — the widget never
    // learns where its props came from.
    expect(child.props()).not.toHaveProperty('deriveFrom');
  });

  it('renders the placeholder for an unknown derivation rather than throwing', async () => {
    const w = mountHost({ name: 'test-rows', props: { deriveFrom: 'nope' } });
    await nextTick();
    expect(w.find('[data-widget-missing]').exists()).toBe(true);
  });

  it('renders the placeholder rather than propagating when a derivation throws', async () => {
    // `onErrorCaptured` cannot see this: `fn(q.value)` runs inside
    // WidgetHost's OWN `resolved` computed, not a descendant component, so a
    // throwing derivation must be caught by WidgetHost itself. Mounting
    // (rather than a bare `mount(...)` expected to throw) is the point —
    // this proves the exception never escapes past `resolved` at all.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const w = mountHost({ name: 'test-rows', props: { deriveFrom: 'broken' } });
    await nextTick();
    expect(w.find('[data-widget-missing]').exists()).toBe(true);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
