import { describe, expect, it, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import Prose from '../src/components/Prose.vue';
import { useGameStore } from '../src/stores/game';
import { i18n } from '../src/i18n';

// No @pinia/testing in this repo (per the task's own ambiguity-resolution
// note: fall back to a real pinia + a seeded store). The store is seeded
// through the SAME path production uses — DendryAdapter.fromJSONText — so
// `game.glossary`/`game.q` are the real computeds, not stand-ins. `data.
// glossary.terms` is added directly to a hand-authored "compiled game"
// object literal, exactly like tests/fixtures/mini-game.ts already does for
// scenes: convertJSONToGame is a plain JSON.parse, so it doesn't care that
// this bypassed the real source/data -> compiler route.
const GAME = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      // artur_mas's allegiances (source/lib/allegiances.js) branch on these —
      // set here so the plural-allegiances test exercises live Q, not a stub.
      onArrival: [{ $code: "Q.dl_formed = true; Q.jxsi_formed = true; Q.ciu_leader = 'Someone';" }],
      content: [{ type: 'paragraph', content: ['Hi.'] }],
      options: [],
    },
  },
  qualities: {},
  qdisplays: {},
  data: {
    glossary: {
      terms: [
        { id: 'ciu', match: ['CiU'], display: 'CiU', colour: 'ciu' },
        {
          id: 'llu_s_companys',
          match: ['Companys'],
          display: 'Companys',
          colour: 'erc',
          tooltip: { title: 'Lluís Companys', q: {} },
        },
        {
          id: 'artur_mas',
          match: ['Mas'],
          display: 'Mas',
          colour: 'cdc',
          tooltip: {
            title: 'Artur Mas i Gavarró',
            q: { ledBy: 'ciu_leader', ideology: 'Center-right Liberalism' },
          },
        },
      ],
    },
  },
};

function mountProse(html: string) {
  const pinia = createPinia();
  setActivePinia(pinia);
  const store = useGameStore();
  store.initFromText(JSON.stringify(GAME));
  store.newGame();
  return mount(Prose, { props: { html }, global: { plugins: [pinia, i18n] } });
}

describe('Prose', () => {
  afterEach(() => {
    document.body.innerHTML = ''; // teleported popovers land on the real body
  });

  // Prose.vue has TWO root nodes (the prose div + the conditional popover),
  // so Vue does not auto-forward a caller's class the way a single-root SFC
  // would — every current call site passes its own layout class
  // (cover-prose/option-title/prose), so a silently-dropped class would
  // break real layout with zero jsdom signal (CSS isn't applied in jsdom
  // regardless). Assert the forwarding explicitly.
  it('forwards a caller class onto the prose root (multi-root fallthrough)', () => {
    // wrapper.element/.classes() resolve to VTU's own mount container for a
    // fragment (multi-root) component, not the component's own root — find
    // the actual `.prose` div and check its classes directly.
    const wrapper = mount(Prose, {
      props: { html: 'x' },
      attrs: { class: 'cover-prose' },
      global: { plugins: [createPinia(), i18n] },
    });
    const div = wrapper.find('.prose');
    expect(div.classes()).toContain('cover-prose');
    expect(div.classes()).toContain('prose');
  });

  it('renders engine HTML and colours a glossary term from its token', () => {
    const wrapper = mountProse('<span class="term" data-term="ciu">CiU</span> governs.');
    const term = wrapper.get('[data-term="ciu"]');
    expect(term.attributes('style')).toContain('var(--ciu)');
  });

  it('leaves an unknown term unstyled rather than throwing', () => {
    const wrapper = mountProse('<span class="term" data-term="nope">X</span>');
    expect(wrapper.get('[data-term="nope"]').attributes('style')).toBeUndefined();
  });

  it('hovering a tooltip-bearing term opens a paper popover with its title', async () => {
    const wrapper = mountProse('<span class="term" data-term="llu_s_companys">Companys</span>');
    await wrapper.get('[data-term="llu_s_companys"]').trigger('mouseover');
    const popover = document.querySelector('[data-test="glossary-popover"]');
    expect(popover).not.toBeNull();
    expect(popover!.textContent).toContain('Lluís Companys');
  });

  it('a single allegiance renders the SINGULAR "Allegiance:" label, coloured by its token', async () => {
    const wrapper = mountProse('<span class="term" data-term="llu_s_companys">Companys</span>');
    await wrapper.get('[data-term="llu_s_companys"]').trigger('mouseover');
    const line = document.querySelector('[data-test="popover-allegiances"]');
    expect(line!.textContent).toContain('Allegiance:');
    expect(line!.textContent).not.toContain('Allegiances:');
    expect(line!.textContent).toContain('ERC');
    const ercSpan = Array.from(line!.querySelectorAll('span')).find((s) => s.textContent === 'ERC');
    expect(ercSpan!.getAttribute('style')).toContain('var(--erc)');
  });

  it('multiple allegiances render the PLURAL "Allegiances:" label and the live "Leader:" Q value', async () => {
    const wrapper = mountProse('<span class="term" data-term="artur_mas">Mas</span>');
    await wrapper.get('[data-term="artur_mas"]').trigger('mouseover');
    const popover = document.querySelector('[data-test="glossary-popover"]')!;
    const line = document.querySelector('[data-test="popover-allegiances"]')!;
    expect(line.textContent).toContain('Allegiances:');
    expect(line.textContent).toContain('CDC');
    expect(line.textContent).toContain('CiU');
    expect(line.textContent).toContain('DL');
    expect(line.textContent).toContain('JxSí');
    // ledBy: Q.ciu_leader was set to 'Someone' by root's onArrival above.
    expect(popover.textContent).toContain('Leader:');
    expect(popover.textContent).toContain('Someone');
  });

  it('moving the pointer off the term closes the popover', async () => {
    const wrapper = mountProse(
      '<span class="term" data-term="llu_s_companys">Companys</span> <span id="plain">plain</span>',
    );
    const term = wrapper.get('[data-term="llu_s_companys"]');
    await term.trigger('mouseover');
    expect(document.querySelector('[data-test="glossary-popover"]')).not.toBeNull();

    const plain = wrapper.get('#plain').element;
    await term.trigger('mouseout', { relatedTarget: plain });
    expect(document.querySelector('[data-test="glossary-popover"]')).toBeNull();
  });

  it('re-decorates and closes any open popover when html changes', async () => {
    const wrapper = mountProse('<span class="term" data-term="llu_s_companys">Companys</span>');
    await wrapper.get('[data-term="llu_s_companys"]').trigger('mouseover');
    expect(document.querySelector('[data-test="glossary-popover"]')).not.toBeNull();

    await wrapper.setProps({ html: '<span class="term" data-term="ciu">CiU</span>' });
    expect(document.querySelector('[data-test="glossary-popover"]')).toBeNull();
    expect(wrapper.get('[data-term="ciu"]').attributes('style')).toContain('var(--ciu)');
  });

  // Task 6: Prose hosts `[data-widget]` markers left by engine-authored HTML
  // — this REPLACES the `mount-registry` sketch in desk_ui_plan.md §6.
  //
  // WidgetHost is mounted via the low-level `render()` API into the marker
  // element directly (see Prose.vue's `mountWidgets`) rather than as a vnode
  // child of Prose's own render output, so it is a SEPARATE render root —
  // `wrapper.findComponent()` (which walks Prose's own internal component
  // tree) cannot see into it. Assert on the real DOM it produces instead
  // (exactly how the existing glossary tests already treat v-html content).
  describe('widget hosting', () => {
    it('mounts the registered component for a data-widget marker', () => {
      // hemicycle stopped being a stub in phase 2.5 Task 7 — assert on the
      // real Hemicycle component's own root instead of the retired
      // `[data-widget-stub]` marker (`data-props` here carries no real
      // `seats`, so it renders its empty-state <svg>, which is exactly what
      // this test needs: proof Prose mounted the REGISTERED component).
      const wrapper = mountProse('<div data-widget="hemicycle" data-props=\'{"seatsKey":"x"}\'></div>');
      expect(wrapper.find('svg.hemicycle').exists()).toBe(true);
    });

    it('renders the striped placeholder for an unregistered widget name', () => {
      const wrapper = mountProse('<div data-widget="nope"></div>');
      expect(wrapper.find('.widget-placeholder').exists()).toBe(true);
      expect(wrapper.find('[data-widget-missing="nope"]').exists()).toBe(true);
    });

    it('unmounts the old widget and mounts the new one when html changes', async () => {
      // hemicycle (real, Task 7) → achievement-gallery (real, Task 8) —
      // deliberately mismatched so the two assertions below can only pass
      // if the OLD widget actually unmounted, not merely if a second one
      // mounted alongside it. No achievements registry on this test's
      // fixture game (data.achievements is absent), so the gallery renders
      // its real empty state (mounted, zero rows) — exactly what proves
      // Prose mounted the REGISTERED component, not a placeholder.
      const wrapper = mountProse('<div data-widget="hemicycle"></div>');
      expect(wrapper.find('svg.hemicycle').exists()).toBe(true);

      await wrapper.setProps({ html: '<div data-widget="achievement-gallery"></div>' });
      expect(wrapper.find('svg.hemicycle').exists()).toBe(false);
      expect(wrapper.find('[data-test="achievement-gallery"]').exists()).toBe(true);
    });

    it('does not parse configFrom itself — passes the marker straight to WidgetHost', () => {
      // WidgetHost is the ONLY thing that resolves `configFrom` against Q
      // (its own header comment states the invariant). This just proves
      // Prose does not choke on / strip a configFrom-bearing marker — the
      // resolution itself is WidgetHost's own responsibility, covered by
      // widget.host.test.ts. Asserting on the real Hemicycle root also
      // proves the component tolerates a configFrom key with nothing behind
      // it in Q (this mount's pinia store never set
      // `parlament_coalition_config`) without throwing.
      const wrapper = mountProse(
        '<div data-widget="hemicycle" data-props=\'{"configFrom":"parlament_coalition_config"}\'></div>',
      );
      expect(wrapper.find('svg.hemicycle').exists()).toBe(true);
    });
  });
});
