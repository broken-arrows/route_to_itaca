import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import { nextTick } from 'vue';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import App from '../src/App.vue';
import GameView from '../src/views/GameView.vue';
import DebugPage from '../src/views/DebugPage.vue';
import DeskView from '../src/views/DeskView.vue';
import PaperPage from '../src/components/desk/PaperPage.vue';
import StageScaler from '../src/components/StageScaler.vue';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';

let pinia: ReturnType<typeof createPinia>;
beforeEach(() => {
  pinia = createPinia();
  setActivePinia(pinia);
  setAnimationsForTest(false);
});

function withPlugins(extra: Record<string, unknown> = {}) {
  return { global: { plugins: [pinia, i18n] }, ...extra };
}

// Minimal fixture that reaches a real 'idle' desk phase (root -> hub, hub
// declares role: desk) — every OTHER phase in the routing table is forced
// directly on the desk/game store, same pattern as
// desk.dossier.test.ts's "DeskView phase wiring" block.
const routingGame = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      newPage: true,
      onArrival: [{ $code: 'Q.month = 3; Q.year = 2014;' }],
      content: [{ type: 'paragraph', content: ['Root.'] }],
      options: [{ id: '@hub' }],
    },
    hub: {
      id: 'hub',
      type: 'scene',
      title: 'Hub',
      newPage: true,
      isHand: true,
      maxCards: 4,
      role: 'desk',
      content: [{ type: 'paragraph', content: ['Hub.'] }],
      options: [],
    },
  },
  qualities: {},
  qdisplays: {},
  tagLookup: {},
};

function mountAtHub() {
  const game = useGameStore();
  const desk = useDeskStore();
  game.initFromText(JSON.stringify(routingGame));
  game.newGame(); // root ('page' effectiveRole, the engine's initial default)
  game.choose(0); // -> hub (role: desk -> idle)
  const wrapper = mount(GameView, withPlugins());
  return { game, desk, wrapper };
}

describe('GameView phase routing', () => {
  it('routes idle/drawing/dossierOpen/resolving to DeskView', async () => {
    const { desk, wrapper } = mountAtHub();
    expect(desk.phase).toBe('idle');
    expect(wrapper.findComponent(DeskView).exists()).toBe(true);
    expect(wrapper.findComponent(PaperPage).exists()).toBe(false);

    for (const phase of ['drawing', 'dossierOpen', 'resolving'] as const) {
      desk.phase = phase;
      await nextTick();
      expect(wrapper.findComponent(DeskView).exists()).toBe(true);
      expect(wrapper.findComponent(PaperPage).exists()).toBe(false);
    }
  });

  it("routes 'eventPage' to PaperPage variant=event", async () => {
    const { desk, wrapper } = mountAtHub();
    desk.phase = 'eventPage';
    await nextTick();
    expect(wrapper.findComponent(DeskView).exists()).toBe(false);
    const page = wrapper.findComponent(PaperPage);
    expect(page.exists()).toBe(true);
    expect(page.props('variant')).toBe('event');
  });

  it("routes 'page' with a non-ending scene to PaperPage variant=page", async () => {
    const { game, desk, wrapper } = mountAtHub();
    game.frame!.effectiveRole = 'page';
    desk.phase = 'page';
    await nextTick();
    const page = wrapper.findComponent(PaperPage);
    expect(page.exists()).toBe(true);
    expect(page.props('variant')).toBe('page');
  });

  it("routes 'page' with an ending scene to PaperPage variant=ending", async () => {
    const { game, desk, wrapper } = mountAtHub();
    game.frame!.effectiveRole = 'ending';
    desk.phase = 'page';
    await nextTick();
    const page = wrapper.findComponent(PaperPage);
    expect(page.exists()).toBe(true);
    expect(page.props('variant')).toBe('ending');
  });

  it("shows the loading state while 'boot' and not yet loadError", () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {}))); // never resolves
    const wrapper = mount(GameView, withPlugins());
    expect(wrapper.find('[data-test="boot-state"]').exists()).toBe(true);
    expect(wrapper.text()).toContain(i18n.global.t('debug.loading'));
    vi.unstubAllGlobals();
  });

  it('shows the error state when the game data fetch fails', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('no network in test'))));
    const wrapper = mount(GameView, withPlugins());
    await flushPromises();
    expect(wrapper.text()).toContain(i18n.global.t('debug.loadError'));
    errSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  it('calls newGame once ready with no frame yet, landing on a real phase', async () => {
    const game = useGameStore();
    game.initFromText(JSON.stringify(routingGame)); // ready, no frame
    const wrapper = mount(GameView, withPlugins());
    await flushPromises();
    expect(game.frame).not.toBeNull();
    expect(wrapper.find('[data-test="boot-state"]').exists()).toBe(false);
  });

  // REGRESSION (C2, spec §9): newGame() was uncaught inside an async
  // onMounted. A game whose root scene cannot be reached makes the engine
  // throw (__changeScene asserts the scene exists), and the rejection was
  // swallowed by the unawaited promise: the phase stayed 'boot' and the UI
  // showed "Loading…" FOREVER, with loadError still false. It must surface.
  it('shows the error state (not an eternal Loading…) when the engine throws on newGame', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const game = useGameStore();
    // No 'root' scene: beginGame -> goToScene('root') -> assert -> throw.
    game.initFromText(
      JSON.stringify({ scenes: {}, qualities: {}, qdisplays: {}, tagLookup: {} }),
    );
    const wrapper = mount(GameView, withPlugins());
    await flushPromises();

    expect(game.frame).toBeNull();
    expect(game.loadError).toBe(true);
    expect(wrapper.text()).toContain(i18n.global.t('debug.loadError'));
    expect(wrapper.text()).not.toContain(i18n.global.t('debug.loading'));
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  // DeskView/PaperPage are authored in 1512x860 design-space absolute
  // pixels — they only render correctly inside phase 1's StageScaler
  // (ui/src/components/StageScaler.vue, default slot inside its .stage
  // element). `.stage-viewport` is StageScaler's own root class (a
  // load-bearing style hook, stable), asserted alongside subtree
  // containment via findComponent chaining.
  it('renders DeskView inside the StageScaler stage', () => {
    const { desk, wrapper } = mountAtHub();
    expect(desk.phase).toBe('idle');
    expect(wrapper.find('.stage-viewport').exists()).toBe(true);
    const stage = wrapper.findComponent(StageScaler);
    expect(stage.exists()).toBe(true);
    expect(stage.findComponent(DeskView).exists()).toBe(true);
  });

  it('renders PaperPage inside the StageScaler stage', async () => {
    const { desk, wrapper } = mountAtHub();
    desk.phase = 'eventPage';
    await nextTick();
    expect(wrapper.find('.stage-viewport').exists()).toBe(true);
    const stage = wrapper.findComponent(StageScaler);
    expect(stage.exists()).toBe(true);
    expect(stage.findComponent(PaperPage).exists()).toBe(true);
  });
});

// PaperPage fixture: a plain (role-less, hence 'page') scene with one
// choosable and one chooseIf-locked option — chooseIf keeps the option
// VISIBLE with canChoose:false (distinct from viewIf, which would drop it),
// exactly what a "locked option is ignored, not dropped" test needs.
const pageGame = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      newPage: true,
      content: [{ type: 'paragraph', content: ['A page of prose.'] }],
      options: [{ id: '@go_open' }, { id: '@go_locked' }],
    },
    go_open: {
      id: 'go_open',
      type: 'scene',
      title: 'Open path',
      subtitle: '[0 resources]',
      content: [],
      goTo: [{ id: 'root' }],
    },
    go_locked: {
      id: 'go_locked',
      type: 'scene',
      title: 'Locked path',
      subtitle: '[9 resources]',
      chooseIf: { $code: 'return false;' },
      content: [],
      goTo: [{ id: 'root' }],
    },
  },
  qualities: {},
  qdisplays: {},
  tagLookup: {},
};

function mountPage(variant: 'page' | 'event' | 'ending' = 'page') {
  const game = useGameStore();
  game.initFromText(JSON.stringify(pageGame));
  game.newGame();
  const wrapper = mount(PaperPage, withPlugins({ props: { variant } }));
  return { game, wrapper };
}

describe('PaperPage', () => {
  it('renders the scene prose via v-html', () => {
    const { wrapper } = mountPage();
    expect(wrapper.html()).toContain('A page of prose.');
  });

  it('shows the red event band for the event variant', () => {
    const { wrapper } = mountPage('event');
    expect(wrapper.find('[data-test="event-band"]').exists()).toBe(true);
  });

  it('has no red event band for the page variant', () => {
    const { wrapper } = mountPage('page');
    expect(wrapper.find('[data-test="event-band"]').exists()).toBe(false);
  });

  it('has no red event band for the ending variant', () => {
    const { wrapper } = mountPage('ending');
    expect(wrapper.find('[data-test="event-band"]').exists()).toBe(false);
  });

  it('shows the ended stamp only for the ending variant', () => {
    const { wrapper } = mountPage('ending');
    expect(wrapper.find('[data-test="ended-stamp"]').exists()).toBe(true);
    expect(wrapper.text()).toContain(i18n.global.t('desk.page.ended'));
  });

  it('has no ended stamp for the page variant', () => {
    const { wrapper } = mountPage('page');
    expect(wrapper.find('[data-test="ended-stamp"]').exists()).toBe(false);
  });

  it('renders one PaperOption per choice', () => {
    const { wrapper } = mountPage();
    expect(wrapper.find('[data-test="paper-option-0"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="paper-option-1"]').exists()).toBe(true);
    expect(wrapper.text()).toContain('Open path');
    expect(wrapper.text()).toContain('Locked path');
  });

  it('clicking a choosable option calls game.choose with its index', async () => {
    const { game, wrapper } = mountPage();
    const spy = vi.spyOn(game, 'choose');
    await wrapper.find('[data-test="paper-option-0"]').trigger('click');
    expect(spy).toHaveBeenCalledWith(0);
  });

  it('clicking a locked option does not call game.choose', async () => {
    const { game, wrapper } = mountPage();
    const spy = vi.spyOn(game, 'choose');
    await wrapper.find('[data-test="paper-option-1"]').trigger('click');
    expect(spy).not.toHaveBeenCalled();
  });
});

// window.location.search stubbing: Object.defineProperty(window, 'location', ...)
// FAILS in this repo's jsdom (26.x) — `location`'s own property descriptor is
// configurable:false (verified directly against the vendored jsdom package),
// so redefining it throws "Cannot redefine property: location". history.pushState
// is the standards-compliant way to change window.location.search without a
// real navigation, and jsdom implements it faithfully — no property-descriptor
// gymnastics needed.
describe('App routing (?debug)', () => {
  afterEach(() => {
    window.history.pushState({}, '', '/');
  });

  it('renders DebugPage when ?debug is present', async () => {
    window.history.pushState({}, '', '/?debug');
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('no network in test'))));

    const wrapper = mount(App, withPlugins());
    await flushPromises();

    expect(wrapper.findComponent(DebugPage).exists()).toBe(true);
    expect(wrapper.findComponent(GameView).exists()).toBe(false);

    errSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  it('renders GameView when ?debug is absent', async () => {
    window.history.pushState({}, '', '/');
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('no network in test'))));

    const wrapper = mount(App, withPlugins());
    await flushPromises();

    expect(wrapper.findComponent(GameView).exists()).toBe(true);
    expect(wrapper.findComponent(DebugPage).exists()).toBe(false);

    errSpy.mockRestore();
    vi.unstubAllGlobals();
  });
});
