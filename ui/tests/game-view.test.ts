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
import Newspaper from '../src/components/desk/Newspaper.vue';
import FrontPage from '../src/components/desk/FrontPage.vue';
import ResponsiveViewport from '../src/components/ResponsiveViewport.vue';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import uiEn from '../../source/locales/en/ui.json';

// Every DeskView mount below now also mounts Clipboard (phase 3b Task 9;
// formerly the inert ClipboardFrame), which reads `brief.context.*` — GAME
// chrome sourced from source/locales/<loc>/ui.json (see i18n.ts's
// initGameLocale, and the same fix in desk.components.test.ts).
i18n.global.mergeLocaleMessage('en', uiEn as never);

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

  it("routes 'newspaper' to Newspaper", async () => {
    const { desk, wrapper } = mountAtHub();
    desk.phase = 'newspaper';
    await nextTick();
    expect(wrapper.findComponent(DeskView).exists()).toBe(false);
    expect(wrapper.findComponent(Newspaper).exists()).toBe(true);
    expect(wrapper.findComponent(PaperPage).exists()).toBe(false);
  });

  it("routes 'eventPage' to FrontPage", async () => {
    const { desk, wrapper } = mountAtHub();
    desk.phase = 'eventPage';
    await nextTick();
    expect(wrapper.findComponent(DeskView).exists()).toBe(false);
    expect(wrapper.findComponent(FrontPage).exists()).toBe(true);
    expect(wrapper.findComponent(PaperPage).exists()).toBe(false);
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

  it('renders DeskView inside the responsive player viewport', () => {
    const { desk, wrapper } = mountAtHub();
    expect(desk.phase).toBe('idle');
    expect(wrapper.find('.responsive-viewport').exists()).toBe(true);
    const viewport = wrapper.findComponent(ResponsiveViewport);
    expect(viewport.exists()).toBe(true);
    expect(viewport.findComponent(DeskView).exists()).toBe(true);
  });

  it('renders FrontPage inside the responsive player viewport', async () => {
    const { desk, wrapper } = mountAtHub();
    desk.phase = 'eventPage';
    await nextTick();
    expect(wrapper.find('.responsive-viewport').exists()).toBe(true);
    const viewport = wrapper.findComponent(ResponsiveViewport);
    expect(viewport.exists()).toBe(true);
    expect(viewport.findComponent(FrontPage).exists()).toBe(true);
  });
});

// Toast lives at the GameView phase-router level (phase 2.5 Task 8), not
// nested inside DeskView — an achievement can unlock on ANY frame,
// including the ending/page surfaces PaperPage renders (game_over.scene.dry
// is reached as ordinary content, not a desk phase), so the toast must stay
// visible across the whole router. See desk.dossier.test.ts's DeskView
// describe block for the negative half of this same move.
describe('Toast (mounted at the GameView router, both surfaces)', () => {
  it('is visible while DeskView is showing', async () => {
    const { desk, wrapper } = mountAtHub();
    expect(desk.phase).toBe('idle');
    desk.toastKey = 'desk.toast.handFull';
    await nextTick();
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(true);
  });

  it('is visible while PaperPage is showing (e.g. an ending page)', async () => {
    const { desk, wrapper } = mountAtHub();
    desk.phase = 'page';
    await nextTick();
    expect(wrapper.findComponent(DeskView).exists()).toBe(false);
    expect(wrapper.findComponent(PaperPage).exists()).toBe(true);

    desk.toastKey = 'desk.toast.engineError';
    await nextTick();
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(true);
    expect(wrapper.text()).toContain(i18n.global.t('desk.toast.engineError'));
  });

  it('shows an achievement toast (dynamic content, not an i18n key) over PaperPage too', async () => {
    const { desk, wrapper } = mountAtHub();
    desk.phase = 'page';
    await nextTick();
    desk.achievementToast = { name: 'Calçotada Popular', image: 'img/x.png', stars: 2 };
    await nextTick();
    expect(wrapper.find('[data-test="toast-achievement"]').exists()).toBe(true);
    expect(wrapper.text()).toContain('Calçotada Popular');
  });

  it('renders nothing during boot', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    const wrapper = mount(GameView, withPlugins());
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="toast-achievement"]').exists()).toBe(false);
    vi.unstubAllGlobals();
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
