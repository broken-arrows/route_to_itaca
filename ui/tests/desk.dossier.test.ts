import { describe, it, expect, beforeEach, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { nextTick } from 'vue';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import PaperOption from '../src/components/desk/PaperOption.vue';
import OpenDossier from '../src/components/desk/OpenDossier.vue';
import FlyingCard from '../src/components/desk/FlyingCard.vue';
import Toast from '../src/components/desk/Toast.vue';
import DeskView from '../src/views/DeskView.vue';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import { DELAYS } from '../src/components/desk/motion';
import type { ChoiceView, CardView } from '../src/engine/types';

// Repo convention (per task brief): register BOTH plugins for new test
// files, like tests/debug-page.test.ts does — unlike desk.components.test.ts's
// `withI18n` helper, which relies on setActivePinia alone.
let pinia: ReturnType<typeof createPinia>;
beforeEach(() => {
  pinia = createPinia();
  setActivePinia(pinia);
  setAnimationsForTest(false);
});

function withPlugins(extra: Record<string, unknown> = {}) {
  return { global: { plugins: [pinia, i18n] }, ...extra };
}

describe('PaperOption', () => {
  function choiceWith(overrides: Partial<ChoiceView> = {}): ChoiceView {
    return {
      id: 'opt1',
      title: 'Resolve quietly',
      subtitle: '[1 resources]',
      canChoose: true,
      tags: [],
      ...overrides,
    };
  }

  it('renders title and subtitle as-is', () => {
    const wrapper = mount(PaperOption, withPlugins({ props: { choice: choiceWith(), index: 0, shaking: false } }));
    expect(wrapper.text()).toContain('Resolve quietly');
    expect(wrapper.text()).toContain('[1 resources]');
  });

  // Fix wave 3, Job 2: choice titles/subtitles are ENGINE OUTPUT — convertLine
  // (_contentToHTML) emits <em>/<strong> and passes `magic` blocks through raw.
  // 10 option titles in the compiled game carry markup (e.g. @root.start's
  // `<span style="font-size: 1.1em;">Start game</span>`, on the very first
  // screen). Interpolated, Vue escaped it and the player saw literal tags.
  // Same trust boundary as the prose, which already renders with v-html.
  it('renders markup in the title/subtitle as elements, not as escaped text', () => {
    const wrapper = mount(
      PaperOption,
      withPlugins({
        props: {
          choice: choiceWith({
            title: '<span style="font-size: 1.1em;">Start game</span>',
            subtitle: 'costs <em>everything</em>',
          }),
          index: 0,
          shaking: false,
        },
      }),
    );
    const title = wrapper.find('.option-title');
    expect(title.find('span').exists()).toBe(true);
    expect(title.text()).toBe('Start game'); // the tags are structure, not content
    expect(wrapper.find('.option-subtitle em').exists()).toBe(true);
    expect(wrapper.text()).not.toContain('<span'); // nothing escaped through
  });

  it('emits pick with its index when canChoose', async () => {
    const wrapper = mount(PaperOption, withPlugins({ props: { choice: choiceWith(), index: 2, shaking: false } }));
    await wrapper.trigger('click');
    expect(wrapper.emitted('pick')).toEqual([[2]]);
  });

  it('adds the locked class and does not emit pick when not canChoose', async () => {
    const wrapper = mount(
      PaperOption,
      withPlugins({ props: { choice: choiceWith({ canChoose: false }), index: 0, shaking: false } }),
    );
    expect(wrapper.classes()).toContain('locked');
    await wrapper.trigger('click');
    expect(wrapper.emitted('pick')).toBeUndefined();
  });

  it('renders the locked chip text when not canChoose', () => {
    const wrapper = mount(
      PaperOption,
      withPlugins({ props: { choice: choiceWith({ canChoose: false }), index: 0, shaking: false } }),
    );
    expect(wrapper.text()).toContain(i18n.global.t('desk.dossier.locked'));
  });

  it('adds the shaking class when the shaking prop is set', () => {
    const wrapper = mount(PaperOption, withPlugins({ props: { choice: choiceWith(), index: 0, shaking: true } }));
    expect(wrapper.classes()).toContain('shaking');
  });

  it('does not add the shaking class when the prop is false', () => {
    const wrapper = mount(PaperOption, withPlugins({ props: { choice: choiceWith(), index: 0, shaking: false } }));
    expect(wrapper.classes()).not.toContain('shaking');
  });
});

describe('Toast', () => {
  it('renders nothing when textKey is null', () => {
    const wrapper = mount(Toast, withPlugins({ props: { textKey: null } }));
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(false);
  });

  it('renders the translated text for the given key', () => {
    const wrapper = mount(Toast, withPlugins({ props: { textKey: 'desk.toast.deckEmpty' } }));
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(true);
    expect(wrapper.text()).toContain(i18n.global.t('desk.toast.deckEmpty'));
  });
});

describe('FlyingCard', () => {
  it('renders the drawn card title', () => {
    const card: CardView = { id: 'c1', title: 'Card One', tags: [], role: 'card-gov' };
    const wrapper = mount(FlyingCard, withPlugins({ props: { card } }));
    expect(wrapper.find('[data-test="flying-card"]').exists()).toBe(true);
    expect(wrapper.text()).toContain('Card One');
  });
});

// Shared game fixture for OpenDossier + DeskView: root -> hub (isHand,
// role: desk) -> a pinned card_a dossier with two options, one of which
// (resolve_costly) is locked via chooseIf (visible, canChoose:false —
// distinct from viewIf, which would drop it from the list entirely).
const dossierGame = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      newPage: true,
      onArrival: [{ $code: 'Q.gold = 1;' }],
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
      options: [{ id: '@card_a' }],
    },
    card_a: {
      id: 'card_a',
      type: 'scene',
      title: 'Card A',
      newPage: true,
      isPinnedCard: true,
      role: 'card-gov',
      content: [{ type: 'paragraph', content: ['Decide wisely.'] }],
      options: [{ id: '@resolve_cheap' }, { id: '@resolve_costly' }],
    },
    resolve_cheap: {
      id: 'resolve_cheap',
      type: 'scene',
      title: 'Resolve quietly',
      subtitle: '[1 resources]',
      content: [],
      goTo: [{ id: 'hub' }],
    },
    resolve_costly: {
      id: 'resolve_costly',
      type: 'scene',
      title: 'Resolve loudly',
      subtitle: '[3 resources]',
      chooseIf: { $code: 'return false;' },
      content: [],
      goTo: [{ id: 'hub' }],
    },
  },
  qualities: {},
  qdisplays: {},
  tagLookup: {},
};

function mountDossierScene() {
  const game = useGameStore();
  const desk = useDeskStore();
  game.initFromText(JSON.stringify(dossierGame));
  game.newGame();
  game.choose(0); // root -> hub
  desk.playPinned(game.frame!.pinned[0]); // hub -> card_a, opens the dossier
  const wrapper = mount(OpenDossier, withPlugins());
  return { game, desk, wrapper };
}

describe('OpenDossier', () => {
  it('renders one PaperOption per choice', () => {
    const { wrapper } = mountDossierScene();
    expect(wrapper.find('[data-test="paper-option-0"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="paper-option-1"]').exists()).toBe(true);
    expect(wrapper.text()).toContain('Resolve quietly');
    expect(wrapper.text()).toContain('Resolve loudly');
  });

  it('renders the cover prose via v-html of frame.html', () => {
    const { wrapper } = mountDossierScene();
    expect(wrapper.html()).toContain('Decide wisely.');
  });

  // Fix wave 3, Job 1: the dismiss affordance is GONE (user decision,
  // overriding phase-2 spec §6). It restored a whole-engine snapshot for free,
  // on any card, at any difficulty — bypassing the difficulty gate on
  // `easy_discard`, the engine's own (costed) route back to the hand. A
  // dossier is now resolved by its papers and by nothing else: the papers are
  // the only interactive elements in it (no <button> anywhere).
  it('offers no way out of the dossier except its papers', () => {
    const { wrapper } = mountDossierScene();
    expect(wrapper.findAll('button')).toHaveLength(0);
    expect(wrapper.text()).not.toContain('✕');
    // ...and the papers themselves are all still there.
    expect(wrapper.findAll('.paper-option')).toHaveLength(2);
  });

  it('clicking a choosable paper resolves the dossier (animations off = instant)', async () => {
    const { desk, wrapper } = mountDossierScene();
    await wrapper.find('[data-test="paper-option-0"]').trigger('click');
    expect(desk.phase).toBe('idle');
    expect(desk.openCard).toBeNull();
  });

  it('clicking a locked paper reaches the store but does not resolve', async () => {
    const { desk, wrapper } = mountDossierScene();
    const spy = vi.spyOn(desk, 'pickPaper');
    await wrapper.find('[data-test="paper-option-1"]').trigger('click');
    expect(spy).toHaveBeenCalledWith(1);
    expect(desk.phase).toBe('dossierOpen');
    expect(desk.openCard).not.toBeNull();
  });

  // REGRESSION (review fix round, Critical): with animations ON the pick's
  // engine call advances game.frame to the destination hub (an isHand scene
  // with no choices) BEFORE the 620ms 'resolving' window starts — a dossier
  // rendering live from the frame would go blank/stale for the whole
  // fly-out. The store must snapshot the pre-pick view (resolveView) and
  // the dossier must render from it during 'resolving'.
  it('keeps showing the pre-pick papers and prose during the resolve fly-out (animations on)', async () => {
    vi.useFakeTimers();
    try {
      setAnimationsForTest(true);
      const { desk, wrapper } = mountDossierScene();
      await wrapper.find('[data-test="paper-option-0"]').trigger('click');

      expect(desk.phase).toBe('resolving');
      // The engine has already moved on (hub has no choices), but the
      // dossier must still show what was picked from.
      expect(wrapper.text()).toContain('Resolve quietly');
      expect(wrapper.text()).toContain('Resolve loudly');
      expect(wrapper.html()).toContain('Decide wisely.');

      vi.advanceTimersByTime(DELAYS.resolve);
      await nextTick();
      expect(desk.phase).toBe('idle');
      expect(desk.resolveView).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });
});

// Review fix round, Important: PaperOption's Enter/Space handler used to
// call the emit path directly — dead, because OpenDossier listens on a
// wrapper element's native click (so locked clicks reach the store), not
// on the `pick` emit. Keyboard must synthesize a native click on the root
// so it takes exactly the mouse path.
describe('PaperOption keyboard activation (through OpenDossier)', () => {
  // Vue's runtime-dom event invoker drops events whose timeStamp is not
  // newer than the listener's attach time (millisecond resolution, Date.now
  // based). This bites NATIVELY DISPATCHED events only — here, the click
  // that PaperOption's keydown handler synthesizes via rootEl.click().
  // Events fired through @vue/test-utils' trigger() are immune: test-utils
  // pre-stamps event._vts = Date.now() + 1 precisely to defeat this guard
  // (their issue #1854 workaround) — so ordinary trigger('click') tests do
  // NOT need this recipe. For the synthesized click, put the clock on fake
  // timers and advance it a couple of ms between mount and dispatch so the
  // click is strictly newer than the listener attach. Test-env artifact
  // only: in a real browser, mount precedes any keypress by far more than
  // a millisecond.
  it('Enter on a choosable paper takes the mouse path and resolves (animations off = instant)', async () => {
    vi.useFakeTimers();
    try {
      const { desk, wrapper } = mountDossierScene();
      vi.advanceTimersByTime(2);
      await wrapper.find('[data-test="paper-option-0"]').trigger('keydown.enter');
      expect(desk.phase).toBe('idle');
      expect(desk.openCard).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('Enter on a locked paper reaches the store and sets shakeIdx', async () => {
    vi.useFakeTimers();
    try {
      setAnimationsForTest(true);
      const { desk, wrapper } = mountDossierScene();
      vi.advanceTimersByTime(2);
      await wrapper.find('[data-test="paper-option-1"]').trigger('keydown.enter');
      expect(desk.phase).toBe('dossierOpen');
      expect(desk.shakeIdx).toBe(1);
      vi.advanceTimersByTime(DELAYS.cancel);
      expect(desk.shakeIdx).toBe(-1);
    } finally {
      vi.useRealTimers();
    }
  });

  // No fake-timer recipe needed here: trigger('click') events carry
  // test-utils' _vts pre-stamp and are immune to the invoker guard above.
  it('one mouse click on a choosable paper dispatches pickPaper exactly once (no double path)', async () => {
    const { desk, wrapper } = mountDossierScene();
    const spy = vi.spyOn(desk, 'pickPaper');
    await wrapper.find('[data-test="paper-option-0"]').trigger('click');
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith(0);
    expect(desk.phase).toBe('idle'); // and the single dispatch really resolved
  });
});

describe('DeskView phase wiring', () => {
  const deskGame = {
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
        options: [{ id: '@cat_gov' }],
      },
      cat_gov: {
        id: 'cat_gov',
        type: 'scene',
        title: 'Generalitat',
        isDeck: true,
        role: 'deck',
        content: [],
        options: [],
      },
    },
    qualities: {},
    qdisplays: {},
    tagLookup: {},
  };

  function mountDesk() {
    const game = useGameStore();
    const desk = useDeskStore();
    game.initFromText(JSON.stringify(deskGame));
    game.newGame();
    game.choose(0); // root -> hub
    const wrapper = mount(DeskView, withPlugins());
    return { game, desk, wrapper };
  }

  it('shows FlyingCard only during the drawing phase', async () => {
    const { desk, wrapper } = mountDesk();
    expect(wrapper.find('[data-test="flying-card"]').exists()).toBe(false);

    desk.phase = 'drawing';
    desk.flying = { id: 'c1', title: 'Card One', tags: [] };
    await nextTick();
    expect(wrapper.find('[data-test="flying-card"]').exists()).toBe(true);

    desk.phase = 'idle';
    await nextTick();
    expect(wrapper.find('[data-test="flying-card"]').exists()).toBe(false);
  });

  it('shows OpenDossier during dossierOpen and resolving, not idle', async () => {
    const { desk, wrapper } = mountDesk();
    expect(wrapper.find('[data-test="open-dossier"]').exists()).toBe(false);

    desk.phase = 'dossierOpen';
    desk.openCard = { id: 'c1', title: 'Card One', tags: [] };
    await nextTick();
    expect(wrapper.find('[data-test="open-dossier"]').exists()).toBe(true);

    desk.phase = 'resolving';
    await nextTick();
    expect(wrapper.find('[data-test="open-dossier"]').exists()).toBe(true);

    desk.phase = 'idle';
    await nextTick();
    expect(wrapper.find('[data-test="open-dossier"]').exists()).toBe(false);
  });

  it('renders the toast when toastKey is set, nothing when null', async () => {
    const { desk, wrapper } = mountDesk();
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(false);

    desk.toastKey = 'desk.toast.handFull';
    await nextTick();
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(true);
  });
});
