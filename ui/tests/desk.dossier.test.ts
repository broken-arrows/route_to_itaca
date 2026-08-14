import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { nextTick } from 'vue';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import PaperOption from '../src/components/desk/PaperOption.vue';
import OpenDossier from '../src/components/desk/OpenDossier.vue';
import OutTray from '../src/components/desk/OutTray.vue';
import FlyingCard from '../src/components/desk/FlyingCard.vue';
import Toast from '../src/components/desk/Toast.vue';
import DeskView from '../src/views/DeskView.vue';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import { DELAYS } from '../src/components/desk/motion';
import { markGlossary, type GlossaryTerm } from '../src/glossary/mark';
import type { ChoiceView, CardView } from '../src/engine/types';
import uiEn from '../../source/locales/en/ui.json';

// Every DeskView mount below now also mounts Clipboard (phase 3b Task 9;
// formerly the inert ClipboardFrame), which reads `brief.context.*` — GAME
// chrome sourced from source/locales/<loc>/ui.json (see i18n.ts's
// initGameLocale, and the same fix in desk.components.test.ts).
i18n.global.mergeLocaleMessage('en', uiEn as never);

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
    expect(wrapper.find('[data-test="toast-achievement"]').exists()).toBe(false);
  });

  it('renders the translated text for the given key', () => {
    const wrapper = mount(Toast, withPlugins({ props: { textKey: 'desk.toast.deckEmpty' } }));
    expect(wrapper.find('[data-test="toast"]').exists()).toBe(true);
    expect(wrapper.text()).toContain(i18n.global.t('desk.toast.deckEmpty'));
  });

  // Achievement toasts (phase 2.5 Task 8) carry DYNAMIC game content
  // (name/image/stars from game.data.achievements) — not an i18n key, per
  // the task brief's ambiguity resolution 1. This is a second, independent
  // channel from textKey.
  describe('achievement payload', () => {
    const achievement = { name: 'Calçotada Popular', image: 'img/achievements/calcotada.png', stars: 3 };

    it('renders the achievement name, image and star count instead of textKey', () => {
      const wrapper = mount(
        Toast,
        withPlugins({ props: { textKey: 'desk.toast.deckEmpty', achievement } }),
      );
      expect(wrapper.find('[data-test="toast-achievement"]').exists()).toBe(true);
      expect(wrapper.find('[data-test="toast"]').exists()).toBe(false); // achievement wins
      expect(wrapper.text()).toContain('Calçotada Popular');
      expect(wrapper.text()).toContain(i18n.global.t('desk.toast.achievementUnlocked'));
      // Registry paths are web-root-relative; the component resolves them
      // against BASE_URL ('/' under vitest), same as HandCard/GlossaryTerm.
      expect(wrapper.get('.toast-achievement-image').attributes('src')).toBe(
        `${import.meta.env.BASE_URL}img/achievements/calcotada.png`,
      );
      expect(wrapper.findAll('.star--filled')).toHaveLength(3);
      expect(wrapper.findAll('.star--empty')).toHaveLength(2);
    });

    it('renders nothing when neither textKey nor achievement is set', () => {
      const wrapper = mount(Toast, withPlugins({ props: { textKey: null, achievement: null } }));
      expect(wrapper.find('[data-test="toast"]').exists()).toBe(false);
      expect(wrapper.find('[data-test="toast-achievement"]').exists()).toBe(false);
    });
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

// Task 7 (Wave 2): a variant of mountDossierScene whose card_a.content is
// swappable — 110 real scene files lead with a `=` heading (e.g.
// erc_campaigning -> visible double "Campaigning", confirmed live via the
// erc_enemies card: "Choosing Our Enemies" cover-title h2 stacked directly
// above the content's own "Choosing our enemies" h1). card_a.title stays
// 'Card A' throughout, distinct from any heading text put in the content,
// so a test can tell the two title sources apart.
function mountDossierSceneWithCardContent(content: unknown[]) {
  const gameJson = {
    ...dossierGame,
    scenes: { ...dossierGame.scenes, card_a: { ...dossierGame.scenes.card_a, content } },
  };
  const game = useGameStore();
  const desk = useDeskStore();
  game.initFromText(JSON.stringify(gameJson));
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

  // Task 7 (Wave 2): 110 scene files lead their content with a `=` heading
  // — dendry compiles that to a leading <h1> — and OpenDossier used to ALSO
  // render card.title as a separate <h2 class="cover-title"> above it,
  // stacking two titles (confirmed live: opening the real erc_enemies card
  // showed "Choosing Our Enemies" then "Choosing our enemies" right below
  // it). The content h1 IS the title now: when the prose leads with one,
  // the separate cover-title element must not render at all.
  it('renders exactly one title when the prose leads with an <h1> — the content heading IS the title', () => {
    const { wrapper } = mountDossierSceneWithCardContent([
      { type: 'heading', content: ['Card A Real Title'] },
      { type: 'paragraph', content: ['Decide wisely.'] },
    ]);
    expect(wrapper.find('.cover-title').exists()).toBe(false);
    const proseHeading = wrapper.find('.cover-prose h1');
    expect(proseHeading.exists()).toBe(true);
    expect(proseHeading.text()).toBe('Card A Real Title');
    // No duplicate anywhere: exactly one heading-level element in the cover.
    expect(wrapper.findAll('h1, h2')).toHaveLength(1);
    expect(wrapper.text()).toContain('Decide wisely.');
  });

  // The other half of the same fix: a scene whose content has no leading
  // heading (card_a's default fixture, plain paragraph) must keep rendering
  // the card.title fallback exactly as before — this is NOT a regression,
  // just confirming the gate goes the other way too.
  it('renders the card.title fallback when the prose has no leading <h1>', () => {
    const { wrapper } = mountDossierScene();
    const title = wrapper.find('.cover-title');
    expect(title.exists()).toBe(true);
    expect(title.text()).toBe('Card A');
    expect(wrapper.find('.cover-prose h1').exists()).toBe(false);
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

// REGRESSION found by this task's own work, not pre-existing: turning on
// window.displayText (Task 5, main.ts) means CardView.title — which reaches
// the Desk through CaptureUI.normalizeCard's convertLine, exactly like the
// prose — can arrive already wrapped in a glossary `<span data-term=...>`.
// 4 of the 6 real pinned advisor cards (source/scenes) have no
// zero-width-space escape on their own name,
// so opening one used to show the literal `<span class="term"...>` text in
// the dossier's cover title (plain `{{ }}` interpolation). Fixed by routing
// it through <Prose tag="span"> like the prose already does. This test
// installs the REAL window.displayText hook (main.ts's own wiring, not a
// stand-in) so it exercises the actual engine -> convertLine -> marked-title
// path, not just Prose's own known-safe v-html rendering.
describe('OpenDossier — a card title the engine itself marked as a glossary term', () => {
  const TERMS: GlossaryTerm[] = [
    { id: 'ernest_maragall', match: ['Ernest Maragall'], display: 'Ernest Maragall', colour: 'psc' },
  ];
  const markedDossierGame = {
    ...dossierGame,
    scenes: { ...dossierGame.scenes, card_a: { ...dossierGame.scenes.card_a, title: 'Ernest Maragall' } },
    data: { glossary: { terms: TERMS } },
  };

  beforeEach(() => {
    window.displayText = (text: string) => markGlossary(text, TERMS);
  });
  afterEach(() => {
    delete (window as { displayText?: unknown }).displayText;
  });

  it('renders the marked title as a coloured element, never as literal tag text', () => {
    const game = useGameStore();
    const desk = useDeskStore();
    game.initFromText(JSON.stringify(markedDossierGame));
    game.newGame();
    game.choose(0); // root -> hub
    desk.playPinned(game.frame!.pinned[0]); // hub -> card_a ("Ernest Maragall"), opens the dossier
    const wrapper = mount(OpenDossier, withPlugins());

    const cover = wrapper.get('.cover-title');
    expect(cover.text()).toBe('Ernest Maragall'); // not the raw markup
    const term = cover.get('[data-term="ernest_maragall"]');
    expect(term.attributes('style')).toContain('var(--psc)');
    expect(wrapper.html()).not.toContain('&lt;span'); // nothing escaped through
  });
});

// Same regression, same fix, the OTHER call site: stores/desk.ts sets
// outTray.title from openCard.title verbatim (see the comment above),
// so the OUT tray's slip needs the identical safety. Tested directly on the
// component (its own contract is a plain `{title}` prop, no engine needed).
describe('OutTray — a marked title', () => {
  it('renders as an element rather than showing the literal tag text', () => {
    const wrapper = mount(
      OutTray,
      withPlugins({
        props: { entry: { title: '<span class="term" data-term="x">Ernest Maragall</span>' } },
      }),
    );
    expect(wrapper.get('[data-test="out-entry"]').text()).toContain('Ernest Maragall');
    expect(wrapper.find('[data-term="x"]').exists()).toBe(true);
    expect(wrapper.html()).not.toContain('&lt;span');
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

  // Toast moved OUT of DeskView in phase 2.5 Task 8 — up to GameView's own
  // template, alongside the DeskView/PaperPage phase router, rather than
  // nested inside just one of the two surfaces it routes between. Reason:
  // an achievement can unlock on ANY frame (e.g. game_over.scene.dry, which
  // renders as PaperPage's 'ending' variant, not DeskView), so the toast
  // must stay visible across that whole router, not just its desk branch.
  // See game-view.test.ts's "Toast" describe block for the real coverage;
  // this is the negative half of that move, kept here so the two together
  // read as one deliberate relocation rather than a silently dropped test.
  it('no longer renders its own Toast — that lives in GameView now', () => {
    const { wrapper } = mountDesk();
    expect(wrapper.findComponent(Toast).exists()).toBe(false);
  });
});
