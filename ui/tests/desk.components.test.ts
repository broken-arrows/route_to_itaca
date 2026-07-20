import { describe, it, expect, beforeEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { nextTick } from 'vue';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import { skinFor } from '../src/components/desk/skins';
import InTray from '../src/components/desk/InTray.vue';
import HandCard from '../src/components/desk/HandCard.vue';
import OutTray from '../src/components/desk/OutTray.vue';
import ActionsTray from '../src/components/desk/ActionsTray.vue';
import DeskMonth from '../src/components/desk/DeskMonth.vue';
import DeskView from '../src/views/DeskView.vue';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import type { DeckView, CardView } from '../src/engine/types';
import uiEn from '../../source/locales/en/ui.json';

// Every DeskView mount below now also mounts ClipboardFrame, which reads
// `brief.tab.*` — GAME chrome sourced from source/locales/<loc>/ui.json, not
// ui/'s own bundled defaults (see i18n.ts's initGameLocale). Production
// merges this at boot via a fetch; these tests never mount App.vue's
// onMounted, so merge the real catalog directly (same source `brief.clipboard
// .test.ts` uses) rather than let every DeskView mount print
// `[intlify] Not found 'brief.tab.*'` noise.
i18n.global.mergeLocaleMessage('en', uiEn as never);

function withI18n(extra: Record<string, unknown> = {}) {
  return { global: { plugins: [i18n] }, ...extra };
}

describe('skinFor', () => {
  it.each([
    ['card-gov', 'gov', '#f4f1e6', '#c9bfa4'],
    ['card-party', 'party', '#e3d3a8', '#c2ad72'],
    ['card-parliament', 'parliament', '#f6f4ec', '#4a5b6a'],
  ] as const)('role %s resolves to the %s skin', (role, key, bg, bd) => {
    const skin = skinFor(role);
    expect(skin.key).toBe(key);
    expect(skin.bg).toBe(bg);
    expect(skin.bd).toBe(bd);
  });

  it.each([undefined, '', 'card', 'deck', 'newspaper', 'card-ministry'])(
    'unrecognized/undefined role %s falls back to neutral',
    (role) => {
      const skin = skinFor(role);
      expect(skin.key).toBe('neutral');
      expect(skin.bg).toBe('#fdfcf8');
      expect(skin.bd).toBe('#e0d9c8');
    },
  );

  it('never throws, even for unexpected input', () => {
    expect(() => skinFor(undefined)).not.toThrow();
    expect(() => skinFor('literally anything')).not.toThrow();
  });

  it.each([
    ['deck-gov', 'gov'],
    ['deck-party', 'party'],
    ['deck-parliament', 'parliament'],
    ['deck', 'neutral'],
  ] as const)('skinFor(%s) -> %s skin', (role, key) => {
    expect(skinFor(role).key).toBe(key);
  });
});

describe('InTray', () => {
  const deck: DeckView = {
    id: 'main.cat_gov',
    title: 'Generalitat',
    canChoose: true,
    tags: [],
    role: 'card-gov',
  };

  it('renders the deck title', () => {
    const wrapper = mount(InTray, withI18n({ props: { deck } }));
    expect(wrapper.text()).toContain('Generalitat');
  });

  it('emits draw with the deck id on click', async () => {
    const wrapper = mount(InTray, withI18n({ props: { deck } }));
    await wrapper.trigger('click');
    expect(wrapper.emitted('draw')).toEqual([['main.cat_gov']]);
  });

  it('does not emit draw when disabled', async () => {
    const wrapper = mount(InTray, withI18n({ props: { deck, disabled: true } }));
    await wrapper.trigger('click');
    expect(wrapper.emitted('draw')).toBeUndefined();
  });

  // CONFIDENCIAL is diegetic stationery text (in-world Catalan document
  // furniture), rendered as a literal constant, NOT an i18n key — a
  // Generalitat folder says CONFIDENCIAL whatever the UI language is.
  it('gov tray folder carries the diegetic CONFIDENCIAL stamp', () => {
    const wrapper = mount(InTray, withI18n({ props: { deck } })); // deck.role = card-gov
    expect(wrapper.text()).toContain('CONFIDENCIAL');
  });

  it('non-gov trays do not carry the CONFIDENCIAL stamp', () => {
    const partyDeck: DeckView = { ...deck, id: 'main.party_erc', role: 'card-party' };
    const wrapper = mount(InTray, withI18n({ props: { deck: partyDeck } }));
    expect(wrapper.text()).not.toContain('CONFIDENCIAL');
  });
});

describe('HandCard', () => {
  function cardWith(role: CardView['role']): CardView {
    return { id: 'c1', title: 'Card One', tags: [], role };
  }

  it.each([
    ['card-gov', 'skin-gov'],
    ['card-party', 'skin-party'],
    ['card-parliament', 'skin-parliament'],
    [undefined, 'skin-neutral'],
  ] as const)('applies the right skin class for role %s', (role, expectedClass) => {
    const wrapper = mount(HandCard, withI18n({ props: { card: cardWith(role), index: 0 } }));
    expect(wrapper.classes()).toContain(expectedClass);
  });

  it('emits play with the card on click', async () => {
    const card = cardWith('card-gov');
    const wrapper = mount(HandCard, withI18n({ props: { card, index: 0 } }));
    await wrapper.trigger('click');
    expect(wrapper.emitted('play')).toEqual([[card]]);
  });

  it('renders the card title', () => {
    const wrapper = mount(HandCard, withI18n({ props: { card: cardWith('card-party'), index: 1 } }));
    expect(wrapper.text()).toContain('Card One');
  });

  // ASSUMPTE: is diegetic stationery text (in-world Catalan document
  // furniture, per the NOTES card anatomy), rendered as a literal constant,
  // NOT an i18n key — it stays Catalan whatever the UI language is.
  it('title block carries the diegetic ASSUMPTE: prefix on every skin', () => {
    for (const role of ['card-gov', 'card-party', undefined] as const) {
      const wrapper = mount(HandCard, withI18n({ props: { card: cardWith(role), index: 0 } }));
      expect(wrapper.text()).toContain('ASSUMPTE:');
      expect(wrapper.text()).toContain('Card One');
    }
  });

  it('gov skin renders the ministry red double rule; party does not', () => {
    const gov = mount(HandCard, withI18n({ props: { card: cardWith('card-gov'), index: 0 } }));
    expect(gov.find('.gov-rule').exists()).toBe(true);
    const party = mount(HandCard, withI18n({ props: { card: cardWith('card-party'), index: 0 } }));
    expect(party.find('.gov-rule').exists()).toBe(false);
  });
});

// The binding project rule is: red = world/Parlament surfaces only. The ministry
// double rule is the ONE ratified exception (it is paper anatomy, not a signal —
// phase-2 final review, I4), and it is a NAMED TOKEN so that the rule stays
// enforceable. It previously sat inline as `rgba(176, 48, 48, .55)` — #b03030, a
// DIFFERENT red from --accent-red — which meant it evaded every grep for the
// reservation and the constraint could not be audited at all. Assert the token,
// not the pixels: a mounted SFC's scoped <style> is not applied in jsdom, so the
// source is the only place this is checkable.
describe('red-reservation: the ministry rule is a declared token, not a literal', () => {
  const read = (p: string) => readFileSync(path.join(__dirname, '..', 'src', p), 'utf8');
  const RULE_SITES = ['components/desk/HandCard.vue', 'components/desk/OpenDossier.vue'];
  // Any red that is not one of the declared tokens. Catches #b03030 and the
  // rgb()/rgba() form it was written in.
  const RAW_RED = /#b0?3030|rgba?\(\s*17[0-9]\s*,\s*4[0-9]\s*,\s*4[0-9]/i;

  it('tokens.css declares --paper-rule-ink as the documented exception', () => {
    const tokens = read('styles/tokens.css');
    expect(tokens).toContain('--paper-rule-ink');
    // The exception must stay *documented* where it is declared, or the next
    // reader has no way to tell it apart from a violation.
    expect(tokens).toMatch(/EXCEPTION[\s\S]*--paper-rule-ink/);
  });

  it.each(RULE_SITES)('%s uses the token and carries no raw red literal', (site) => {
    const src = read(site);
    expect(src).toContain('var(--paper-rule-ink)');
    expect(src, `${site} has an off-token red literal — see tokens.css`).not.toMatch(RAW_RED);
  });
});

describe('ActionsTray', () => {
  const pinned: CardView[] = [
    { id: 'p1', title: 'Advisor One', tags: [], role: 'pinned-action' },
    { id: 'p2', title: 'Advisor Two', tags: [], role: 'pinned-action' },
    { id: 'p3', title: 'Advisor Three', tags: [], role: 'pinned-action' },
  ];

  // Every advisor name now renders through <Prose> (same glossary-safety fix
  // as OutTray's slip title and OpenDossier's cover title — an advisor card's
  // title can arrive already glossary-marked, and 4 of the 6 real ones name a
  // party/person term; see LEARNINGS 2026-07-17). Prose calls useGlossary()
  // internally, so an active pinia is required for every mount here.
  let pinia: ReturnType<typeof createPinia>;
  beforeEach(() => {
    pinia = createPinia();
    setActivePinia(pinia);
  });

  it('renders one entry per pinned card', () => {
    const wrapper = mount(ActionsTray, { global: { plugins: [pinia, i18n] }, props: { pinned } });
    expect(wrapper.findAll('[data-test="pinned-card"]')).toHaveLength(3);
  });

  it('emits play with the clicked card', async () => {
    const wrapper = mount(ActionsTray, { global: { plugins: [pinia, i18n] }, props: { pinned } });
    await wrapper.findAll('[data-test="pinned-card"]')[1].trigger('click');
    expect(wrapper.emitted('play')).toEqual([[pinned[1]]]);
  });

  it('does not emit when disabled', async () => {
    const wrapper = mount(ActionsTray, {
      global: { plugins: [pinia, i18n] },
      props: { pinned, disabled: true },
    });
    await wrapper.findAll('[data-test="pinned-card"]')[0].trigger('click');
    expect(wrapper.emitted('play')).toBeUndefined();
  });
});

describe('DeskMonth', () => {
  it('renders the localized month name and the year', () => {
    const wrapper = mount(DeskMonth, withI18n({ props: { month: 3, year: 2014 } }));
    expect(wrapper.text()).toContain(i18n.global.t('desk.month.3'));
    expect(wrapper.text()).toContain('2014');
  });

  it.each([1, 12])('renders month %i via the desk.month.<m> key', (m) => {
    const wrapper = mount(DeskMonth, withI18n({ props: { month: m, year: 2012 } }));
    expect(wrapper.text()).toContain(i18n.global.t(`desk.month.${m}`));
  });

  it('does not throw when month/year are null', () => {
    expect(() => mount(DeskMonth, withI18n({ props: { month: null, year: null } }))).not.toThrow();
  });

  // Task 7 (Wave 2): the desk month title now comes from CONTENT — the
  // extracted leading <h1> from post_event.scene.dry's own heading
  // (`= [+ month : month +] [+ year +][? if rubicon:, Week [+ week +]?]`),
  // not a UI-hardcoded Q.month/Q.year read. DeskMonth renders it via
  // <Prose tag="span"> (glossary/insert-safe, same as every other engine
  // title reaching this app) and ignores month/year entirely while present
  // — the two must never stack (that would be the double-title bug this
  // whole task exists to kill, one level down).
  it('renders titleHtml via Prose when present, ignoring month/year entirely', () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const wrapper = mount(DeskMonth, {
      global: { plugins: [pinia, i18n] },
      props: { month: 3, year: 2014, titleHtml: 'Novembre 2012, Week 2' },
    });
    expect(wrapper.text()).toContain('Novembre 2012, Week 2');
    expect(wrapper.text()).not.toContain('2014');
    expect(wrapper.text()).not.toContain(i18n.global.t('desk.month.3'));
  });

  // The boot case (spec: "no post_event has run, no h1 exists yet") — the
  // Q-based rendering this component always had stays the fallback.
  it('falls back to the Q-based month/year rendering when titleHtml is absent', () => {
    const wrapper = mount(DeskMonth, withI18n({ props: { month: 3, year: 2014, titleHtml: null } }));
    expect(wrapper.text()).toContain(i18n.global.t('desk.month.3'));
    expect(wrapper.text()).toContain('2014');
  });
});

describe('OutTray', () => {
  it('shows the empty state when entry is null', () => {
    const wrapper = mount(OutTray, withI18n({ props: { entry: null } }));
    expect(wrapper.text()).toContain(i18n.global.t('desk.out.empty'));
  });

  it('shows the resolved entry title and stamp when an entry is present', () => {
    // Task 5: the slip title now renders through <Prose> (same fix as
    // OpenDossier's cover title — a resolved card's title can arrive
    // already glossary-marked; see tests/desk.dossier.test.ts's "OutTray —
    // a marked title" for the regression itself), which calls
    // useGameStore() internally — needs an active pinia, unlike the empty-
    // state test above (that branch never mounts <Prose> at all).
    const pinia = createPinia();
    setActivePinia(pinia);
    const wrapper = mount(OutTray, {
      global: { plugins: [pinia, i18n] },
      props: { entry: { title: 'Card One' } },
    });
    expect(wrapper.text()).toContain('Card One');
    expect(wrapper.text()).toContain(i18n.global.t('desk.out.resolved'));
  });
});

// Light integration smoke test: DeskView assembles the six static components
// from the real stores. The tray-rendering tests deliberately use ARBITRARY
// deck ids (not the game's `main.*` ones) — Task 2 killed DeskView's old
// per-id tray-matching table, so tray derivation is now generic (render one
// InTray per deskView.decks entry, in that order, skinned by its own role);
// using ids that don't look like the real game proves no id table crept back in.
// The real compiled ids are still exercised for real — see
// tests/integration.desk-loop.test.ts's real-game tray assertion, which is
// the guard that actually holds this honest.
describe('DeskView', () => {
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
        options: [{ id: '@x.alpha' }, { id: '@x.beta' }, { id: '@x.gamma' }, { id: '@pin1' }],
      },
      // decks fixture used by the DeskView mount — ids are arbitrary (NOT the
      // game's), roles are differentiated to exercise the skin routing:
      // deck-gov / deck-party / plain deck (the neutral-skin fallback
      // @debug_deck also uses in the real content).
      'x.alpha': {
        id: 'x.alpha',
        type: 'scene',
        title: 'Alpha',
        isDeck: true,
        role: 'deck-gov',
        content: [],
        options: [],
      },
      // The one deck with a card in it, so the hand can be filled for real.
      // x.alpha stays empty on purpose — the "empty deck -> toast" test below
      // depends on it.
      'x.beta': {
        id: 'x.beta',
        type: 'scene',
        title: 'Beta',
        isDeck: true,
        role: 'deck-party',
        content: [],
        options: [{ id: '#pcard' }],
      },
      pc1: {
        id: 'pc1',
        type: 'scene',
        title: 'Party Card',
        newPage: true,
        isCard: true,
        tags: ['pcard'],
        role: 'card-party',
        content: [{ type: 'paragraph', content: ['Party prose.'] }],
        options: [{ id: '@hub' }],
      },
      'x.gamma': {
        id: 'x.gamma',
        type: 'scene',
        title: 'Gamma',
        isDeck: true,
        role: 'deck',
        content: [],
        options: [],
      },
      pin1: {
        id: 'pin1',
        type: 'scene',
        title: 'Advisor',
        isPinnedCard: true,
        role: 'pinned-action',
        content: [{ type: 'paragraph', content: ['Advisor prose.'] }],
        options: [{ id: '@hub' }],
      },
    },
    qualities: {},
    qdisplays: {},
    tagLookup: { pcard: { pc1: true } },
  };

  let pinia: ReturnType<typeof createPinia>;
  beforeEach(() => {
    pinia = createPinia();
    setActivePinia(pinia);
    setAnimationsForTest(false);
  });

  function mountDesk() {
    const game = useGameStore();
    const desk = useDeskStore();
    game.initFromText(JSON.stringify(deskGame));
    game.newGame();
    game.choose(0); // root -> hub
    const wrapper = mount(DeskView, withI18n());
    return { game, desk, wrapper };
  }

  // Task 4 (typed note): builds a variant of deskGame whose hub.content is
  // swapped out for the given raw dendry content array, so the compiled
  // frame.html — and therefore the store's snapshot, deskStore.deskView.html
  // — carries an exact, controllable prose string. deskGame's other scenes
  // are reused via shallow spread (only hub.content is replaced).
  function mountDeskWithHubContent(content: unknown[]) {
    const gameJson = {
      ...deskGame,
      scenes: { ...deskGame.scenes, hub: { ...deskGame.scenes.hub, content } },
    };
    const game = useGameStore();
    const desk = useDeskStore();
    game.initFromText(JSON.stringify(gameJson));
    game.newGame();
    game.choose(0); // root -> hub
    const wrapper = mount(DeskView, withI18n());
    return { game, desk, wrapper };
  }

  it('renders one tray per deck, in deck order, skinned by the deck role', () => {
    const { wrapper } = mountDesk();
    const trays = wrapper.findAll('.in-tray');
    expect(trays).toHaveLength(3);
    expect(trays[0].classes()).toContain('skin-gov');
    expect(trays[1].classes()).toContain('skin-party');
    expect(trays[2].classes()).toContain('skin-neutral');
    expect(trays[0].text()).toContain('Alpha'); // caption = deck.title, no chrome label
  });

  it('renders no fixed chrome caption above trays', () => {
    const { wrapper } = mountDesk();
    expect(wrapper.find('.tray-kind-label').exists()).toBe(false);
  });

  it('renders the actions tray with the pinned advisor and the month/out tray chrome', () => {
    const { wrapper } = mountDesk();
    expect(wrapper.text()).toContain(i18n.global.t('desk.actions.title'));
    expect(wrapper.findAll('[data-test="pinned-card"]')).toHaveLength(1);
    expect(wrapper.text()).toContain(i18n.global.t('desk.month.3'));
    expect(wrapper.text()).toContain('2014');
    expect(wrapper.text()).toContain(i18n.global.t('desk.out.empty'));
  });

  it('clicking a tray dispatches to the desk store (empty deck -> toast, no crash)', async () => {
    const { desk, wrapper } = mountDesk();
    expect(desk.phase).toBe('idle');
    await wrapper.find('[data-test="in-tray-x.alpha"]').trigger('click');
    expect(desk.toastKey).toBe('desk.toast.deckEmpty');
  });

  // REGRESSION (C3): the desk used to render hand/decks/pinned straight from
  // the live frame, and the engine reports all three as [] on a card scene —
  // so the instant a card was played the hand, all three in-trays and the
  // whole actions tray VANISHED for the entire dossier window. The actions
  // tray and the right-hand cards sit outside the 850px dossier, so this was
  // plainly visible, not hidden behind it. It also made DeskView's
  // cardDimmed() unreachable dead code.
  it('keeps the hand, in-trays and actions tray on the desk while a dossier is open', async () => {
    const { game, desk, wrapper } = mountDesk();

    desk.drawFrom('x.beta');
    await nextTick();
    expect(wrapper.find('[data-test="hand-card-pc1"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="hand-card-pc1"]').classes()).not.toContain('dimmed');

    desk.playFromHand(game.frame!.hand[0]);
    await nextTick();
    expect(desk.phase).toBe('dossierOpen');
    expect(game.frame!.hand).toEqual([]); // the live frame really is blank

    // The desk is still there, and still furnished.
    const card = wrapper.find('[data-test="hand-card-pc1"]');
    expect(card.exists()).toBe(true);
    expect(card.classes()).toContain('dimmed'); // cardDimmed() is live code now
    expect(wrapper.find('[data-test="in-tray-x.alpha"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="in-tray-x.beta"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="in-tray-x.gamma"]').exists()).toBe(true);
    expect(wrapper.findAll('[data-test="pinned-card"]')).toHaveLength(1);
    expect(wrapper.text()).toContain(i18n.global.t('desk.actions.title'));
  });

  it('dossier dim covers the desk region only — never the Brief (user rule 2026-07-19)', async () => {
    const { game, desk, wrapper } = mountDesk();
    desk.drawFrom('x.beta');
    await nextTick();
    desk.playFromHand(game.frame!.hand[0]);
    await nextTick();
    expect(desk.phase).toBe('dossierOpen');

    const dim = wrapper.find('[data-test="desk-dim"]');
    expect(dim.exists()).toBe(true);
    // Structural guarantee: the overlay is inside .desk-region (inset:0 there),
    // and the clipboard is its sibling, outside it.
    expect(wrapper.find('.desk-region [data-test="desk-dim"]').exists()).toBe(true);
    expect(wrapper.find('.clipboard-frame [data-test="desk-dim"]').exists()).toBe(false);
    expect(wrapper.find('.clipboard-frame').exists()).toBe(true);
  });

  // Task 4 (spec §5.1 regression): DeskView never rendered frame.html, so
  // the desk scene's own monthly prose ([+ month : events2012 +],
  // historical_event, event_rubicon in main.scene.dry) was dropped on the
  // floor. The note reads the STORE's snapshot (deskStore.deskView.html),
  // not the live frame, for the same continuity reason as the rest of the
  // furniture — see stores/desk.ts's deskView comment.
  it('renders the desk prose as a typed note; hides it when empty', () => {
    const { wrapper } = mountDeskWithHubContent([
      { type: 'paragraph', content: ['Something moved this month.'] },
    ]);
    const note = wrapper.find('[data-test="desk-note"]');
    expect(note.exists()).toBe(true);
    expect(note.text()).toContain('Something moved this month.');
  });

  it('renders no note element when the desk prose is empty', () => {
    const empty = mountDeskWithHubContent([]);
    expect(empty.wrapper.find('[data-test="desk-note"]').exists()).toBe(false);

    // Whitespace-only/empty-tag HTML (e.g. every conditional insert on the
    // real scene suppressed for the current month) must not render an empty
    // paper scrap either — a run of tags with nothing but whitespace inside.
    const whitespace = mountDeskWithHubContent([{ type: 'paragraph', content: ['  \n  '] }]);
    expect(whitespace.wrapper.find('[data-test="desk-note"]').exists()).toBe(false);
  });

  // Review finding (fix round 1), REVERSED by user ruling 2026-07-19 Wave 2:
  // on the standard monthly path, dendry's paragraph buffer only clears on
  // `new-page: true`, and the desk hub scenes don't set it — so frame.html
  // carries a leftover `<h1>[month] [year]</h1>` heading from the PREVIOUS
  // page (post_event.scene.dry) ahead of the desk's own prose. That h1 IS
  // the desk's month+year title (translatable, carries the Rubicon week) —
  // fix round 1 stripped and discarded it; Task 7 EXTRACTS it instead, to
  // DeskMonth, rather than dropping it on the floor. A `heading`-type
  // content node is exactly what a `=` line in the .dry source compiles to
  // (vendor/dendrynexus-ten/lib/ui/content/html.js's `_paragraphsToHTML`,
  // case 'heading' -> '<h1>...</h1>'), so this fixture is the real compiled
  // shape, not a hand-written HTML string.
  it('extracts a leading <h1> heading as the desk title (DeskMonth), instead of stripping it from the note', () => {
    const { wrapper } = mountDeskWithHubContent([
      { type: 'heading', content: ['Novembre 2012'] },
      { type: 'paragraph', content: ['Events happened.'] },
    ]);

    // The content heading IS the title now — DeskMonth shows it, and the
    // Q-based month/year fallback must not ALSO render alongside it (that
    // would just move the double-title bug one level down).
    const month = wrapper.find('.pos-month');
    expect(month.text()).toContain('Novembre 2012');
    expect(month.find('.month').exists()).toBe(false);
    expect(month.find('.year').exists()).toBe(false);

    // The note keeps only the body prose — the heading is gone from it
    // (moved, not duplicated).
    const note = wrapper.find('[data-test="desk-note"]');
    expect(note.exists()).toBe(true);
    expect(note.text()).toContain('Events happened.');
    expect(note.text()).not.toContain('Novembre 2012');
  });

  it('renders no note element when the desk prose is only a leading heading — the heading becomes the title instead of vanishing', () => {
    const { wrapper } = mountDeskWithHubContent([{ type: 'heading', content: ['Novembre 2012'] }]);
    expect(wrapper.find('[data-test="desk-note"]').exists()).toBe(false);
    expect(wrapper.find('.pos-month').text()).toContain('Novembre 2012');
  });

  it('preserves a non-leading <h1> heading in mid-prose', () => {
    const { wrapper } = mountDeskWithHubContent([
      { type: 'paragraph', content: ['Intro.'] },
      { type: 'heading', content: ['Mid heading'] },
      { type: 'paragraph', content: ['More.'] },
    ]);
    const note = wrapper.find('[data-test="desk-note"]');
    expect(note.exists()).toBe(true);
    expect(note.text()).toContain('Intro.');
    expect(note.text()).toContain('Mid heading');
    expect(note.text()).toContain('More.');
  });

  // Regression (Task 7, Wave 2): when frame.html carries multiple leading
  // <h1> tags (first from post_event.scene.dry, second as accidental duplicate),
  // the while-loop at DeskView.vue:124 must drop all but the first. The first
  // becomes DeskMonth's title, the rest are removed, and body prose stays intact.
  it('drops duplicate leading <h1> headings; only the first becomes the title', () => {
    const { wrapper } = mountDeskWithHubContent([
      { type: 'heading', content: ['Setembre 2012'] },
      { type: 'heading', content: ['Duplicate'] },
      { type: 'paragraph', content: ['Body.'] },
    ]);

    // Only the first heading becomes the title
    const month = wrapper.find('.pos-month');
    expect(month.text()).toContain('Setembre 2012');

    // The note shows only the body, not either heading
    const note = wrapper.find('[data-test="desk-note"]');
    expect(note.exists()).toBe(true);
    expect(note.text()).toContain('Body.');
    expect(note.text()).not.toContain('Setembre 2012');
    expect(note.text()).not.toContain('Duplicate');
  });
});
