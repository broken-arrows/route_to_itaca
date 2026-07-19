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

  it('renders one entry per pinned card', () => {
    const wrapper = mount(ActionsTray, withI18n({ props: { pinned } }));
    expect(wrapper.findAll('[data-test="pinned-card"]')).toHaveLength(3);
  });

  it('emits play with the clicked card', async () => {
    const wrapper = mount(ActionsTray, withI18n({ props: { pinned } }));
    await wrapper.findAll('[data-test="pinned-card"]')[1].trigger('click');
    expect(wrapper.emitted('play')).toEqual([[pinned[1]]]);
  });

  it('does not emit when disabled', async () => {
    const wrapper = mount(ActionsTray, withI18n({ props: { pinned, disabled: true } }));
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
// from the real stores (mirrors main.scene.dry's known deck ids so the
// government/party/parlament tray-matching table is exercised for real).
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
        options: [
          { id: '@main.cat_gov' },
          { id: '@main.party_erc' },
          { id: '@main.parlament_deck' },
          { id: '@pin1' },
        ],
      },
      // NB the `main.` prefix: dendry compiles a `@section` of `main.scene.dry`
      // to the scene id `main.<section>`. These fixtures once used the bare
      // names, which is exactly the id DeskView's TRAY_KINDS wrongly matched on
      // — so the fixture agreed with the bug and the desk rendered no in-trays
      // against the real game while these tests stayed green. Keep fixture ids
      // shaped like the real compiled ones.
      'main.cat_gov': {
        id: 'main.cat_gov',
        type: 'scene',
        title: 'Generalitat',
        isDeck: true,
        role: 'deck',
        content: [],
        options: [],
      },
      // The one deck with a card in it, so the hand can be filled for real.
      // main.cat_gov stays empty on purpose — the "empty deck -> toast" test
      // below depends on it.
      'main.party_erc': {
        id: 'main.party_erc',
        type: 'scene',
        title: 'Party Affairs',
        isDeck: true,
        role: 'deck',
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
      'main.parlament_deck': {
        id: 'main.parlament_deck',
        type: 'scene',
        title: 'Parlament',
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

  it('renders the government, party and parlament trays with the fixed chrome labels', () => {
    const { wrapper } = mountDesk();
    const text = wrapper.text();
    // Assert the literal rendered captions — NOT t('desk.tray.*'), which is
    // vacuous here: the component renders those same keys through the same
    // i18n instance, so expected and actual would match even if the key moved.
    expect(text).toContain('Generalitat');
    expect(text).toContain('Party Affairs');
    expect(text).toContain('Parlament');
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
    await wrapper.find('[data-test="in-tray-main.cat_gov"]').trigger('click');
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

    desk.drawFrom('main.party_erc');
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
    expect(wrapper.find('[data-test="in-tray-main.cat_gov"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="in-tray-main.party_erc"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="in-tray-main.parlament_deck"]').exists()).toBe(true);
    expect(wrapper.findAll('[data-test="pinned-card"]')).toHaveLength(1);
    expect(wrapper.text()).toContain(i18n.global.t('desk.actions.title'));
  });
});
