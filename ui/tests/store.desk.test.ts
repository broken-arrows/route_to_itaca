import { describe, it, expect, beforeEach, vi } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import { DELAYS } from '../src/components/desk/motion';

function compile(files: { name: string; contents: string }[]): Promise<any> {
  return new Promise((res, rej) => compileGame(files, (e, g) => (e ? rej(e) : res(g))));
}
async function jsonFor(files: { name: string; contents: string }[]): Promise<string> {
  const game = await compile(files);
  return new Promise<string>((res, rej) =>
    convertGameToJSON(game, 0, (e: Error | null, out?: string) => (e ? rej(e) : res(out!))),
  );
}

// Base fixture, copied from adapter.role.test.ts (Task 3) per the brief:
// implementers may read tasks independently, so this file carries its own
// copy rather than importing across test files.
const FILES = [
  { name: 'info.dry', contents: 'title: T\nauthor: A\nlanguages: en ca\n' },
  { name: 'root.scene.dry', contents: 'title: Root\n\nIntro.\n\n- @hub\n' },
  {
    name: 'hub.scene.dry',
    contents: 'title: Hub\nrole: desk\nis-hand: true\nmax-cards: 3\n\nDesk.\n\n- @gov_deck\n',
  },
  { name: 'gov_deck.scene.dry', contents: 'title: Gov\nrole: deck\nis-deck: true\n\n- #gcard\n' },
  {
    name: 'c1.scene.dry',
    contents: 'title: Card One\nrole: card-party\ntags: gcard\n\nCard prose.\n\n- @c1_next\n',
  },
  { name: 'c1_next.scene.dry', contents: 'title: After\n\nOutcome.\n\n- @hub: Back\n' },
];

// Variant: hub caps the hand at 1 card, so a single draw already fills it —
// used to force the 'no_space_in_hand' sentinel independently of deck state
// (the engine checks hand space BEFORE deck availability).
const FILES_HANDFULL = [
  FILES[0],
  FILES[1],
  {
    name: 'hub.scene.dry',
    contents: 'title: Hub\nrole: desk\nis-hand: true\nmax-cards: 1\n\nDesk.\n\n- @gov_deck\n',
  },
  FILES[3],
  FILES[4],
  FILES[5],
];

// Variant: c1 gets a third option (index 2) whose destination scene carries
// `choose-if: false`, so it's always locked (canChoose === false) — used for
// the shake guard.
const FILES_LOCKED = [
  FILES[0],
  FILES[1],
  FILES[2],
  FILES[3],
  {
    name: 'c1.scene.dry',
    contents:
      'title: Card One\nrole: card-party\ntags: gcard\n\nCard prose.\n\n- @c1_next\n- @c1_alt\n- @c1_locked\n',
  },
  FILES[5],
  { name: 'c1_alt.scene.dry', contents: 'title: Alt\n\nAlt outcome.\n\n- @hub: Back\n' },
  { name: 'c1_locked.scene.dry', contents: 'title: Locked\nchoose-if: false\n\nNope.\n\n- @hub: Back\n' },
];

// Variant: hub gains a pinned card (`role: pinned-action` derives
// isPinnedCard at compile time — no explicit attr needed), which the engine
// surfaces on frame.pinned rather than in the choices list.
const FILES_PIN = [
  FILES[0],
  FILES[1],
  {
    name: 'hub.scene.dry',
    contents:
      'title: Hub\nrole: desk\nis-hand: true\nmax-cards: 3\n\nDesk.\n\n- @gov_deck\n- @pin1\n',
  },
  FILES[3],
  FILES[4],
  FILES[5],
  {
    name: 'pin1.scene.dry',
    contents: 'title: Pin One\nrole: pinned-action\n\nPin prose.\n\n- @hub: Done\n',
  },
];

// Variant for Task 5's autosave tests: root seeds year/month/party_resources
// (so the stamp and SaveMeta.resources are assertable), and the outcome
// scene (c1_next, reached BEFORE the final "Back" pick) bumps month on
// arrival — per the brief, this is what drives the `${year}-${month}` stamp
// across a card's resolution back to the (role: desk) hub.
const FILES_AUTOSAVE = [
  FILES[0],
  {
    name: 'root.scene.dry',
    contents: 'title: Root\non-arrival: year = 2012; month = 1; party_resources = 3;\n\nIntro.\n\n- @hub\n',
  },
  FILES[2],
  FILES[3],
  FILES[4],
  { name: 'c1_next.scene.dry', contents: 'title: After\non-arrival: month += 1;\n\nOutcome.\n\n- @hub: Back\n' },
];

// Variant for Task 4 (typed note): hub's own prose is distinctive so the
// snapshot's `html` field is unambiguously assertable against `c1`'s own
// ("Card prose.", reused as-is) once the dossier is open. `new-page: true`
// on both is required here — without it dendry's CaptureUI never clears its
// paragraph buffer between scenes (only `scene.newPage` does, see
// engine.js's displaySceneContent), so each scene's prose would otherwise
// leak into the next one's html (real convention, confirmed against actual
// content: root.scene.dry:2 and every real card scene, e.g.
// generalitat_economy_card.scene.dry:2, set new-page: true).
const FILES_NOTE = [
  FILES[0],
  FILES[1],
  {
    name: 'hub.scene.dry',
    contents: 'title: Hub\nnew-page: true\nrole: desk\nis-hand: true\nmax-cards: 3\n\nNovember brief.\n\n- @gov_deck\n',
  },
  FILES[3],
  {
    name: 'c1.scene.dry',
    contents:
      'title: Card One\nnew-page: true\nrole: card-party\ntags: gcard\n\nCard prose.\n\n- @c1_next\n',
  },
  FILES[5],
];

// Variant for I1: c1's outcome routes OFF the desk, to a `role: event`
// scene. This is the STANDARD monthly path in the real game — a card's
// outcome go-to's post_event, which go-to's events_choice (role: event)
// whenever an event is due — so the "pick resolves back to role: desk"
// branch is NOT the common case it was written as.
const FILES_EVENT = [
  FILES[0],
  FILES[1],
  FILES[2],
  FILES[3],
  FILES[4],
  { name: 'c1_next.scene.dry', contents: 'title: After\n\nOutcome.\n\n- @ev1: To the event\n' },
  {
    name: 'ev1.scene.dry',
    contents: 'title: Event One\nrole: event\n\nAn event happens.\n\n- @hub: Back\n',
  },
];

// Raw-JSON fixture: it deliberately contains a dangling reference (`ghost`
// is not a scene), which the .dry compiler would reject at BUILD time —
// which is exactly why engine throws are a RUNTIME concern (a renamed
// scene, a stale save, a hand-edited game.json). The engine asserts a scene
// exists in both __changeScene (engine.js:978) and _compileChoices
// (engine.js:1467), so every fault below raises a REAL engine Error through
// the REAL adapter. No mocks, no stubs.
const brokenGame = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      newPage: true,
      onArrival: [{ $code: 'Q.year = 2012; Q.month = 1;' }],
      content: [{ type: 'paragraph', content: ['Root.'] }],
      options: [{ id: '@hub' }],
    },
    hub: {
      id: 'hub',
      type: 'scene',
      title: 'Hub',
      newPage: true,
      isHand: true,
      maxCards: 3,
      role: 'desk',
      content: [{ type: 'paragraph', content: ['Hub.'] }],
      options: [{ id: '@gov_deck' }, { id: '@pin1' }],
    },
    gov_deck: {
      id: 'gov_deck',
      type: 'scene',
      title: 'Gov',
      isDeck: true,
      role: 'deck',
      content: [],
      options: [{ id: '#gcard' }],
    },
    c1: {
      id: 'c1',
      type: 'scene',
      title: 'Card One',
      newPage: true,
      isCard: true,
      tags: ['gcard'],
      role: 'card-party',
      content: [{ type: 'paragraph', content: ['Card prose.'] }],
      options: [{ id: '@into_the_void' }],
    },
    // A perfectly ordinary, choosable paper whose outcome go-to's a scene
    // that does not exist: the throw happens INSIDE game.choose(i).
    into_the_void: {
      id: 'into_the_void',
      type: 'scene',
      title: 'Into the void',
      content: [],
      goTo: [{ id: 'ghost' }],
    },
    pin1: {
      id: 'pin1',
      type: 'scene',
      title: 'Pin One',
      isPinnedCard: true,
      role: 'pinned-action',
      content: [{ type: 'paragraph', content: ['Pin prose.'] }],
      options: [{ id: '@hub' }],
    },
  },
  qualities: {},
  qdisplays: {},
  tagLookup: { gcard: { c1: true } },
};

async function boot(files: { name: string; contents: string }[]) {
  const game = useGameStore();
  const desk = useDeskStore();
  game.initFromText(await jsonFor(files));
  game.newGame();
  return { game, desk };
}

function bootBroken() {
  const game = useGameStore();
  const desk = useDeskStore();
  game.initFromText(JSON.stringify(brokenGame));
  game.newGame();
  game.choose(0); // -> hub, idle
  return { game, desk };
}

describe('desk store', () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    setAnimationsForTest(false);
  });

  it('routes phases through a full draw→play→pick loop', async () => {
    const game = useGameStore();
    const desk = useDeskStore();
    expect(desk.phase).toBe('boot');

    game.initFromText(await jsonFor(FILES));
    game.newGame(); // -> root, role-less
    expect(desk.phase).toBe('page');

    game.choose(0); // -> hub, role: desk
    expect(desk.phase).toBe('idle');

    desk.drawFrom('gov_deck');
    // animations off: the commit in after() runs synchronously.
    expect(desk.phase).toBe('idle');
    expect(desk.flying).toBeNull();
    expect(game.frame?.hand.map((c) => c.id)).toEqual(['c1']);

    const card = game.frame!.hand[0];
    desk.playFromHand(card);
    expect(desk.phase).toBe('dossierOpen');
    expect(desk.openCard?.id).toBe('c1');

    desk.pickPaper(0); // -> c1_next, role-less: inherits card-party
    expect(desk.phase).toBe('dossierOpen'); // continuation, still open

    desk.pickPaper(0); // -> hub, role: desk -> resolves (0ms)
    expect(desk.phase).toBe('idle');
    expect(desk.outTray?.title).toBe('Card One');
    expect(desk.openCard).toBeNull();
  });

  it('guards: draw while dossier open is a no-op; empty/full-hand sentinels toast', async () => {
    const { game, desk } = await boot(FILES);
    game.choose(0); // -> hub, idle

    desk.drawFrom('gov_deck');
    const card = game.frame!.hand[0];
    desk.playFromHand(card);
    expect(desk.phase).toBe('dossierOpen');

    // guard: draw while dossier is open is a no-op.
    desk.drawFrom('gov_deck');
    expect(desk.phase).toBe('dossierOpen');
    expect(desk.flying).toBeNull();

    // Resolving the papers is the ONLY way out of a dossier — there is no
    // cancel (2026-07-13 user decision; returning a card to hand is the
    // engine's own difficulty-gated `easy_discard` paper, not a UI affordance).
    desk.pickPaper(0); // -> c1_next, continuation
    desk.pickPaper(0); // -> hub, role: desk -> idle
    expect(desk.phase).toBe('idle');

    // The deck offers c1 again once it has left the hand; draw it back in.
    desk.drawFrom('gov_deck');
    expect(game.frame?.hand.map((c) => c.id)).toEqual(['c1']);

    // Empty-deck sentinel: c1 is the deck's only card and it's already in
    // hand, so the next draw finds nothing available. The toast's dismiss
    // delay is FIXED at DELAYS.toast (information delivery, not motion — it
    // does not scale with the animations toggle), so toastKey is directly
    // assertable even with animations off. Fake timers around this one call
    // let us also pin the auto-clear at exactly the fixed delay.
    vi.useFakeTimers();
    try {
      desk.drawFrom('gov_deck');
      expect(desk.toastKey).toBe('desk.toast.deckEmpty');
      expect(desk.phase).toBe('idle'); // no state change
      vi.advanceTimersByTime(DELAYS.toast - 1);
      expect(desk.toastKey).toBe('desk.toast.deckEmpty'); // still showing
      vi.advanceTimersByTime(1);
      expect(desk.toastKey).toBeNull(); // auto-cleared at the fixed delay
    } finally {
      vi.useRealTimers();
    }

    // Full-hand sentinel: separate fixture/store pair, hub caps the hand at 1.
    setActivePinia(createPinia());
    setAnimationsForTest(false);
    const { game: game2, desk: desk2 } = await boot(FILES_HANDFULL);
    game2.choose(0); // -> hub, idle
    desk2.drawFrom('gov_deck'); // fills the single hand slot
    expect(game2.frame?.hand).toHaveLength(1);

    // Fake timers so the toast's 1700ms dismiss timer never leaks out of
    // the test as a real pending timeout (same hygiene as the deckEmpty
    // block above; useRealTimers discards the pending fake).
    vi.useFakeTimers();
    try {
      desk2.drawFrom('gov_deck');
      expect(desk2.toastKey).toBe('desk.toast.handFull');
      expect(desk2.phase).toBe('idle'); // no state change
    } finally {
      vi.useRealTimers();
    }
  });

  it('re-nudging the same key restarts the dismiss timer (single timer, latest wins)', () => {
    const desk = useDeskStore();
    vi.useFakeTimers();
    try {
      desk.nudge('k'); // t=0
      expect(desk.toastKey).toBe('k');
      vi.advanceTimersByTime(1000); // t=1000
      desk.nudge('k'); // SAME key re-triggered (e.g. double-clicking an empty deck's DRAW)
      vi.advanceTimersByTime(800); // t=1800 — past the FIRST nudge's 1700ms deadline
      expect(desk.toastKey).toBe('k'); // the stale first timer must not clear it early
      vi.advanceTimersByTime(900); // t=2700 — the SECOND nudge's full 1700ms elapsed
      expect(desk.toastKey).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('locked paper shakes and does not advance the engine', async () => {
    const { game, desk } = await boot(FILES_LOCKED);
    game.choose(0); // -> hub, idle

    desk.drawFrom('gov_deck');
    const card = game.frame!.hand[0];
    desk.playFromHand(card); // dossierOpen; c1 has 3 options, index 2 locked

    const lockedIndex = 2;
    expect(game.frame?.choices[lockedIndex]?.canChoose).toBe(false);

    // Unlike toasts, the shake IS motion (decorative) and its reset delay
    // scales with the animations toggle — with animations off it sets and
    // resets synchronously, which is the correct real-usage behaviour. Flip
    // animations on for just this call to observe shakeIdx, under fake
    // timers so the reset timer is controllable and never leaks as a real
    // pending timeout.
    setAnimationsForTest(true);
    vi.useFakeTimers();
    try {
      desk.pickPaper(lockedIndex);
      expect(desk.shakeIdx).toBe(lockedIndex);
      vi.advanceTimersByTime(DELAYS.cancel);
      expect(desk.shakeIdx).toBe(-1); // reset after the shake duration
    } finally {
      vi.useRealTimers();
      setAnimationsForTest(false);
    }

    expect(desk.phase).toBe('dossierOpen'); // unchanged
    expect(game.frame?.sceneId).toBe('c1'); // engine did not advance
  });

  // The desk store must expose NO way to un-play a card (2026-07-13 user
  // decision, overriding phase-2 spec §6): the engine's own `easy_discard`
  // paper is the one and only route back to the hand, and it is gated on
  // `difficulty <= 0 and not last_advisor_action` and charges a month action
  // (plus the card's timer and visit count). The removed store action restored
  // a whole-engine snapshot for free, on any card, at any difficulty — a
  // bypass of that gate. What follows is the guard: an open dossier's ONLY
  // engine-touching action is pickPaper.
  it('once a card is played, the only action that can leave the dossier is a paper', async () => {
    const { game, desk } = await boot(FILES);
    game.choose(0); // -> hub, idle
    desk.drawFrom('gov_deck');
    desk.playFromHand(game.frame!.hand[0]);
    expect(desk.phase).toBe('dossierOpen');

    // Every other action on the store surface is phase-guarded to a no-op.
    desk.drawFrom('gov_deck');
    desk.playFromHand({ id: 'c1', title: 'Card One', tags: [] });
    desk.playPinned({ id: 'c1', title: 'Card One', tags: [] });
    expect(desk.phase).toBe('dossierOpen');
    expect(game.frame?.sceneId).toBe('c1'); // the engine never moved
    expect(desk.openCard?.id).toBe('c1');

    desk.pickPaper(0); // the one thing that does move it
    expect(game.frame?.sceneId).toBe('c1_next');
  });

  it('playPinned opens a dossier and resolves to the out tray', async () => {
    const { game, desk } = await boot(FILES_PIN);
    game.choose(0); // -> hub, idle
    expect(desk.phase).toBe('idle');

    const pin = game.frame!.pinned.find((p) => p.id === 'pin1');
    expect(pin).toBeDefined();

    desk.playPinned(pin!);
    expect(desk.phase).toBe('dossierOpen');
    expect(desk.openCard?.id).toBe('pin1');

    // The single paper is the only way out — pick it back to the desk.
    desk.pickPaper(0); // -> hub, role: desk -> resolving -> idle (0ms)
    expect(desk.phase).toBe('idle');
    expect(desk.outTray?.title).toBe('Pin One');
    expect(desk.openCard).toBeNull();
  });

  it('the first desk arrival of a session initializes the autosave stamp without saving', async () => {
    const { game, desk } = await boot(FILES_AUTOSAVE);
    game.choose(0); // -> hub, idle: first-ever idle entry this session
    expect(desk.phase).toBe('idle');
    expect(game.listSlots()).toHaveLength(0); // stamp initialized silently, nothing written
  });

  it('autosaves into the auto-1/auto-2 rotation only when the year-month stamp changes', async () => {
    const { game, desk } = await boot(FILES_AUTOSAVE);
    game.choose(0); // -> hub, idle: initializes the stamp at month 1, no save yet
    expect(game.listSlots()).toHaveLength(0);

    // Play the deck's one card through to its outcome: c1_next bumps month
    // on arrival, so the FINAL pick (c1_next -> hub) crosses a stamp
    // boundary and autosaves into 'auto-1'.
    desk.drawFrom('gov_deck');
    let card = game.frame!.hand[0];
    desk.playFromHand(card);
    desk.pickPaper(0); // -> c1_next (continuation, still dossierOpen): month 1 -> 2
    desk.pickPaper(0); // -> hub (role desk): resolves -> idle, stamp changed -> autosave

    let slots = game.listSlots();
    expect(slots.map((s) => s.slot)).toEqual(['auto-1']);
    expect(slots[0].month).toBe(2);
    expect(slots[0].resources).toBe(3); // SaveMeta.resources <- Q.party_resources
    const auto1SavedAt = slots[0].savedAt;

    // Same-stamp action: drawing again re-enters idle via the frame watch
    // (the deck redraws freely once a card leaves the hand), but month
    // hasn't moved, so nothing new is written — the existing slot is
    // byte-identical, not just "still present".
    desk.drawFrom('gov_deck');
    slots = game.listSlots();
    expect(slots.map((s) => s.slot)).toEqual(['auto-1']);
    expect(slots[0].savedAt).toBe(auto1SavedAt);

    // A second month-crossing resolution flips the rotation to 'auto-2' and
    // leaves 'auto-1' untouched.
    card = game.frame!.hand[0];
    desk.playFromHand(card);
    desk.pickPaper(0); // -> c1_next: month 2 -> 3
    desk.pickPaper(0); // -> hub: resolves -> idle, stamp changed -> autosave

    slots = game.listSlots();
    expect(slots.map((s) => s.slot).sort()).toEqual(['auto-1', 'auto-2']);
    expect(slots.find((s) => s.slot === 'auto-2')!.month).toBe(3);
    expect(slots.find((s) => s.slot === 'auto-1')!.savedAt).toBe(auto1SavedAt);
  });

  // REGRESSION (I3): the rotation slot used to be a CLOSURE variable
  // (`lastAutoSlot`), reset to null on every page load — so the first
  // month-crossing of every new session always wrote 'auto-1' and destroyed
  // the newest autosave, while 'auto-2' sat empty. The target slot must be
  // derived from what is actually STORED.
  it('the rotation survives a reload: a new session writes the free slot, not the newest one', async () => {
    // --- session A: one month-crossing -> auto-1.
    const { game, desk } = await boot(FILES_AUTOSAVE);
    game.choose(0);
    desk.drawFrom('gov_deck');
    desk.playFromHand(game.frame!.hand[0]);
    desk.pickPaper(0); // -> c1_next: month 1 -> 2
    desk.pickPaper(0); // -> hub: autosave -> auto-1
    expect(game.listSlots().map((s) => s.slot)).toEqual(['auto-1']);
    const auto1SavedAt = game.listSlots()[0].savedAt;

    // --- session B: a page reload. Fresh pinia = fresh store closures (the
    // whole point), while localStorage persists exactly as a real reload.
    setActivePinia(createPinia());
    setAnimationsForTest(false);
    const { game: game2, desk: desk2 } = await boot(FILES_AUTOSAVE);
    game2.choose(0); // -> hub, idle: re-initializes the stamp at month 1
    desk2.drawFrom('gov_deck');
    desk2.playFromHand(game2.frame!.hand[0]);
    desk2.pickPaper(0); // month 1 -> 2
    desk2.pickPaper(0); // -> hub: stamp moved -> autosave

    const slots = game2.listSlots();
    // Pre-fix this was ['auto-1'] — session B overwrote the only autosave.
    expect(slots.map((s) => s.slot).sort()).toEqual(['auto-1', 'auto-2']);
    expect(slots.find((s) => s.slot === 'auto-1')!.savedAt).toBe(auto1SavedAt); // untouched
    expect(slots.find((s) => s.slot === 'auto-2')!.month).toBe(2);
  });

  // REGRESSION (C3): the engine only fills displayHand/displayDecks/
  // displayPinnedCards on `is-hand` scenes (engine.js:334), and
  // CaptureUI.resetTransient() clears those buffers at the start of every
  // adapter action — so on a CARD scene the live frame reports hand, decks
  // and pinned as []. DeskView used to render straight from the live frame,
  // which blanked the entire desk (hand, in-trays, actions tray) for the
  // whole dossierOpen window. The store must keep its own snapshot, the
  // same continuity duty it already owns for openCard and resolveView.
  it('keeps the desk furniture (hand/decks/pinned) while the dossier is open', async () => {
    const { game, desk } = await boot(FILES_PIN);
    game.choose(0); // -> hub, idle

    desk.drawFrom('gov_deck');
    expect(desk.deskView.hand.map((c) => c.id)).toEqual(['c1']); // live at idle
    expect(desk.deskView.decks.map((d) => d.id)).toEqual(['gov_deck']);
    expect(desk.deskView.pinned.map((p) => p.id)).toEqual(['pin1']);
    expect(desk.deskView.maxCards).toBe(3);

    desk.playFromHand(game.frame!.hand[0]);
    expect(desk.phase).toBe('dossierOpen');

    // The live frame really is blank — this is the engine behaviour the
    // snapshot exists to paper over, asserted so the fix cannot rot.
    expect(game.frame!.isHand).toBe(false);
    expect(game.frame!.hand).toEqual([]);
    expect(game.frame!.decks).toEqual([]);
    expect(game.frame!.pinned).toEqual([]);

    // ...and the desk is still fully furnished behind the dossier.
    expect(desk.deskView.hand.map((c) => c.id)).toEqual(['c1']);
    expect(desk.deskView.decks.map((d) => d.id)).toEqual(['gov_deck']);
    expect(desk.deskView.pinned.map((p) => p.id)).toEqual(['pin1']);
    expect(desk.deskView.maxCards).toBe(3);
  });

  // Task 4 (typed note): the desk scene's own prose (frame.html — dropped on
  // the floor since phase 2, see docs/design/desk_ui_plan.md §5.1) must be
  // snapshotted like the rest of the furniture, and must keep showing the
  // DESK's prose (not the open card's) for the whole dossierOpen window —
  // same continuity duty deskSnapshot already owns for hand/decks/pinned.
  it('deskView.html carries the desk scene prose and survives the dossier window', async () => {
    const { game, desk } = await boot(FILES_NOTE);
    game.choose(0); // -> hub, role: desk, is-hand
    expect(desk.deskView.html).toBe('<p>November brief.</p>');

    desk.drawFrom('gov_deck');
    desk.playFromHand(game.frame!.hand[0]); // -> c1: furniture-less card frame
    expect(desk.phase).toBe('dossierOpen');
    expect(game.frame!.html).toBe('<p>Card prose.</p>'); // the live frame really is the card's
    expect(desk.deskView.html).toBe('<p>November brief.</p>'); // snapshot, not the card's
  });

  // REGRESSION (I1): pickPaper only had two branches — "lands on role: desk"
  // (resolve) and "still a dossier" (continuation). A pick that routes off
  // the desk entirely fell through BOTH: openCard leaked and the OUT tray
  // was never stamped. That is the standard monthly path in the real game.
  it('a pick that routes off the desk still lands the card in the OUT tray', async () => {
    const { game, desk } = await boot(FILES_EVENT);
    game.choose(0); // -> hub, idle

    desk.drawFrom('gov_deck');
    desk.playFromHand(game.frame!.hand[0]);
    desk.pickPaper(0); // -> c1_next: continuation, still dossierOpen
    expect(desk.phase).toBe('dossierOpen');

    desk.pickPaper(0); // -> ev1 (role: event): routes OFF the desk
    expect(desk.phase).toBe('eventPage');
    expect(desk.outTray?.title).toBe('Card One'); // the card actually landed
    expect(desk.openCard).toBeNull(); // ...and did not leak

    // Back to the desk from the event page: no stale openCard, so a later
    // redraw of the same card id would not render permanently dimmed.
    game.choose(0); // ev1 -> hub
    expect(desk.phase).toBe('idle');
    expect(desk.openCard).toBeNull();
  });

  // The choreography is presentation only: with animations ON the off-desk
  // pick plays the SAME resolve fly-out as a desk-bound pick before the page
  // takes over, and with animations off it jumps straight to 'eventPage'
  // through that identical code path (after(0) is synchronous).
  it('with animations on, an off-desk pick plays the resolve fly-out, then routes to the page', async () => {
    const { game, desk } = await boot(FILES_EVENT);
    game.choose(0);
    desk.drawFrom('gov_deck');
    desk.playFromHand(game.frame!.hand[0]);
    desk.pickPaper(0); // -> c1_next, continuation

    vi.useFakeTimers();
    setAnimationsForTest(true);
    try {
      desk.pickPaper(0); // -> ev1
      expect(desk.phase).toBe('resolving'); // fly-out is in flight
      expect(desk.outTray?.title).toBe('Card One');
      expect(desk.resolveView).not.toBeNull();

      vi.advanceTimersByTime(DELAYS.resolve);
      expect(desk.phase).toBe('eventPage'); // ...then the page takes over
      expect(desk.openCard).toBeNull();
      expect(desk.resolveView).toBeNull();
    } finally {
      vi.useRealTimers();
      setAnimationsForTest(false);
    }
  });
});

// C2 / spec §9 (docs/design/desk_ui_plan.md:329-332): "wrap adapter actions
// in try/catch; surface a visible toast + console detail. Never leave the
// choreography state machine stuck — on error, return to `idle` with the
// desk unlocked." Every fault below is a REAL throw out of the REAL engine
// (see brokenGame), driven through the real store.
describe('desk store — engine errors never leave the machine stuck (spec §9)', () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    setAnimationsForTest(false);
  });

  // Fake timers: the recovery toast schedules a real 1700ms dismiss (its
  // delay is deliberately NOT animMs-scaled), which would otherwise leak out
  // of the test as a pending timeout. console.error is silenced but ASSERTED
  // — "console detail" is half the spec requirement.
  function withSilencedError<T>(fn: (errSpy: ReturnType<typeof vi.spyOn>) => T): T {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.useFakeTimers();
    try {
      return fn(errSpy);
    } finally {
      vi.useRealTimers();
      errSpy.mockRestore();
    }
  }

  it('a throwing draw recovers to an unlocked idle desk with an error toast', () => {
    const { desk } = bootBroken();
    withSilencedError((errSpy) => {
      // 'ghost_deck' is not a scene: _drawFromDeck -> _compileChoices asserts.
      expect(() => desk.drawFrom('ghost_deck')).not.toThrow();
      expect(desk.phase).toBe('idle');
      expect(desk.flying).toBeNull();
      expect(desk.toastKey).toBe('desk.toast.engineError');
      expect(errSpy).toHaveBeenCalled();
    });
  });

  it('a throwing play recovers to an unlocked idle desk (no half-open dossier)', () => {
    const { desk } = bootBroken();
    withSilencedError(() => {
      expect(() => desk.playFromHand({ id: 'ghost', title: 'Ghost', tags: [] })).not.toThrow();
      expect(desk.phase).toBe('idle');
      expect(desk.openCard).toBeNull();
      expect(desk.toastKey).toBe('desk.toast.engineError');
    });
  });

  it('a throwing pinned play recovers to an unlocked idle desk', () => {
    const { desk } = bootBroken();
    withSilencedError(() => {
      expect(() => desk.playPinned({ id: 'ghost', title: 'Ghost', tags: [] })).not.toThrow();
      expect(desk.phase).toBe('idle');
      expect(desk.openCard).toBeNull();
      expect(desk.toastKey).toBe('desk.toast.engineError');
    });
  });

  // THE stuck state the spec forbids, and now the ONLY escape from a dossier:
  // with the ✕ removed (2026-07-13), picking a paper is the sole interaction
  // inside one. If a throwing pick left the machine in 'dossierOpen', the next
  // pick would throw again — permanently unplayable. runEngine's recovery to a
  // clean idle is what makes removing the cancel safe.
  it('a throwing pick does not strand the desk in a dossier', () => {
    const { game, desk } = bootBroken();
    desk.drawFrom('gov_deck');
    desk.playFromHand(game.frame!.hand[0]);
    expect(desk.phase).toBe('dossierOpen');

    withSilencedError(() => {
      // into_the_void go-to's a scene that does not exist -> engine throws.
      expect(() => desk.pickPaper(0)).not.toThrow();
      expect(desk.phase).toBe('idle'); // NOT stuck in 'dossierOpen'
      expect(desk.openCard).toBeNull();
      expect(desk.resolveView).toBeNull();
      expect(desk.toastKey).toBe('desk.toast.engineError');
    });

    // The engine is stranded off the desk (its frame is still the card
    // scene, which carries no furniture) — but thanks to C3's snapshot the
    // player gets a real, usable desk back rather than a blank one.
    expect(game.frame!.isHand).toBe(false);
    expect(desk.deskView.hand.map((c) => c.id)).toEqual(['c1']);
    expect(desk.deskView.decks.map((d) => d.id)).toEqual(['gov_deck']);
  });
});
