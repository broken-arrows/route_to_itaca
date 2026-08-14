import { computed, ref, watch } from 'vue';
import { defineStore } from 'pinia';
import { useGameStore } from './game';
import { useSettingsStore } from './settings';
import { newlyUnlocked } from './achievements';
import type { CardView, ChoiceView, DeckView, EffectiveRole } from '../engine/types';
import { DELAYS, after } from '../components/desk/motion';
import type { AchievementToastPayload } from '../components/desk/Toast.vue';

// The Desk's own achievement-unlock toast payload (phase 2.5 Task 8 — a
// parity gap since phase 2: the old shell has always had achievementNotif,
// the Desk never had an equivalent). Deliberately NOT an i18n key like
// `toastKey` below — the name/image/stars are dynamic, per-achievement game
// content (game.data.achievements), not a fixed string the catalogs can
// hold. `toastKey` stays exactly as it is for the fixed-copy nudges
// (engineError/handFull/deckEmpty); this is a second, parallel channel.
// Type itself lives in Toast.vue (the component whose props define the
// contract) and is re-exported here for every other consumer — was
// declared byte-identically in both files.
export type { AchievementToastPayload };

// The desk's own furniture. Kept as a store-owned snapshot rather than read
// live off the frame — see `deskView` below for why.
export interface DeskFurniture {
  hand: CardView[];
  decks: DeckView[];
  pinned: CardView[];
  maxCards: number;
  html: string; // the desk scene's own prose — the typed note (phase 3a)
}
const EMPTY_DESK: DeskFurniture = { hand: [], decks: [], pinned: [], maxCards: 0, html: '' };

export type DeskPhase =
  | 'boot'
  | 'page'
  | 'idle'
  | 'drawing'
  | 'dossierOpen'
  | 'resolving'
  | 'newspaper'
  | 'eventPage';

// Animations toggle now lives in the settings store; this wrapper keeps the
// name/shape every existing caller (Task 4's tests) already uses. It needs
// an active pinia (same as calling useSettingsStore() anywhere else).
export function setAnimationsForTest(on: boolean): void {
  useSettingsStore().setAnimations(on);
}

function routePhase(role: EffectiveRole): DeskPhase {
  if (role === 'desk') return 'idle';
  if (role === 'newspaper') return 'newspaper';
  if (role === 'event') return 'eventPage';
  if (role.startsWith('pinned-') || role.startsWith('card')) return 'dossierOpen';
  return 'page';
}

// The desk store is presentation-only choreography: every transition here
// is caused by, and resolves into, an ordinary engine action on the game
// store. It owns exactly one thing the game store doesn't: the phase/motion
// state machine that sequences those actions into draw/play/pick beats.
//
// There is deliberately NO "close the dossier" action (2026-07-13, user
// decision; it overrides phase-2 spec §6's cancel contract). Playing a card
// is a committed engine action: the ONLY way back to the hand is the game's
// own `easy_discard` paper, which every card offers, is gated on
// `difficulty <= 0 and not last_advisor_action`, and charges a real cost (a
// month action, the card's timer, its visit count). The old ✕ restored a
// whole-engine snapshot for free, on any card, at any difficulty — a bypass
// of that gate, not a duplicate of it.
export const useDeskStore = defineStore('desk', () => {
  const game = useGameStore();
  const settings = useSettingsStore();

  const phase = ref<DeskPhase>('boot');
  const flying = ref<CardView | null>(null);
  const flyingFrom = ref<string | null>(null);
  const openCard = ref<CardView | null>(null);
  const outTray = ref<{ title: string } | null>(null);
  const toastKey = ref<string | null>(null);
  const achievementToast = ref<AchievementToastPayload | null>(null);
  const shakeIdx = ref(-1);
  // Snapshot of the dossier's cover prose + papers, taken by pickPaper just
  // BEFORE the resolving pick's engine call and published for the whole
  // 'resolving' window: game.choose(i) advances the frame to the destination
  // desk synchronously (the goTo chain recurses inside one choose), so the
  // live frame already shows the hub — an isHand scene with no choices —
  // while the dossier is still flying out. Same continuity duty the store
  // already owns for openCard/outTray. Null outside 'resolving'.
  const resolveView = ref<{ html: string; choices: ChoiceView[] } | null>(null);

  // Snapshot of the desk's furniture, taken from the last frame that actually
  // CARRIED any. The engine fills displayHand/displayDecks/displayPinnedCards
  // only for `is-hand` scenes (engine.js:334), and CaptureUI.resetTransient()
  // clears those buffers at the start of every adapter action — so on a card
  // scene (the whole 'dossierOpen' window) frame.hand/decks/pinned are all [].
  // Rendering the desk straight from the live frame therefore blanked it — the
  // hand, all three in-trays and the actions tray — the instant a card was
  // played. Exactly the same continuity duty the store already owns for the
  // dossier's prose (resolveView) and its card (openCard).
  const deskSnapshot = ref<DeskFurniture | null>(null);

  // Render live ONLY while we are actually at the desk and the frame still
  // carries furniture; otherwise fall back to the last known desk. The
  // `isHand` half of the guard is what keeps the desk usable after an engine
  // error too: recovery forces phase back to 'idle' while the engine is still
  // stranded on a (furniture-less) card scene.
  const deskView = computed<DeskFurniture>(() => {
    const f = game.frame;
    if (phase.value === 'idle' && f?.isHand) {
      return { hand: f.hand, decks: f.decks, pinned: f.pinned, maxCards: f.maxCards, html: f.html };
    }
    return deskSnapshot.value ?? EMPTY_DESK;
  });

  // Single-timer handles for the two re-triggerable effects. Cancelling the
  // pending timer before scheduling a new one is what makes "latest wins"
  // true for IDENTICAL keys/indices too — without it, a stale earlier timer
  // fires on schedule and cuts the re-triggered effect short (e.g.
  // double-clicking an empty deck's DRAW would show the second toast for
  // less than its full duration).
  let toastTimer: ReturnType<typeof setTimeout> | undefined;
  let shakeTimer: ReturnType<typeof setTimeout> | undefined;

  function animMs(key: keyof typeof DELAYS): number {
    return settings.animations ? DELAYS[key] : 0;
  }

  // Turn-boundary autosave. `autosaveStamp` is this session's last-seen
  // `${year}-${month}`; `null` means "not initialized yet". The FIRST desk
  // arrival of a session only initializes the stamp — it must NOT save,
  // otherwise loading a save (which immediately routes to idle) would
  // clobber the auto-1/auto-2 rotation before the player has done anything
  // new. Every later desk OR newspaper-listing entry whose stamp differs
  // from the last-seen one rotates the slot. The newspaper entry matters:
  // it is the normal first stable frame after a month action, and auto-1 must
  // capture that listing rather than wait until the complete news sequence
  // has returned to the desk.
  let autosaveStamp: string | null = null;

  let prevQ: Record<string, unknown> = {};
  let seeded = false;
  let achievementQueue: AchievementToastPayload[] = [];
  let achievementToastTimer: ReturnType<typeof setTimeout> | undefined;

  function showNextAchievementToast(): void {
    const next = achievementQueue.shift();
    achievementToast.value = next ?? null;
    if (!next) return;
    achievementToastTimer = after(DELAYS.achievementToast, () => {
      achievementToastTimer = undefined;
      showNextAchievementToast();
    });
  }

  function enqueueAchievementToast(payload: AchievementToastPayload): void {
    achievementQueue.push(payload);
    // A toast is already showing (and its timer already scheduled) — it
    // will pick up the queue when it dismisses. Otherwise start showing now.
    if (achievementToast.value === null && achievementToastTimer === undefined) {
      showNextAchievementToast();
    }
  }

  function checkAchievements(): void {
    if (!game.frame) return; // pre-boot: nothing real to seed from yet
    const q = game.q;
    if (!seeded) {
      prevQ = q;
      seeded = true;
      return;
    }
    const ids = newlyUnlocked(prevQ, q);
    prevQ = q;
    if (ids.length === 0) return;
    for (const id of ids) {
      const entry = game.achievements.find((a) => a.id === id);
      if (!entry) continue; // registry gap — nothing to show, not a crash
      enqueueAchievementToast({ name: entry.name, image: entry.image, stars: entry.stars });
    }
  }

  function maybeAutosave(): void {
    const year = game.q.year;
    const month = game.q.month;
    const stamp = `${year}-${month}`;
    if (autosaveStamp === null) {
      autosaveStamp = stamp;
      return;
    }
    if (stamp === autosaveStamp) return;
    autosaveStamp = stamp;
    game.saveAutosave();
  }

  function recoverToIdle(): void {
    flying.value = null;
    flyingFrom.value = null;
    openCard.value = null;
    resolveView.value = null;
    phase.value = 'idle';
    nudge('desk.toast.engineError');
  }

  function runEngine<T>(fn: () => T): { ok: true; value: T } | { ok: false; value?: undefined } {
    try {
      return { ok: true, value: fn() };
    } catch (err) {
      console.error('[desk] engine action failed:', err);
      recoverToIdle();
      return { ok: false };
    }
  }

  // The single source of truth for "what does the frame we're currently on
  // say the phase should be". Called automatically (see the watch below)
  // whenever game.frame changes, AND explicitly by actions that want the
  // transition to read as "caused by the engine state", per the contract.
  // Mid-animation phases ('drawing'/'resolving') are commitments an action
  // already made on purpose — a frame change that happens to occur while
  // one is in flight (there isn't one today, but the guard is what makes
  // that safe to add later) must not be able to stomp on it.
  function syncFromFrame(): void {
    // Runs on EVERY frame change regardless of phase — achievements are
    // content-driven, independent of the desk's own choreography, and must
    // not be skipped just because a fly-out/resolve animation happens to be
    // in flight (see checkAchievements's own comment).
    checkAchievements();
    if (phase.value === 'drawing' || phase.value === 'resolving') return;
    const f = game.frame;
    const next = f ? routePhase(game.effectiveRole) : 'boot';
    // Re-snapshot the desk from every frame that carries furniture (an
    // is-hand scene), so the dossier/event windows — where the frame carries
    // none — always have a last-known-good desk to render behind them.
    if (f?.isHand) {
      deskSnapshot.value = { hand: f.hand, decks: f.decks, pinned: f.pinned, maxCards: f.maxCards, html: f.html };
    }
    phase.value = next;
    if (next === 'idle' || next === 'newspaper') maybeAutosave();
  }

  watch(() => game.frame, syncFromFrame, { immediate: true, flush: 'sync' });

  function drawFrom(deckId: string): void {
    if (phase.value !== 'idle') return;
    const drawn = runEngine(() => game.draw(deckId));
    if (!drawn.ok) return;
    const result = drawn.value;
    if (result.id === null) {
      nudge(result.title === 'no_space_in_hand' ? 'desk.toast.handFull' : 'desk.toast.deckEmpty');
      return;
    }
    flying.value = result;
    flyingFrom.value = deckId;
    phase.value = 'drawing';
    after(animMs('draw'), () => {
      flying.value = null;
      phase.value = 'idle';
    });
  }

  function playFromHand(card: CardView): void {
    if (phase.value !== 'idle') return;
    if (!runEngine(() => game.play(card.id)).ok) return;
    openCard.value = card;
    // Belt-and-suspenders: the flush:'sync' frame watch already applied this
    // transition during game.play(); kept as explicit intent, not load-bearing.
    syncFromFrame();
  }

  function playPinned(card: CardView): void {
    if (phase.value !== 'idle' || card.canChoose === false) return;
    if (!runEngine(() => game.playPinned(card.id)).ok) return;
    openCard.value = card;
    // Belt-and-suspenders: the flush:'sync' frame watch already applied this
    // transition during game.playPinned(); kept as explicit intent, not load-bearing.
    syncFromFrame();
  }

  function pickPaper(i: number): void {
    if (phase.value !== 'dossierOpen') return;
    const choice = game.frame?.choices[i];
    if (!choice || !choice.canChoose) {
      // Same stored-handle pattern as nudge (re-clicking a locked paper
      // must restart the shake, not get cut short by the first click's
      // reset timer). Stays animMs-scaled: the shake is decorative motion,
      // so with animations off it sets + resets synchronously.
      shakeIdx.value = i;
      if (shakeTimer !== undefined) clearTimeout(shakeTimer);
      shakeTimer = after(animMs('cancel'), () => {
        shakeTimer = undefined;
        shakeIdx.value = -1;
      });
      return;
    }
    // Capture the pre-pick dossier view BEFORE the engine advances the
    // frame (see resolveView above). Safe to snapshot by reference: the
    // game store replaces frame.value wholesale on every action, it never
    // mutates the old frame's html/choices in place.
    const view = game.frame ? { html: game.frame.html, choices: game.frame.choices } : null;
    if (!runEngine(() => game.choose(i)).ok) return;

    // A multi-step continuation: the destination is role-less and inherits
    // the card's role, so the frame watch already reasserted 'dossierOpen'.
    // The papers just show the new choices now.
    if (routePhase(game.effectiveRole) === 'dossierOpen') return;

    // Otherwise the dossier RESOLVED. Two destinations, one choreography:
    // back to the desk (role: desk), or off the desk entirely — an event or
    // ending page. The latter is the standard monthly path in the real game
    // (a card's outcome go-to's post_event, which go-to's events_choice when
    // an event is due) and used to fall through both branches: the card never
    // reached the OUT tray and `openCard` leaked into the next desk arrival,
    // where cardDimmed() would render that card permanently dimmed if it were
    // ever drawn back into the hand.
    resolveView.value = view;
    outTray.value = { title: openCard.value ? openCard.value.title : '' };
    phase.value = 'resolving';
    after(animMs('resolve'), () => {
      openCard.value = null;
      resolveView.value = null;
      // Release the mid-animation guard, then let the LIVE frame say where we
      // land: 'idle' at the desk, or 'eventPage'/'page' when the pick routed
      // off it. maybeAutosave() may re-run here for the desk case (the watch
      // already fired during game.choose) — it is idempotent for an unchanged
      // stamp. With animations off, after(0) runs this synchronously: the
      // machine jumps straight to its destination through the SAME path.
      phase.value = 'idle';
      syncFromFrame();
    });
  }

  // Newspaper stories and front-page answers are ordinary engine choices.
  // These thin, phase-guarded entry points keep the presentation from
  // acquiring any event-specific navigation or eligibility rules.
  function chooseNewspaperStory(i: number): void {
    if (phase.value !== 'newspaper') return;
    const choice = game.frame?.choices[i];
    if (!choice?.canChoose) return;
    if (!runNewsEngine(() => game.choose(i))) return;

    // Most listing choices open an event page and are not a completed story.
    // A choice whose authored go-to resolves immediately is still valid; in
    // that case checkpoint the resulting listing/desk just like a front-page
    // resolution.
    if (routePhase(game.effectiveRole) !== 'eventPage') game.checkpointAutosave();
  }

  function chooseEventChoice(i: number): void {
    if (phase.value !== 'eventPage') return;
    const choice = game.frame?.choices[i];
    if (!choice?.canChoose) return;
    if (!runNewsEngine(() => game.choose(i))) return;

    // Role inheritance keeps multi-stage event continuations on eventPage.
    // Only leaving that effective role means the story resolved. The engine
    // itself decides whether the destination is the refreshed event paper,
    // the successive election paper, the desk, or an ending.
    if (routePhase(game.effectiveRole) !== 'eventPage') game.checkpointAutosave();
  }

  function runNewsEngine(fn: () => void): boolean {
    try {
      fn();
      return true;
    } catch (err) {
      console.error('[desk] engine action failed:', err);
      // News errors must not use recoverToIdle(): doing so would visually
      // unlock the desk while the engine remains on a newspaper/event frame.
      phase.value = game.frame ? routePhase(game.effectiveRole) : 'boot';
      nudge('desk.toast.engineError');
      return false;
    }
  }

  // Toast auto-dismiss deliberately does NOT scale with the animations
  // toggle: a toast is information delivery, not motion — with animations
  // off an animMs-scaled delay would set + clear it synchronously, making
  // the "hand is full"/"deck empty" feedback invisible in real usage.
  // Single timer, latest wins: every nudge cancels the pending dismiss and
  // restarts the full duration (DELAYS.toast > 0 always, so `after` always
  // schedules and always returns a handle here).
  function nudge(key: string): void {
    toastKey.value = key;
    if (toastTimer !== undefined) clearTimeout(toastTimer);
    toastTimer = after(DELAYS.toast, () => {
      toastTimer = undefined;
      toastKey.value = null;
    });
  }

  return {
    phase,
    flying,
    flyingFrom,
    openCard,
    outTray,
    toastKey,
    achievementToast,
    shakeIdx,
    resolveView,
    deskView,
    animMs,
    syncFromFrame,
    drawFrom,
    playFromHand,
    playPinned,
    pickPaper,
    chooseNewspaperStory,
    chooseEventChoice,
    nudge,
  };
});
