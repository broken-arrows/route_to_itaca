<script setup lang="ts">
// The Desk — static surfaces (Task 6) + motion overlays (Task 7) of the
// Desk UI Phase 2 plan. Assembles InTray/HandCard/ActionsTray/DeskMonth/
// OutTray from the game/desk stores, plus FlyingCard (drawing phase only),
// OpenDossier (dossierOpen/resolving, wrapped in <Transition> for the
// shrink leave when it unmounts), a desk-dim overlay, and Toast. Renders inside
// StageScaler's 1512x860 design-space slot as absolutely positioned
// elements (spec: docs/design/desk_ui_plan.md §3, "Stage scaling").
import { computed } from 'vue';
import { useGameStore } from '../stores/game';
import { useDeskStore } from '../stores/desk';
import type { CardView } from '../engine/types';
import InTray from '../components/desk/InTray.vue';
import HandCard from '../components/desk/HandCard.vue';
import ActionsTray from '../components/desk/ActionsTray.vue';
import DeskMonth from '../components/desk/DeskMonth.vue';
import OutTray from '../components/desk/OutTray.vue';
import FlyingCard from '../components/desk/FlyingCard.vue';
import OpenDossier from '../components/desk/OpenDossier.vue';
import ClipboardFrame from '../components/brief/ClipboardFrame.vue';
import Prose from '../components/Prose.vue';

const gameStore = useGameStore();
const deskStore = useDeskStore();

const isIdle = computed(() => deskStore.phase === 'idle');

const deskMonth = computed(() => (typeof gameStore.q.month === 'number' ? gameStore.q.month : null));
const deskYear = computed(() => (typeof gameStore.q.year === 'number' ? gameStore.q.year : null));

// The desk's furniture comes from the desk store's `deskView`, NOT from the
// live frame: the engine only fills hand/decks/pinned on `is-hand` scenes, so
// the live frame reports all three as [] on a card scene — rendering it
// directly blanked the whole desk (hand, in-trays AND the actions tray, which
// sits outside the dossier) for the entire dossierOpen window. `deskView` is
// live at the desk and a last-known-good snapshot everywhere else; see the
// comment on it in stores/desk.ts.

// Two rows of three, desk-region-relative (canvas handSix pattern, y ×0.935;
// see docs/design/reference/desk-frames.md "The hand"). Rotation jitter is
// part of the design ("loose dossiers"), applied per slot, not per card.
//
// Deviation from the brief's literal sample: the brief's handSlotStyle()
// returns an inline `transform: rotate(...)`, landing on HandCard's root via
// attrs fallthrough. HandCard's OWN scoped CSS already sets `transform` on
// that same root (base `rotate(var(--jitter))` + a `:hover` lift/flatten
// rule) — an inline style attribute always beats a selector-based CSS rule
// regardless of specificity or pseudo-class, so an inline transform here
// would silently freeze the base rotation AND permanently kill the hover
// lift (the `:hover` rule can never win against it). This is exactly the
// rotation-conflict contingency the brief's Step 5 note calls out. Fix used:
// pass the slot rotation as a `--slot-rot` custom property instead (custom
// properties don't collide with HandCard's own `--jitter`/`--card-bg`/
// `--card-bd` inline vars — Vue merges distinct style keys, it does not let
// one clobber the other), and HandCard's stylesheet now reads
// `rotate(var(--slot-rot, var(--jitter)))`, falling back to its own per-card
// jitter when no slot rotation is supplied (e.g. if it's ever mounted
// standalone, as the component tests do).
// x carries the hand AREA's own left origin (240 — desk-frames §3 "Hand
// area left:240") added to the canvas handSix column pattern: the raw
// canvas xs (180/392/604) are measured from a different container origin,
// and used desk-region-relative they park the left column at x180 — on top
// of the tray column, which ends at x=26+186=212 (confirmed live
// 2026-07-20: a drawn card covered the Party tray and its DRAW chip).
const HAND_SLOTS = [
  { left: 240, top: 131, rot: -1.6 },
  { left: 452, top: 120, rot: 1.4 },
  { left: 664, top: 137, rot: -0.8 },
  { left: 240, top: 415, rot: 1.2 },
  { left: 452, top: 426, rot: -1.3 },
  { left: 664, top: 411, rot: 1.8 },
];
function handSlotStyle(index: number): Record<string, string> {
  const slot = HAND_SLOTS[index % HAND_SLOTS.length];
  return { left: `${slot.left}px`, top: `${slot.top}px`, '--slot-rot': `${slot.rot}deg` };
}

function onDraw(deckId: string): void {
  deskStore.drawFrom(deckId);
}
function onPlayHand(card: CardView): void {
  deskStore.playFromHand(card);
}
function onPlayPinned(card: CardView): void {
  deskStore.playPinned(card);
}

// Per NOTES motion sequence #2 ("played hand card gets opacity:0; other
// hand cards dim to .45"): HandCard exposes one `dimmed` boolean (0.45
// opacity, no true hidden mode), so both conditions collapse to the same
// visual — named explicitly so the openCard-specific rule isn't silently
// lost if the blanket "any non-idle phase dims everything" rule is ever
// narrowed later (e.g. once drawing stops dimming siblings).
function cardDimmed(card: CardView): boolean {
  return !isIdle.value || card.id === deskStore.openCard?.id;
}

// The desk scene's own prose (spec §5.1 regression: DeskView never rendered
// frame.html at all). Reads the STORE's snapshot (deskStore.deskView.html),
// not the live frame — same continuity duty as the rest of the furniture, so
// the note keeps showing the DESK's prose while a dossier is open, not the
// open card's.
//
// EXTRACT, don't strip (Task 7, Wave 2 — user ruling 2026-07-19, reversing
// fix round 1 below): dendry's paragraph buffer only clears on
// `new-page: true`, and the desk hub scenes don't set it, so on the standard
// monthly path frame.html carries a leftover
// `<h1>[+ month : month +] [+ year +][? if rubicon:, Week [+ week +]?]</h1>`
// from the PREVIOUS page (post_event.scene.dry) ahead of the desk's own
// prose. That h1 IS the desk's month+year title — translatable, and it
// carries the Rubicon week suffix when the scene sets it — so it is handed
// to DeskMonth to render instead of DeskView hardcoding Q.month/Q.year (see
// deskTitleHtml below and DeskMonth.vue). Fix round 1 (LEARNINGS.md
// 2026-07-19) discarded this same heading outright, reasoning DeskMonth
// "already prints the month/year" from Q — true then, but that made the
// hardcoded Q read the source of truth instead of the content, backwards per
// this repo's own `source/` = the game, UI = game-agnostic rule. Repeated
// leading h1s: the first one wins as the title (same as fix round 1), the
// rest are dropped. A heading that appears mid-prose (not leading) is left
// alone in the note body, exactly as before.
const LEADING_H1 = /^\s*<h1(?:\s[^>]*)?>([\s\S]*?)<\/h1>/i;

// Empty <p></p> runs (suppressed conditionals) can precede the heading —
// and the note's own first line — on the standard monthly path; strip them
// so the h1 extraction still fires and the note doesn't open with a blank
// typed line.
const EMPTY_LEAD = /^(?:\s*<p>\s*<\/p>)+\s*/i;

const deskTitleAndBody = computed(() => {
  let html = deskStore.deskView.html.replace(EMPTY_LEAD, '');
  let titleHtml: string | null = null;
  const first = LEADING_H1.exec(html);
  if (first) {
    titleHtml = first[1];
    html = html.slice(first[0].length);
    html = html.replace(EMPTY_LEAD, '');
    while (LEADING_H1.test(html)) {
      html = html.replace(LEADING_H1, '').replace(EMPTY_LEAD, '');
    }
  }
  return { titleHtml, bodyHtml: html };
});

// Fed to DeskMonth as its `titleHtml` prop; null (no leading h1 — the boot
// case, before any post_event has run) falls back to DeskMonth's own
// Q-based month+year rendering.
const deskTitleHtml = computed(() => deskTitleAndBody.value.titleHtml);

const deskNoteHtml = computed(() => {
  const html = deskTitleAndBody.value.bodyHtml;
  // Strip tags to decide emptiness: a run of empty <p></p> from suppressed
  // conditionals (or html that was ONLY the extracted heading) must not
  // render an empty paper scrap.
  return html && html.replace(/<[^>]*>/g, '').trim() ? html : null;
});
</script>

<template>
  <div v-if="gameStore.frame" class="desk-view">
    <ClipboardFrame />
    <div class="desk-region">
      <DeskMonth class="pos-month" :month="deskMonth" :year="deskYear" :title-html="deskTitleHtml" />

      <!-- The desk scene's own prose, as a typed note (Task 4; spec §5.1 —
           DeskView never rendered frame.html at all). Placement deviates
           from the plan's "near the month" wording: see DeskView.vue's
           .desk-note comment below and the task report for the reason. -->
      <div v-if="deskNoteHtml" class="desk-note" data-test="desk-note">
        <Prose :html="deskNoteHtml" />
      </div>

      <!-- Tray set = the desk scene's own deck options: order is option order,
           caption is the deck's translatable title (rendered inside InTray),
           paper skin is the deck scene's compiled deck-* role. Deck ids here are
           COMPILED ids (dendry prefixes sections with the file id: `main.cat_gov`);
           the real-game mount in tests/integration.desk-loop.test.ts is the guard
           that keeps this honest — do not weaken it. -->
      <div class="pos-trays">
        <div v-for="deck in deskStore.deskView.decks" :key="deck.id" class="tray-slot">
          <InTray :deck="deck" :disabled="!isIdle" @draw="onDraw" />
        </div>
      </div>

      <HandCard
        v-for="(card, i) in deskStore.deskView.hand"
        :key="card.id"
        class="hand-slot"
        :style="handSlotStyle(i)"
        :card="card"
        :index="i"
        :dimmed="cardDimmed(card)"
        @play="onPlayHand"
      />

      <ActionsTray class="pos-actions" :pinned="deskStore.deskView.pinned" :disabled="!isIdle" @play="onPlayPinned" />

      <OutTray class="pos-out" :entry="deskStore.outTray" />

      <!-- Dim rule (user, 2026-07-19): the dim overlay lives INSIDE
           .desk-region so the Brief is never dimmed or blocked. The leave
           fade shares the cancel duration so the desk doesn't snap bright
           while the dossier is still shrinking away (it also softens the
           resolve commit, where the same v-if drops). -->
      <Transition name="dim">
        <div
          v-if="deskStore.phase === 'dossierOpen' || deskStore.phase === 'resolving'"
          class="desk-dim"
          data-test="desk-dim"
          :style="{ '--dim-ms': `${deskStore.animMs('cancel')}ms` }"
        ></div>
      </Transition>

      <FlyingCard v-if="deskStore.phase === 'drawing' && deskStore.flying" :card="deskStore.flying" />

      <Transition name="dossier">
        <OpenDossier v-if="deskStore.phase === 'dossierOpen' || deskStore.phase === 'resolving'" />
      </Transition>
    </div>
  </div>
</template>

<style scoped>
.desk-view {
  position: relative;
  width: 100%;
  height: 100%;
}
/* Right two-thirds: x 474 -> 1512. Desk surface texture per desk-frames.md §1. */
.desk-region {
  position: absolute;
  left: 474px;
  top: 0;
  right: 0;
  bottom: 0;
  overflow: hidden;
  background:
    repeating-linear-gradient(0deg, rgba(90, 70, 40, 0.028) 0 1px, transparent 1px 6px),
    radial-gradient(120% 110% at 50% 0%, #e2d9c4 0%, #dad0b8 60%, #d2c7ac 100%);
}
.pos-month { position: absolute; left: 36px; top: 24px; }
/* Placement deviation (record for the user's visual pass): the amended plan
   says "near the month", but the month's left strip is occupied by the
   trays column (left:26, top:123, 186 wide, ending ~y763) — bottom-left is
   the closest strip of desk guaranteed free (hand columns start at x180 but
   rows end ~y665; the out tray sits bottom-RIGHT). The canvas itself
   scatters its typed/hand-written notes into free desk corners. */
.desk-note {
  position: absolute;
  left: 26px;
  bottom: 24px;
  width: 186px;
  max-height: 180px;
  overflow-y: auto;
  /* The canvas's typed notes are scraps of paper — no scrollbar chrome.
     Long prose still scrolls (wheel/keys); the chrome would break the
     diegesis more than the hidden affordance costs on an interim slot. */
  scrollbar-width: none;
  background: #fdfcf8;
  border: 1px solid #e2dcc9;
  box-shadow: 0 3px 8px rgba(60, 45, 20, 0.14);
  transform: rotate(-1deg);
  padding: 8px 12px;
  font-family: var(--font-typed);
  font-size: 11px;
  line-height: 1.45;
  color: #6b655a;
  z-index: 2;
}
.desk-note::-webkit-scrollbar {
  display: none;
}
.pos-trays {
  position: absolute;
  left: 26px;
  top: 123px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.tray-slot { display: flex; flex-direction: column; gap: 6px; }
.hand-slot { position: absolute; }
.pos-actions { position: absolute; top: 43px; right: 16px; }
.pos-out { position: absolute; bottom: 24px; right: 28px; }
.desk-dim {
  position: absolute;
  inset: 0;
  background: rgba(28, 26, 21, 0.35);
  z-index: 30;
  pointer-events: none;
}
.dim-leave-active {
  transition: opacity var(--dim-ms, 420ms) ease;
}
.dim-leave-to {
  opacity: 0;
}
</style>
