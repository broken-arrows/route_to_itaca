<script setup lang="ts">
// The Desk — static surfaces (Task 6) + motion overlays (Task 7) of the
// Desk UI Phase 2 plan. Assembles InTray/HandCard/ActionsTray/DeskMonth/
// OutTray from the game/desk stores, plus FlyingCard (drawing phase only),
// OpenDossier (dossierOpen/resolving, wrapped in <Transition> for the
// shrink leave when it unmounts), a desk-dim overlay, and Toast. Renders inside
// StageScaler's 1512x860 design-space slot as absolutely positioned
// elements (spec: docs/design/desk_ui_plan.md §3, "Stage scaling").
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../stores/game';
import { useDeskStore } from '../stores/desk';
import type { CardView, DeckView } from '../engine/types';
import InTray from '../components/desk/InTray.vue';
import HandCard from '../components/desk/HandCard.vue';
import ActionsTray from '../components/desk/ActionsTray.vue';
import DeskMonth from '../components/desk/DeskMonth.vue';
import OutTray from '../components/desk/OutTray.vue';
import FlyingCard from '../components/desk/FlyingCard.vue';
import OpenDossier from '../components/desk/OpenDossier.vue';

const { t } = useI18n();
const gameStore = useGameStore();
const deskStore = useDeskStore();

const isIdle = computed(() => deskStore.phase === 'idle');

const deskMonth = computed(() => (typeof gameStore.q.month === 'number' ? gameStore.q.month : null));
const deskYear = computed(() => (typeof gameStore.q.year === 'number' ? gameStore.q.year : null));

// The deck SCENE's own `role` is mechanically always 'deck' (role: deck ->
// isDeck derivation, see compiler.role-derivation.test.ts) — it carries no
// gov/party/parliament distinction. main.scene.dry's hub only ever offers
// THESE known deck ids for the three tray kinds (party_erc/party_cup are
// mutually exclusive via view-if on player_party), so DeskView is the one
// place allowed to know them by id: it supplies the fixed chrome label AND
// re-tags the deck it hands to InTray with the visual card-* role so
// skinFor renders the right paper.
// GAME-SPECIFIC CONFIG: these ids are Route to Ítaca's — to be driven from
// a registry in the later game-agnostic pass, not hardcoded here.
//
// They are the COMPILED scene ids, and dendry prefixes a section scene with its
// FILE id: `@party_erc` inside `main.scene.dry` compiles to `main.party_erc`.
// Writing the bare section name here rendered zero in-trays against the real
// game — the desk was undrawable — while every fixture-based component test
// passed, because the fixtures used the same bare id the code assumed. The
// guard that actually holds this honest is the real-game mount in
// `tests/integration.desk-loop.test.ts`; keep it.
interface TrayKind {
  ids: string[];
  labelKey: string;
  skinRole: 'card-gov' | 'card-party' | 'card-parliament';
}
const TRAY_KINDS: TrayKind[] = [
  { ids: ['main.cat_gov'], labelKey: 'desk.tray.government', skinRole: 'card-gov' },
  {
    ids: ['main.party_erc', 'main.party_cup'],
    labelKey: 'desk.tray.party',
    skinRole: 'card-party',
  },
  { ids: ['main.parlament_deck'], labelKey: 'desk.tray.parliament', skinRole: 'card-parliament' },
];

// The desk's furniture comes from the desk store's `deskView`, NOT from the
// live frame: the engine only fills hand/decks/pinned on `is-hand` scenes, so
// the live frame reports all three as [] on a card scene — rendering it
// directly blanked the whole desk (hand, in-trays AND the actions tray, which
// sits outside the dossier) for the entire dossierOpen window. `deskView` is
// live at the desk and a last-known-good snapshot everywhere else; see the
// comment on it in stores/desk.ts.
const visibleTrays = computed(() => {
  const decks = deskStore.deskView.decks;
  return TRAY_KINDS.flatMap((kind) => {
    const deck = decks.find((d) => kind.ids.includes(d.id));
    if (!deck) return [];
    const skinnedDeck: DeckView = { ...deck, role: kind.skinRole };
    return [{ labelKey: kind.labelKey, deck: skinnedDeck }];
  });
});

// Two rows of three (NOTES "Hand geometry"); maxCards caps the real hand at
// 5-6 so this never needs a third row. Absolute stage-space coordinates,
// clear of the trays column (x: 40-212) and the actions/out column (x from
// ~1262).
const HAND_SLOTS = [
  { left: 440, top: 260 },
  { left: 662, top: 260 },
  { left: 884, top: 260 },
  { left: 440, top: 550 },
  { left: 662, top: 550 },
  { left: 884, top: 550 },
];
function handSlotStyle(index: number): Record<string, string> {
  const slot = HAND_SLOTS[index % HAND_SLOTS.length];
  return { left: `${slot.left}px`, top: `${slot.top}px` };
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
</script>

<template>
  <div v-if="gameStore.frame" class="desk-view">
    <DeskMonth class="pos-month" :month="deskMonth" :year="deskYear" />

    <div class="pos-trays">
      <div v-for="tray in visibleTrays" :key="tray.deck.id" class="tray-slot">
        <p class="tray-kind-label">{{ t(tray.labelKey) }}</p>
        <InTray :deck="tray.deck" :disabled="!isIdle" @draw="onDraw" />
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

    <!-- Brief-side dim is deferred: the Brief panel doesn't exist yet
         (phase 3 of the desk_ui_plan.md build order) — nothing to dim.
         The leave fade shares the cancel duration so the desk doesn't
         snap bright while the dossier is still shrinking away (it also
         softens the resolve commit, where the same v-if drops). -->
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
</template>

<style scoped>
.desk-view {
  position: relative;
  width: 100%;
  height: 100%;
}
.pos-month {
  position: absolute;
  left: 40px;
  top: 24px;
}
.pos-trays {
  position: absolute;
  left: 40px;
  top: 110px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.tray-slot {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.tray-kind-label {
  margin: 0;
  font-family: var(--font-title);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--ink-0);
  opacity: 0.7;
}
.hand-slot {
  position: absolute;
}
.pos-actions {
  position: absolute;
  top: 24px;
  right: 40px;
}
.pos-out {
  position: absolute;
  bottom: 40px;
  right: 40px;
}
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
