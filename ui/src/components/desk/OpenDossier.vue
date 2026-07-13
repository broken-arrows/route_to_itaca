<script setup lang="ts">
// The open dossier overlay. Spec: prototype-draw-to-dossier-NOTES.md "Open
// dossier" (left flap = cover: skin header, big title, prompt prose; right
// side = papers column) + "Motion sequences" #2 (scale-up .55s + cover
// swing .75s/.28s delay, on mount) and #3 (resolve fly-out, driven by the
// store's own animMs('resolve') timer since this component stays mounted
// for the whole 'resolving' window — DeskView's v-if covers both
// 'dossierOpen' and 'resolving'). No props: reads openCard/phase/frame
// directly, per the brief's contract ("OpenDossier (no props — reads
// stores)").
//
// There is NO close/✕ affordance (2026-07-13, user decision overriding the
// phase-2 spec §6 cancel contract): an open dossier is a committed engine
// action, and the only way back to the hand is the game's own
// `easy_discard` paper — which is difficulty-gated and costs a month
// action. See the header comment in stores/desk.ts.
//
// Every CSS duration routes through desk.animMs() (keys in motion.ts
// DELAYS: dossierOpen/coverSwing/coverSwingDelay/cancel/resolve), so
// "animations off" = 0ms everywhere with ONE timing source.
//
// During 'resolving' the prose/papers render from the store's resolveView
// snapshot, NOT the live frame: the pick's engine call has already
// advanced the frame to the destination desk (an isHand scene with no
// choices) before the fly-out starts — rendering live would blank the
// dossier mid-animation (review fix round, Critical).
//
// Prose safety: v-html only ever receives frame.html content — either the
// live game.frame.html or the store's verbatim pre-pick snapshot of it
// (engine contentToHTML output, same trust level as phase 1's debug page).
import { computed } from 'vue';
import { useDeskStore } from '../../stores/desk';
import { useGameStore } from '../../stores/game';
import { skinFor } from './skins';
import PaperOption from './PaperOption.vue';

const desk = useDeskStore();
const game = useGameStore();

const skin = computed(() => skinFor(desk.openCard?.role));
const resolving = computed(() => desk.phase === 'resolving');
const choices = computed(() =>
  resolving.value && desk.resolveView ? desk.resolveView.choices : (game.frame?.choices ?? []),
);
const proseHtml = computed(() =>
  resolving.value && desk.resolveView ? desk.resolveView.html : (game.frame?.html ?? ''),
);

const styleVars = computed(() => ({
  '--cover-bg': skin.value.bg,
  '--cover-bd': skin.value.bd,
  '--scale-ms': `${desk.animMs('dossierOpen')}ms`,
  '--swing-ms': `${desk.animMs('coverSwing')}ms`,
  '--swing-delay': `${desk.animMs('coverSwingDelay')}ms`,
  '--cancel-ms': `${desk.animMs('cancel')}ms`,
  '--resolve-ms': `${desk.animMs('resolve')}ms`,
}));

// PaperOption's own `pick` emit only fires for a choosable option (its own
// tested contract) — a locked click must still reach pickPaper so the
// store can shake/toast it, so every paper slot is wrapped with a plain
// click listener that always calls the store (pickPaper re-derives
// canChoose from the live frame itself, so calling it unconditionally is
// safe/idempotent).
function onPick(i: number): void {
  desk.pickPaper(i);
}
</script>

<template>
  <div
    class="open-dossier"
    :class="[`skin-${skin.key}`, { resolving }]"
    :style="styleVars"
    data-test="open-dossier"
  >
    <div class="cover">
      <span v-if="skin.key === 'gov'" class="cover-seal" aria-hidden="true"></span>
      <span v-if="skin.key === 'gov'" class="cover-rule" aria-hidden="true"></span>
      <span v-if="skin.key === 'party'" class="cover-tie" aria-hidden="true"></span>
      <h2 class="cover-title">{{ desk.openCard?.title }}</h2>
      <div class="cover-prose" v-html="proseHtml"></div>
    </div>
    <div class="papers">
      <div class="papers-list">
        <div
          v-for="(choice, i) in choices"
          :key="choice.id"
          class="paper-slot"
          @click="onPick(i)"
        >
          <PaperOption :choice="choice" :index="i" :shaking="desk.shakeIdx === i" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.open-dossier {
  position: absolute;
  left: 60px;
  top: 100px;
  width: 850px;
  height: 580px;
  display: flex;
  z-index: 50;
  filter: drop-shadow(0 24px 40px rgba(60, 40, 20, 0.3));
  transform-origin: 15% 85%;
  animation: dossier-open var(--scale-ms) cubic-bezier(0.2, 0.8, 0.3, 1) both;
}
@keyframes dossier-open {
  from {
    transform: scale(0.22);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}
.open-dossier.resolving {
  animation: dossier-resolve var(--resolve-ms) ease forwards;
}
@keyframes dossier-resolve {
  to {
    transform: translate(300px, 260px) scale(0.16) rotate(-4deg);
    opacity: 0;
  }
}
/* Unmount leave: driven by <Transition name="dossier"> in DeskView. Vue
   applies these class names to THIS component's root element even though
   the <Transition> wrapper lives in the parent — scoped CSS still matches
   because Vue stamps the scoping data attribute on this root regardless of
   who added the class token. With the ✕ gone this only ever runs at the end
   of a resolve (the fly-out has already faded the dossier to opacity 0), so
   it is a safety net for the removal, not a beat of its own. */
.dossier-leave-active {
  transition: transform var(--cancel-ms) ease, opacity var(--cancel-ms) ease;
}
.dossier-leave-to {
  transform: scale(0.2) translate(-200px, 200px);
  opacity: 0;
}
.cover {
  position: relative;
  flex: 0 0 320px;
  background: var(--cover-bg);
  border: 1px solid var(--cover-bd);
  border-radius: 4px 0 0 4px;
  padding: 24px 20px;
  display: flex;
  flex-direction: column;
  transform-origin: left center;
  animation: cover-swing var(--swing-ms) cubic-bezier(0.3, 0.9, 0.35, 1) var(--swing-delay) both;
}
@keyframes cover-swing {
  from {
    transform: rotateY(-88deg);
  }
  to {
    transform: rotateY(7deg);
  }
}
.cover-seal {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 3px solid var(--ink-0);
  opacity: 0.18;
}
/* Ministry stationery frame (matches HandCard's gov-rule) — paper anatomy,
   not the red-as-signal reservation. The exception is declared by the
   --paper-rule-ink token in tokens.css; keep it off-literal. */
.cover-rule {
  display: block;
  margin: 10px 0;
  border-bottom: 3px double var(--paper-rule-ink);
}
.cover-tie {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--ink-0);
  opacity: 0.35;
}
.cover-title {
  font-family: var(--font-news);
  font-size: 26px;
  margin: 8px 0 12px;
  color: var(--ink-0);
}
.cover-prose {
  flex: 1;
  font-family: var(--font-body);
  font-size: 13px;
  line-height: 1.5;
  color: var(--ink-0);
  overflow-y: auto;
}
.papers {
  flex: 1;
  background: var(--paper-0);
  border: 1px solid var(--ink-0);
  border-radius: 0 4px 4px 0;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.papers-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
}
.paper-slot {
  cursor: pointer;
}
</style>
