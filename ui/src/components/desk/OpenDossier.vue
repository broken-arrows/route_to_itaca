<script setup lang="ts">
// The open dossier overlay — a single open FOLDER spread, per
// docs/design/reference/desk_ui_dossier_open.png: both halves are the same
// folder paper (manila for a party card, cream for gov), the left half is the
// cover (typed header row, big serif title, prompt prose, framed poster art),
// the right half holds the option papers as WHITE slips floating on the
// folder. No white "papers panel" with an ink border, no ✕, no instructional
// footer text (user rulings 2026-07-13 / 2026-07-20; the "THE PAPERS INSIDE —
// CHOOSE ONE TO ACT" line and "Banked capital…/Choosing a paper is the turn…"
// captions in the design are filler slots, skipped per plan §2).
//
// There is NO close/✕ affordance (user decision overriding phase-2 spec §6):
// an open dossier is a committed engine action; the only way back to the hand
// is the game's own `easy_discard` paper. See the header in stores/desk.ts.
//
// Every CSS duration routes through desk.animMs() (motion.ts DELAYS:
// dossierOpen/coverSwing/coverSwingDelay/cancel/resolve), so "animations off"
// = 0ms everywhere with ONE timing source.
//
// During 'resolving' the prose/papers render from resolveView (the pick's
// engine call already advanced the frame), not the live frame — rendering
// live would blank the dossier mid-animation.
import { computed, ref } from 'vue';
import { useDeskStore } from '../../stores/desk';
import { useGameStore } from '../../stores/game';
import { skinFor } from './skins';
import PaperOption from './PaperOption.vue';
import Prose from '../Prose.vue';

const props = withDefaults(
  defineProps<{ origin?: { x: number; y: number } }>(),
  { origin: () => ({ x: 0, y: 0 }) },
);

const desk = useDeskStore();
const game = useGameStore();

const skin = computed(() => skinFor(desk.openCard?.role));
const resolving = computed(() => desk.phase === 'resolving');
const choices = computed(() =>
  resolving.value && desk.resolveView ? desk.resolveView.choices : (game.frame?.choices ?? []),
);
// Leading empty <p></p> runs (suppressed conditionals ahead of the card's own
// heading) are stripped so the cover neither shows a blank first line nor
// defeats hasLeadingTitle / the :first-child h1 styling below.
const proseHtml = computed(() => {
  const raw = resolving.value && desk.resolveView ? desk.resolveView.html : (game.frame?.html ?? '');
  return raw.replace(/^(?:\s*<p>\s*<\/p>)+\s*/i, '');
});

const coverArt = computed(() => {
  const img = desk.openCard?.image;
  return img ? `${import.meta.env.BASE_URL}${img}` : null;
});
const artBroken = ref(false);

// Diegetic header labels — constant per skin (document furniture, not i18n).
const header = computed(() => {
  if (skin.value.key === 'gov') return { left: 'PRESIDÈNCIA · GOV-04', right: 'GENERALITAT DE CATALUNYA' };
  if (skin.value.key === 'party') return { left: 'DOSSIER DE PARTIT', right: 'MANIOBRA' };
  if (skin.value.key === 'parliament') return { left: 'ORDRE DEL DIA', right: 'PARLAMENT' };
  return null;
});

// Task 7 (Wave 2): 110 scene files lead their content with a `=` heading —
// dendry compiles that to a leading <h1>. Gate the separate cover-title
// element on whether the SAME prose the template renders (proseHtml) leads
// with one, so the two never stack.
const hasLeadingTitle = computed(() => /^\s*<h1[\s>]/i.test(proseHtml.value));

const styleVars = computed(() => ({
  '--card-bd': skin.value.bd,
  '--scale-ms': `${desk.animMs('dossierOpen')}ms`,
  '--swing-ms': `${desk.animMs('coverSwing')}ms`,
  '--swing-delay': `${desk.animMs('coverSwingDelay')}ms`,
  '--cancel-ms': `${desk.animMs('cancel')}ms`,
  '--resolve-ms': `${desk.animMs('resolve')}ms`,
  '--origin-x': `${props.origin.x}px`,
  '--origin-y': `${props.origin.y}px`,
}));

// PaperOption's `pick` emit only fires for a choosable option; a locked click
// must still reach pickPaper so the store can shake/toast it, so every slot is
// wrapped with a plain click listener that always calls the store (pickPaper
// re-derives canChoose from the live frame, so calling it is safe/idempotent).
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
      <div v-if="header" class="dossier-header" aria-hidden="true">
        <span class="hd-left">
          <span v-if="skin.key === 'gov'" class="crest"></span>{{ header.left }}
        </span>
        <span class="hd-right">{{ header.right }}</span>
      </div>
      <span v-if="skin.key === 'gov'" class="cover-rule" aria-hidden="true"></span>

      <!-- desk.openCard.title reaches here already glossary-marked (see the
           OutTray/ActionsTray comments); render via <Prose tag="span"> not
           {{ }}. Wrapped in an <h2> the cover owns, gated on the prose not
           already leading with its own <h1>. -->
      <h2 v-if="!hasLeadingTitle" class="cover-title"><Prose tag="span" :html="desk.openCard?.title ?? ''" /></h2>

      <!-- Caller-owned wrapper, NOT `<Prose class="cover-prose">`: Prose is a
           multi-root component, so a scoped .cover-prose rule never lands on
           its root (LEARNINGS 2026-07-20). -->
      <div class="cover-prose">
        <Prose :html="proseHtml" />
      </div>

      <div v-if="coverArt && !artBroken" class="cover-art">
        <img :src="coverArt" alt="" @error="artBroken = true" />
      </div>
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
  left: 47%;
  top: 50%;
  width: clamp(760px, 64vw, 1400px);
  max-width: calc(100% - clamp(44px, 6vw, 96px));
  height: min(820px, calc(100% - clamp(56px, 10vh, 120px)));
  display: flex;
  z-index: 50;
  border: 1px solid var(--card-bd);
  border-radius: 5px;
  filter: drop-shadow(0 24px 40px rgba(60, 40, 20, 0.3));
  transform-origin: center;
  perspective: 1400px;
  transform-style: preserve-3d;
  will-change: transform, opacity;
  animation: dossier-open var(--scale-ms) cubic-bezier(0.22, 0.72, 0.24, 1) both;
}
/* The folder paper — both halves. Lighter than the hand-card manila, to match
   the open-spread reference. */
.skin-gov { background: #f7f4ec; }
.skin-party { background: #e9dcae; }
.skin-parliament { background: #eef0f2; }
.skin-neutral { background: #f6f4ec; }

@keyframes dossier-open {
  from {
    transform:
      translate(calc(-50% + var(--origin-x)), calc(-50% + var(--origin-y)))
      scale(0.22)
      rotate(-1.5deg);
    opacity: 0.25;
  }
  to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
}
.open-dossier.resolving {
  animation: dossier-resolve var(--resolve-ms) ease forwards;
}
@keyframes dossier-resolve {
  to { transform: translate(-12%, -8%) scale(0.16) rotate(-4deg); opacity: 0; }
}
/* Unmount leave (via <Transition name="dossier"> in DeskView). */
.dossier-leave-active {
  transition: transform var(--cancel-ms) ease, opacity var(--cancel-ms) ease;
}
.dossier-leave-to {
  transform: translate(-85%, -15%) scale(0.2);
  opacity: 0;
}

.cover {
  flex: 0 0 46%;
  padding: 26px 26px 22px;
  display: flex;
  flex-direction: column;
  transform-origin: right center;
  backface-visibility: hidden;
  transform-style: preserve-3d;
  animation: cover-swing var(--swing-ms) cubic-bezier(0.22, 0.72, 0.24, 1) var(--swing-delay) both;
}
@keyframes cover-swing {
  from { transform: rotateY(-28deg); }
  to { transform: rotateY(0deg); }
}
.dossier-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  font-family: var(--font-typed);
  font-size: 9px;
  letter-spacing: 0.08em;
  color: #8a7a58;
}
.hd-left { display: inline-flex; align-items: center; gap: 6px; }
.crest {
  width: 14px;
  height: 17px;
  background: repeating-linear-gradient(180deg, #f4c318 0 2px, #d4232a 2px 4px);
  border: 1px solid #b8901a;
  border-radius: 2px 2px 6px 6px / 2px 2px 9px 9px;
}
/* Ministry red double rule — the ratified red exception (--paper-rule-ink). */
.cover-rule {
  display: block;
  margin: 10px 0 4px;
  border-bottom: 3px double var(--paper-rule-ink);
}
.cover-title {
  font-family: var(--font-news);
  font-size: 27px;
  font-weight: 700;
  margin: 10px 0 10px;
  color: var(--ink-0);
}
.cover-prose {
  font-family: var(--font-body);
  font-size: 13.5px;
  line-height: 1.5;
  color: var(--ink-0);
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #c9c0aa transparent;
}
/* When the prose leads with its own <h1>, it replaces .cover-title — give it
   the same treatment. :deep() because the h1 arrives via v-html inside Prose. */
.cover-prose :deep(h1:first-child) {
  font-family: var(--font-news);
  font-size: 27px;
  font-weight: 700;
  margin: 10px 0 10px;
  color: var(--ink-0);
}
/* Framed poster art on the cover — pasted photo, white border + shadow,
   slight rotation (matches the hand card's framed party art). */
.cover-art {
  margin-top: 14px;
  align-self: flex-start;
  max-width: 90%;
  border: 5px solid #fdfcf8;
  box-shadow: 0 4px 10px rgba(46, 42, 34, 0.28);
  transform: rotate(-1deg);
}
.cover-art img {
  display: block;
  width: 100%;
  height: auto;
  max-height: 200px;
  object-fit: cover;
}

.papers {
  flex: 1;
  padding: 26px 26px 22px;
  display: flex;
  flex-direction: column;
  /* Center fold line between the two halves of the open folder. */
  border-left: 1px solid rgba(90, 70, 40, 0.18);
}
.papers-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
  /* Room for the slips' rotation + hover-slide so no horizontal scrollbar
     appears (CSS overflow coupling — see LEARNINGS 2026-07-19). */
  padding: 2px 10px;
  scrollbar-width: thin;
  scrollbar-color: #c9c0aa transparent;
}
.paper-slot {
  cursor: pointer;
}
</style>
