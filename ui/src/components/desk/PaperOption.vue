<script setup lang="ts">
// A single option (paper slip) inside an open dossier. Spec:
// prototype-draw-to-dossier-NOTES.md "Open dossier" (paper slips, dead
// options dashed border/dimmed/grey/not-allowed, slight per-option
// rotation, hover translateX(7px)) + "Motion sequences" #6 (shake,
// translateX(0->-7->+7->0), .4s, on the rejected option only).
//
// Cost text is `choice.subtitle` rendered AS-IS — the "[n resources]"
// convention lives in content; no parsing, no cost-coin badge (pinned
// decision, task brief). A generic "locked" chip stands in for the
// prototype's specific "NEEDS .../NEED (n)" lock tags since we deliberately
// don't parse subtitle to know which resource is short.
//
// `pick` only fires when the option is choosable — that's this component's
// own tested, reusable contract. OpenDossier does NOT rely on it to reach
// the store: a locked click must still reach deskStore.pickPaper so the
// store can shake/toast it, so the integration wires a plain click
// listener around this component instead (see OpenDossier.vue).
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import type { ChoiceView } from '../../engine/types';
import { useDeskStore } from '../../stores/desk';
import Prose from '../Prose.vue';

const props = defineProps<{
  choice: ChoiceView;
  index: number;
  shaking: boolean;
}>();
const emit = defineEmits<{ pick: [index: number] }>();

const { t } = useI18n();
const desk = useDeskStore();

const locked = computed(() => !props.choice.canChoose);

// Deterministic slight per-option rotation (NOTES "slight per-option
// rotations") — same seeded-not-random approach as HandCard's jitter, kept
// small since these are paper slips, not whole cards.
const ROTATE_DEG = [-1.1, 0.8, -0.6, 1.2, -0.9, 0.7];
const rotate = computed(() => `${ROTATE_DEG[props.index % ROTATE_DEG.length]}deg`);

// The shake CSS duration mirrors the exact ms the store's own shakeTimer
// uses to clear shakeIdx (animMs('cancel')), so the visual never gets cut
// short by, or outlives, the state that triggered it. With animations off
// this is 0ms, i.e. instant (no visible shake).
const shakeMs = computed(() => `${desk.animMs('cancel')}ms`);

function onClick(): void {
  if (props.choice.canChoose) emit('pick', props.index);
}

// Keyboard activation synthesizes a NATIVE click on the root element so
// Enter/Space take exactly the mouse path: the click bubbles up to
// whatever integration wrapper is listening (OpenDossier's .paper-slot ->
// deskStore.pickPaper, LOCKED activations included, which is what drives
// the shake). Calling onClick()/the emit directly here would leave
// keyboard silently dead for any parent that — like OpenDossier — listens
// on a wrapper element rather than the `pick` emit. No double dispatch:
// a div[role=button] gets no browser-default click from Enter/Space (that
// is <button>-only behaviour), so the synthesized click is the only one.
const rootEl = ref<HTMLElement | null>(null);
function onKeyActivate(): void {
  rootEl.value?.click();
}
</script>

<template>
  <div
    ref="rootEl"
    class="paper-option"
    :class="{ locked, shaking }"
    :style="{ '--rotate': rotate, '--shake-ms': shakeMs }"
    :data-test="`paper-option-${index}`"
    role="button"
    :aria-disabled="locked ? 'true' : 'false'"
    tabindex="0"
    @click="onClick"
    @keydown.enter.prevent="onKeyActivate"
    @keydown.space.prevent="onKeyActivate"
  >
    <!-- Engine output, NOT user input: title/subtitle come out of
         convertLine (_contentToHTML), which emits <em>/<strong> and passes
         `magic` blocks through raw — exactly the same trust boundary as the
         prose, which already renders through <Prose>. Interpolating them
         escaped the markup and showed the player literal tags (e.g.
         root.start's `<span style="font-size: 1.1em;">Start game</span>` on
         the very first screen). convertLine also passes through
         window.displayText per text run (same as the prose paragraphs), so a
         party name in a card title gets the same live colour/tooltip here. -->
    <!-- Caller-owned wrappers, NOT `<Prose class="option-title">`: Prose is
         a multi-root (fragment) component, so this component's scope id
         never lands on it and the scoped .option-title/.option-subtitle
         rules matched nothing (both rendered as inherited 16px body text,
         confirmed live 2026-07-20 — see OpenDossier.vue's cover-prose
         comment for the full mechanism). -->
    <p class="option-title"><Prose tag="span" :html="choice.title" /></p>
    <p v-if="choice.subtitle" class="option-subtitle"><Prose tag="span" :html="choice.subtitle" /></p>
    <span v-if="locked" class="lock-chip">{{ t('desk.dossier.locked') }}</span>
  </div>
</template>

<style scoped>
.paper-option {
  position: relative;
  /* White slip floating on the folder (desk_ui_dossier_open.png): near-white,
     no hard border, soft drop shadow, generous padding, rounded. */
  background: #fdfcfb;
  border: 1px solid rgba(90, 70, 40, 0.1);
  border-radius: 6px;
  padding: 13px 16px;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(60, 45, 20, 0.16);
  transform: rotate(var(--rotate));
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.paper-option:not(.locked):hover {
  transform: translateX(7px) rotate(var(--rotate));
  box-shadow: 0 6px 14px rgba(46, 42, 34, 0.24);
}
.paper-option.locked {
  border-style: dashed;
  background: #f4efdf;
  opacity: 0.72;
  cursor: not-allowed;
}
.paper-option.shaking {
  animation: paper-shake var(--shake-ms) ease;
}
@keyframes paper-shake {
  0% {
    transform: translateX(0) rotate(var(--rotate));
  }
  33% {
    transform: translateX(-7px) rotate(var(--rotate));
  }
  66% {
    transform: translateX(7px) rotate(var(--rotate));
  }
  100% {
    transform: translateX(0) rotate(var(--rotate));
  }
}
.option-title {
  margin: 0;
  font-family: var(--font-title);
  font-size: 14.5px;
  font-weight: 700;
  color: var(--ink-0);
}
.option-subtitle {
  margin: 4px 0 0;
  font-family: var(--font-body);
  font-size: 12px;
  line-height: 1.35;
  color: var(--ink-0);
  opacity: 0.7;
}
.lock-chip {
  display: inline-block;
  margin-top: 6px;
  font-family: var(--font-typed);
  font-size: 10px;
  letter-spacing: 0.06em;
  color: var(--ink-0);
  opacity: 0.6;
  border: 1px solid var(--ink-0);
  border-radius: 2px;
  padding: 1px 6px;
}
</style>
