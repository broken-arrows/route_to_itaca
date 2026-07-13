<script setup lang="ts">
// A drawn, face-up hand card. Spec: prototype-draw-to-dossier-NOTES.md
// "Hand geometry" (186x250, two rows of three, +/-2deg jitter, hover lift)
// + "Card anatomy" (title block "ASSUMPTE: <title>"; gov = seal + red
// double rule ministry stationery). The paper-tab file-ref is NOT
// reproduced (no per-card data to source a file-ref code from — inventing
// a numbering scheme would be filler). Positioning (the two-rows-of-three
// grid) is DeskView's job — this component only owns its own +/-2deg
// jitter rotation (seeded from `index`) and the hover-lift interaction.
import { computed, ref } from 'vue';
import type { CardView } from '../../engine/types';
import { skinFor } from './skins';

const props = defineProps<{
  card: CardView;
  index: number;
  dimmed?: boolean;
}>();
const emit = defineEmits<{ play: [card: CardView] }>();

// Deterministic +/-2deg jitter table (NOT random — keeps renders stable and
// the component pure/testable). Six entries cover the two-rows-of-three cap.
const JITTER_DEG = [-2, 1.4, -1.6, 2, -1.2, 1.8];

const skin = computed(() => skinFor(props.card.role));
const jitter = computed(() => `${JITTER_DEG[props.index % JITTER_DEG.length]}deg`);

const imgBroken = ref(false);
const imgSrc = computed(() =>
  props.card.image ? `${import.meta.env.BASE_URL}${props.card.image}` : null,
);

function onClick(): void {
  emit('play', props.card);
}
</script>

<template>
  <div
    class="hand-card"
    :class="[`skin-${skin.key}`, { dimmed }]"
    :style="{ '--card-bg': skin.bg, '--card-bd': skin.bd, '--jitter': jitter }"
    :data-test="`hand-card-${card.id}`"
    role="button"
    tabindex="0"
    @click="onClick"
    @keydown.enter.prevent="onClick"
    @keydown.space.prevent="onClick"
  >
    <span v-if="skin.key === 'parlament'" class="rule-accent" aria-hidden="true"></span>
    <div class="card-art">
      <img v-if="imgSrc && !imgBroken" :src="imgSrc" alt="" @error="imgBroken = true" />
      <div v-else class="art-placeholder" aria-hidden="true"></div>
    </div>
    <span v-if="skin.key === 'gov'" class="gov-rule" aria-hidden="true"></span>
    <div class="card-body">
      <!-- "ASSUMPTE:" is diegetic stationery text, not UI chrome —
           intentionally not i18n: in-world Catalan document furniture that
           stays Catalan whatever the UI language is. -->
      <p class="card-title"><span class="assumpte">ASSUMPTE:</span> {{ card.title }}</p>
    </div>
    <span v-if="skin.key === 'party'" class="tie-accent" aria-hidden="true"></span>
  </div>
</template>

<style scoped>
.hand-card {
  width: 186px;
  height: 250px;
  background: var(--card-bg);
  border: 1px solid var(--card-bd);
  border-radius: 4px;
  box-shadow: 0 6px 14px rgba(46, 42, 34, 0.3);
  display: flex;
  flex-direction: column;
  cursor: pointer;
  position: relative;
  transform: rotate(var(--jitter));
  transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.2s ease;
}
.hand-card:hover {
  transform: translateY(-12px) rotate(0deg);
  box-shadow: 0 14px 24px rgba(46, 42, 34, 0.35);
  z-index: 5;
}
.hand-card.dimmed {
  opacity: 0.45;
}
.rule-accent {
  height: 4px;
  background: var(--accent-red);
  border-radius: 4px 4px 0 0;
}
.card-art {
  flex: 1;
  margin: 10px 10px 0;
  border-radius: 2px;
  overflow: hidden;
  background: var(--paper-3);
}
.card-art img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.art-placeholder {
  width: 100%;
  height: 100%;
  background-image: repeating-linear-gradient(
    45deg,
    var(--paper-3),
    var(--paper-3) 8px,
    var(--paper-1) 8px,
    var(--paper-1) 16px
  );
}
/* Ministry stationery frame (desk plan §5.3: "seal, department line, red
   double rule") — part of the gov paper's own anatomy, distinct from the
   red-as-signal reservation which governs hovers/bubbles/event chrome.
   --paper-rule-ink is the token that DECLARES that exception; see tokens.css.
   Never inline the literal here again — an off-token red is unauditable. */
.gov-rule {
  margin: 8px 10px 0;
  border-bottom: 3px double var(--paper-rule-ink);
}
.card-body {
  padding: 8px 10px 12px;
}
.card-title {
  margin: 0;
  font-family: var(--font-title);
  font-size: 13px;
  line-height: 1.25;
  color: var(--ink-0);
}
.assumpte {
  font-family: var(--font-typed);
  font-size: 10px;
  letter-spacing: 0.06em;
  opacity: 0.6;
}
.tie-accent {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--ink-0);
  opacity: 0.35;
}
</style>
