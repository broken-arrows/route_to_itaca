<script setup lang="ts">
// The Desk's own person/party tooltip — the first time it has ever shown one
// (§4 of this task's brief: the Vue app has rendered the glossary as plain
// stripped text since phase 2, so it never had a tooltip at all). Positioned
// with @floating-ui/vue's useFloating (not hand-rolled). Prose.vue owns ONE
// instance and re-points it (via the `anchor`/`term-id` props) at whichever
// `[data-term]` span is currently hovered — same "one shared tooltip" shape
// as the old shell's `_tipEl` (out/html/game.js).
//
// The DATA composed here is the same set `out/html/game.js`'s
// renderTipContent builds — title -> subtitle -> img -> "Leader:" (from
// tooltip.q.ledBy, live Q) -> ideology (tooltip.q.ideology, live Q, or the
// literal string when it isn't a Q key) -> "Allegiance(s):" (from
// gameLib.allegiances[termId](Q), source/lib/allegiances.js) — reproduced
// from that function directly (grepped and read, not guessed), rendered here
// as the Desk's own paper-styled markup rather than copying its HTML.
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { autoUpdate, flip, offset, shift, useFloating } from '@floating-ui/vue';
import { useGameStore } from '../stores/game';
import { useGlossary } from '../glossary/useGlossary';

const props = defineProps<{
  termId: string;
  anchor: HTMLElement | null;
}>();

const { t } = useI18n();
const game = useGameStore();
const { termFor, colourValue, allegiancesFor } = useGlossary();

const reference = ref<HTMLElement | null>(null);
const floating = ref<HTMLElement | null>(null);
watch(() => props.anchor, (el) => { reference.value = el; }, { immediate: true });

const { floatingStyles } = useFloating(reference, floating, {
  placement: 'top',
  whileElementsMounted: autoUpdate,
  middleware: [offset(10), flip(), shift({ padding: 8 })],
});

const term = computed(() => termFor(props.termId));
const tooltip = computed(() => term.value?.tooltip);

const imgBroken = ref(false);
watch(() => props.termId, () => { imgBroken.value = false; });
// Art paths in compiled content are relative to the RENDERING UI's web root
// (see docs/design/LEARNINGS.md 2026-07-13 "no art rendered" finding) — same
// convention HandCard.vue's imgSrc uses. Computed here, not inline in the
// template: `import.meta` cannot appear inside a compiled template expression.
const imgSrc = computed(() =>
  tooltip.value?.img ? `${import.meta.env.BASE_URL}${tooltip.value.img}` : null,
);

// "Leader: <live Q value>" ONLY when that Q key actually exists right now;
// otherwise fall back to the term's own static infoDesc (a DIFFERENT field —
// this is NOT "show ledBy literally", matching renderTipContent exactly).
const ledByText = computed<string | undefined>(() => {
  const key = tooltip.value?.q?.ledBy;
  if (key && Object.prototype.hasOwnProperty.call(game.q, key)) return String(game.q[key]);
  return undefined;
});
const infoDesc = computed(() => tooltip.value?.infoDesc);

// Ideology: a live Q value when tooltip.q.ideology names one, else the
// configured string itself IS the display text (e.g. a literal ideology
// description rather than a Q-key indirection — both shapes are real in
// source/data/glossary.json).
const ideologyText = computed<string | undefined>(() => {
  const key = tooltip.value?.q?.ideology;
  if (!key) return undefined;
  return Object.prototype.hasOwnProperty.call(game.q, key) ? String(game.q[key]) : key;
});

const allegiances = computed(() => allegiancesFor(props.termId));
const allegianceLabel = computed(() =>
  allegiances.value.length > 1 ? t('desk.glossary.allegiances') : t('desk.glossary.allegiance'),
);
</script>

<template>
  <Teleport to="body">
    <div
      v-if="tooltip"
      ref="floating"
      :style="floatingStyles"
      class="glossary-popover"
      role="tooltip"
      data-test="glossary-popover"
    >
      <img
        v-if="imgSrc && !imgBroken"
        :src="imgSrc"
        :alt="`${tooltip.title} image`"
        class="popover-img"
        @error="imgBroken = true"
      />
      <div class="popover-text">
        <div class="popover-title">{{ tooltip.title }}</div>
        <div v-if="tooltip.subtitle" class="popover-subtitle">{{ tooltip.subtitle }}</div>
        <div v-if="ledByText" class="popover-line">
          {{ t('desk.glossary.leader') }} <span class="popover-ledby">{{ ledByText }}</span>
        </div>
        <div v-else-if="infoDesc" class="popover-line popover-infodesc">{{ infoDesc }}</div>
        <div v-if="ideologyText" class="popover-line popover-ideology">{{ ideologyText }}</div>
        <div v-if="allegiances.length" class="popover-line popover-allegiances" data-test="popover-allegiances">
          {{ allegianceLabel }}
          <template v-for="(a, i) in allegiances" :key="i">
            <span :style="{ color: colourValue(a.colour) }">{{ a.label }}</span><span
              v-if="a.note"
            > ({{ a.note }})</span><span v-if="i < allegiances.length - 1">, </span>
          </template>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.glossary-popover {
  z-index: 1000;
  width: 260px;
  display: flex;
  gap: 10px;
  background: var(--paper-0);
  border: 1px solid var(--ink-0);
  border-radius: 3px;
  padding: 10px 12px;
  box-shadow: 0 10px 24px rgba(46, 42, 34, 0.3);
  font-family: var(--font-body);
  color: var(--ink-0);
  pointer-events: none; /* a hover popover; never eats the pointer */
}
.popover-img {
  flex: none;
  width: 44px;
  height: 44px;
  object-fit: cover;
  border: 1px solid var(--ink-0);
  border-radius: 2px;
}
.popover-text {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  line-height: 1.45;
}
.popover-title {
  font-family: var(--font-news);
  font-size: 14px;
  font-weight: bold;
}
.popover-subtitle {
  font-style: italic;
  opacity: 0.85;
}
.popover-line {
  margin-top: 4px;
}
</style>
