<script setup lang="ts">
// Month printed on the desk, top-left. Spec: prototype-draw-to-dossier-NOTES.md
// "Other desk objects" — Newsreader small-caps, rotate(-1deg), underline rule.
//
// Task 7 (Wave 2): the title is CONTENT now, not a UI-hardcoded Q read.
// `titleHtml` is DeskView's extraction of the leading <h1> dendry leaves at
// the top of frame.html on the standard monthly path (post_event.scene.dry's
// own heading, `= [+ month : month +] [+ year +][? if rubicon:, Week
// [+ week +]?]` — already translated/formatted by the content, week suffix
// included when the scene sets it). Rendered via <Prose tag="span"> — same
// glossary/insert-safe path every other engine title reaching this app
// already uses — not `{{ }}`. `month`/`year` stay as the Q-based FALLBACK
// for the one window titleHtml is null: boot, before any post_event has run
// (verified live — the desk's first frame has no leading h1 yet).
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import Prose from '../Prose.vue';

const props = withDefaults(
  defineProps<{
    month: number | null;
    year: number | null;
    titleHtml?: string | null;
  }>(),
  { titleHtml: null },
);

const { t } = useI18n();

const monthLabel = computed(() =>
  props.month !== null && props.month >= 1 && props.month <= 12 ? t(`desk.month.${props.month}`) : '',
);
</script>

<template>
  <div class="desk-month">
    <Prose v-if="titleHtml" tag="span" class="month-title" :html="titleHtml" />
    <template v-else>
      <span class="month">{{ monthLabel }}</span>
      <span v-if="year !== null" class="year">{{ year }}</span>
    </template>
  </div>
</template>

<style scoped>
/* desk-frames.md §3 "Month": 800 34px small-caps, letter-spacing .1em,
   rotate(-1deg), fixed 210×2 underline rule below (not a text-width
   border). Face comes from the swappable --font-news token. */
.desk-month {
  display: inline-block;
  font-family: var(--font-news);
  font-variant: small-caps;
  font-weight: 800;
  font-size: 34px;
  letter-spacing: 0.1em;
  color: rgba(78, 62, 36, 0.52);
  transform: rotate(-1deg);
  line-height: 1;
}
.desk-month::after {
  content: '';
  display: block;
  width: 210px;
  height: 2px;
  margin-top: 7px;
  background: rgba(78, 62, 36, 0.3);
}
/* Gap only applies on the Q-fallback path (two spans DeskMonth owns) — the
   titleHtml path is a single Prose span, and being a fragment-root child it
   would not receive this scoped rule anyway (see the Prose wrapper fix in
   OpenDossier.vue). */
.month {
  margin-right: 8px;
}
.year {
  font-size: 22px;
}
</style>
