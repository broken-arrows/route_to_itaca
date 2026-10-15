<script setup lang="ts">
// Tabletop calendar, top-left; Mazius lettering shares the game's identity face.
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

// Style only text nodes; keep authored markup, localized wording and week suffixes.
// The authored year is authoritative even when the fallback props differ.
const calendarTitle = computed(() => props.titleHtml?.split(/(<[^>]*>)/g)
  .map((part) => part.startsWith('<') ? part : part.replace(/\b(\d{4}\b.*)/, '<span class="calendar-year">$1</span>'))
  .join(''));
</script>

<template>
  <div class="desk-month">
    <span class="calendar-rings" aria-hidden="true"></span>
    <div class="calendar-date">
    <Prose v-if="titleHtml" tag="span" :html="calendarTitle ?? ''" />
    <template v-else>
      <span class="month">{{ monthLabel }}</span>
      <span v-if="year !== null" class="calendar-year"> {{ year }}</span>
    </template>
    </div>
  </div>
</template>

<style scoped>
.desk-month {
  font: var(--type-calendar);
  letter-spacing: var(--type-tracking);
  position: relative;
  width: clamp(190px, 16vw, 240px);
  padding: 0;
  text-align: center;
  color: #45433d;
  background: #bdb6a6;
  border: 1px solid #aaa79e;
  border-radius: 3px 3px 1px 1px;
  box-shadow: 0 2px 0 #dedbd2, 0 3px 0 #aaa79e, 0 5px 0 #e8e5dc,
    0 6px 0 #aaa79e, 0 12px 12px rgba(46, 42, 34, .2);
  transform: rotate(-2deg);
  transform-origin: center top;
  isolation: isolate;
}
.desk-month::before {
  content: '';
  position: absolute;
  z-index: 0;
  inset: 5px -5px -7px 5px;
  border: 1px solid #aaa79e;
  border-radius: 2px;
  background: #ccc9bf;
  transform: skewX(-3deg);
}
.calendar-rings::before, .calendar-rings::after {
  content: '';
  position: absolute;
  z-index: 2;
  top: -7px;
  width: 8px;
  height: 19px;
  border: 2px solid #6c6b65;
  border-radius: 5px;
  background: linear-gradient(90deg, #77766f, #e7e5de 50%, #77766f);
  box-shadow: 0 1px 2px rgba(60, 45, 20, .24);
}
.calendar-rings::before { left: 24%; }
.calendar-rings::after { right: 24%; }
.calendar-date {
  position: relative;
  z-index: 1;
  padding: 20px 12px 10px;
  background: linear-gradient(#ded8ca 0 10px, var(--paper-0) 10px);
  border-radius: 2px;
  overflow-wrap: anywhere;
}
.calendar-date :deep(.calendar-year) {
  display: block;
  margin-top: 5px;
  padding-top: 5px;
  border-top: 1px solid #c8beab;
  font: var(--type-heading-compact);
  font-variant-numeric: lining-nums tabular-nums;
  letter-spacing: .18em;
}
</style>
