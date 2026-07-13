<script setup lang="ts">
// Month printed on the desk, top-left. Spec: prototype-draw-to-dossier-NOTES.md
// "Other desk objects" — Newsreader small-caps, rotate(-1deg), underline rule.
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  month: number | null;
  year: number | null;
}>();

const { t } = useI18n();

const monthLabel = computed(() =>
  props.month !== null && props.month >= 1 && props.month <= 12 ? t(`desk.month.${props.month}`) : '',
);
</script>

<template>
  <div class="desk-month">
    <span class="month">{{ monthLabel }}</span>
    <span v-if="year !== null" class="year">{{ year }}</span>
  </div>
</template>

<style scoped>
.desk-month {
  display: inline-flex;
  align-items: baseline;
  gap: 8px;
  font-family: var(--font-news);
  font-variant: small-caps;
  font-size: 34px;
  color: rgba(78, 62, 36, 0.52);
  transform: rotate(-1deg);
  border-bottom: 1.5px solid rgba(78, 62, 36, 0.35);
  padding-bottom: 4px;
  line-height: 1;
}
.year {
  font-size: 22px;
}
</style>
