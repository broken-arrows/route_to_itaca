<script setup lang="ts">
import { computed } from 'vue';
import { useBand } from '../brief/useBand';

defineOptions({ name: 'LevelBars' });

interface LevelRow {
  id: string;
  label: string;
  value: number;
  valueDisplay?: string | null;
  share: number;
}

const props = withDefaults(
  defineProps<{ rows?: LevelRow[]; q?: Record<string, unknown> }>(),
  { rows: () => [], q: undefined },
);
const { band } = useBand();
const clamp = (n: number) => Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));

const views = computed(() =>
  props.rows.map((row) => {
    const classified = band(row.value, row.valueDisplay);
    return {
      ...row,
      width: `${clamp(row.share) * 100}%`,
      band: classified.band,
      display: classified.label || String(row.value),
    };
  }),
);
</script>

<template>
  <div class="level-bars" data-test="level-bars">
    <div v-for="row in views" :key="row.id" class="level-row" :data-test="`level-${row.id}`">
      <div class="level-head">
        <span>{{ row.label }}</span>
        <span class="level-word" :data-band="row.band || undefined">{{ row.display }}</span>
      </div>
      <div class="level-track" aria-hidden="true">
        <span class="level-fill" :data-band="row.band || undefined" :style="{ width: row.width }" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.level-bars { display: grid; gap: 8px; }
.level-head {
  font: var(--type-compact);
  letter-spacing: var(--type-tracking);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;

  color: #3a342c;
}
.level-word {
  font: var(--type-compact);
  letter-spacing: var(--type-tracking);
  flex: 0 0 auto;
  text-transform: uppercase;
}
.level-track {
  height: 7px;
  overflow: hidden;
  background: #eee8d8;
}
.level-fill {
  display: block;
  height: 100%;
  min-width: 1px;
  background: currentColor;
}
</style>
