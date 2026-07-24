<script setup lang="ts">
import { computed } from 'vue';
import { useBand } from '../brief/useBand';

defineOptions({ name: 'TensionRows' });

interface TensionRow {
  id: string;
  label: string;
  strength: number;
  dissent: number;
  strengthDisplay?: string | null;
  dissentDisplay?: string | null;
}

const props = withDefaults(
  defineProps<{ rows?: TensionRow[]; q?: Record<string, unknown> }>(),
  { rows: () => [], q: undefined },
);
const { band } = useBand();
const views = computed(() =>
  props.rows.map((row) => ({
    ...row,
    strengthBand: band(row.strength, row.strengthDisplay),
    dissentBand: band(row.dissent, row.dissentDisplay),
  })),
);
</script>

<template>
  <div class="tension-rows" data-test="tension-rows">
    <div v-for="row in views" :key="row.id" class="tension-row">
      <span class="tension-name">{{ row.label }}</span>
      <span class="tension-leader" aria-hidden="true" />
      <span class="metric strength">
        STR <b :data-band="row.strengthBand.band || undefined">{{ row.strengthBand.label }}</b>
      </span>
      <span class="metric-dot">·</span>
      <span class="metric dissent">
        DIS <b :data-band="row.dissentBand.band || undefined">{{ row.dissentBand.label }}</b>
      </span>
    </div>
  </div>
</template>

<style scoped>
.tension-rows { display: grid; gap: 8px; }
.tension-row {
  display: flex;
  align-items: baseline;
  min-width: 0;
  gap: 6px;
}
.tension-name {
  font: 700 13px/1.2 var(--font-news);
  color: #2e2a22;
}
.tension-leader {
  flex: 1 1 20px;
  min-width: 10px;
  border-bottom: 1.5px dotted #c6bda8;
}
.metric {
  flex: 0 0 auto;
  font: 700 9.5px/1 var(--font-title);
  letter-spacing: .06em;
  color: #8a8273;
}
.metric b { text-transform: uppercase; }
.strength b { color: #a9821f; }
.metric-dot { color: #c6bda8; }
</style>
