<script setup lang="ts">
import { computed } from 'vue';
import { useBand } from '../brief/useBand';
import { usePartyInk } from './usePartyInk';

defineOptions({ name: 'LeaderRows' });

interface LeaderRow {
  id: string;
  label: string;
  value: string | number;
  valueDisplay?: string | null;
  party?: string | null;
}

const props = withDefaults(
  defineProps<{ rows?: LeaderRow[]; q?: Record<string, unknown> }>(),
  { rows: () => [], q: undefined },
);
const { band } = useBand();
const partyInk = usePartyInk();
const views = computed(() =>
  props.rows.map((row) => {
    const classified = band(row.value, row.valueDisplay);
    return {
      ...row,
      classified,
      display: classified.label || String(row.value ?? ''),
      isControl: row.valueDisplay === 'control',
    };
  }),
);
</script>

<template>
  <div class="leader-rows" data-test="leader-rows">
    <div v-for="row in views" :key="row.id" class="brief-row leader-row">
      <span class="leader-label">{{ row.label }}</span>
      <span v-if="row.isControl" class="control-value" :data-band="row.classified.band">
        <b>{{ row.display }}</b>
      </span>
      <span v-else class="person-value">
        <i v-if="row.party" class="party-square" :style="{ background: partyInk(row.party) }" />
        {{ row.display }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.leader-rows { display: grid; gap: 7px; }
.leader-row { display: flex; align-items: baseline; gap: 7px; }
.leader-label { flex: 0 0 auto; font: 600 11.5px/1.2 var(--font-title); color: #3a342c; }
.person-value, .control-value { order: 2; flex: 0 0 auto; }
.person-value { display: inline-flex; align-items: center; font: 600 11px/1.2 var(--font-title); }
.party-square { width: 8px; height: 8px; margin-right: 5px; }
.control-value { display: inline-flex; align-items: center; }
.control-value b {
  min-width: 54px;
  font: 750 9.5px/1 var(--font-title);
  letter-spacing: .07em;
  text-align: right;
  text-transform: uppercase;
}
</style>
