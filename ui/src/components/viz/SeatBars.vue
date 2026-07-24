<script setup lang="ts">
import { computed } from 'vue';
import { usePartyInk } from './usePartyInk';

defineOptions({ name: 'SeatBars' });

interface SeatRow {
  id: string;
  label: string;
  value: number;
  majority: number;
}

const props = withDefaults(
  defineProps<{ rows?: SeatRow[]; q?: Record<string, unknown> }>(),
  { rows: () => [], q: undefined },
);
const partyInk = usePartyInk();
const total = computed(() => props.rows.reduce((sum, row) => sum + Math.max(0, row.value), 0));
const majority = computed(() => props.rows[0]?.majority ?? 0);
const majorityLeft = computed(() =>
  total.value > 0 ? `${Math.min(100, Math.max(0, (majority.value / total.value) * 100))}%` : '0%',
);
</script>

<template>
  <div class="seat-bars" data-test="seat-bars">
    <div class="seat-strip-wrap">
      <span class="majority-number" :style="{ left: majorityLeft }">{{ majority }}</span>
      <div class="seat-strip" role="img" :aria-label="`${total} seats; majority ${majority}`">
        <span
          v-for="row in rows"
          :key="row.id"
          class="seat-segment"
          :title="`${row.label}: ${row.value}`"
          :style="{ width: total ? `${(Math.max(0, row.value) / total) * 100}%` : '0%', background: partyInk(row.id) }"
        />
        <span class="majority-tick" :style="{ left: majorityLeft }" />
      </div>
    </div>
    <div class="seat-legend">
      <span v-for="row in rows" :key="row.id">
        <i :style="{ background: partyInk(row.id) }" />{{ row.label }} {{ row.value }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.seat-strip-wrap { position: relative; padding-top: 15px; }
.seat-strip { position: relative; display: flex; height: 18px; background: #eee8d8; }
.seat-segment { display: block; min-width: 1px; }
.majority-tick {
  position: absolute;
  top: -4px;
  bottom: -4px;
  width: 2px;
  transform: translateX(-1px);
  background: #2e2a22;
}
.majority-number {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  font: 700 9px/1 var(--font-typed);
  color: #2e2a22;
}
.seat-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
  margin-top: 7px;
  font: 600 9px/1.2 var(--font-title);
  color: #6b655a;
}
.seat-legend span { white-space: nowrap; }
.seat-legend i { display: inline-block; width: 7px; height: 7px; margin-right: 3px; }
</style>
