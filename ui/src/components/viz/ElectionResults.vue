<script setup lang="ts">
import { computed } from 'vue';
import { gameLib } from '../../game-bindings';
import { useSceneWidgetText } from './useSceneWidgetText';
const props = withDefaults(defineProps<{ layout: string; voteHeading?: string; q?: Record<string, unknown> }>(), { voteHeading: 'Popular Vote', q: () => ({}) });
const rows = computed(() => gameLib.electionResultRows(props.q, props.layout));
const { label, html, change } = useSceneWidgetText();
</script>

<template>
  <div class="election-results">
    <table>
      <thead><tr><th scope="col">{{ label('Party') }}</th><th scope="col">{{ label(voteHeading) }}</th><th scope="col">{{ label('Change') }}</th><th scope="col" aria-sort="descending">{{ label('Seats') }}</th><th scope="col">{{ label('Change') }}</th></tr></thead>
      <tbody><tr v-for="row in rows" :key="row.id" :data-party="row.id">
        <th scope="row"><i :style="{ background: row.colour }" /><span v-html="html(row.label)" /></th>
        <td>{{ row.votes }}%<small v-if="row.catalonia !== null"> ({{ row.catalonia }}%)</small></td>
        <td>{{ change(row.voteChange, 2, '%') }}</td><td class="seats">{{ row.seats }}</td><td>{{ change(row.seatChange) }}</td>
      </tr></tbody>
    </table>
  </div>
</template>

<style scoped>
.election-results { overflow-x: auto; margin: 12px 0; }
table { width: 100%; border-collapse: collapse; font: var(--type-compact); font-variant-numeric: tabular-nums; }
th, td { padding: 7px 8px; text-align: right; border-bottom: 1px solid var(--paper-4, #ded8c6); white-space: nowrap; }
th:first-child { text-align: left; }
thead { color: var(--ink-2, #615747); }
tbody th { font-weight: normal; }
i { display: inline-block; width: 9px; height: 9px; margin-right: 7px; }
.seats { font-weight: bold; }
small { color: var(--ink-2, #615747); }
</style>
