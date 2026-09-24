<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { gameLib } from '../../game-bindings';
import { usePartyInk, usePartyTerm } from './usePartyInk';
import { useElectionSvg } from './useElectionSvg';

const props = withDefaults(defineProps<{ q?: Record<string, unknown> }>(), { q: () => ({}) });
const root = ref<HTMLElement | null>(null);
const selected = ref<string | null>(null);
const hoveredRest = ref(false);
const { markup, failed } = useElectionSvg('congreso-results.svg');
const model = computed(() => gameLib.congresoResultsMap(props.q));
const ink = usePartyInk();
const partyTerm = usePartyTerm();
const panel = computed(() => selected.value ? model.value.panels[selected.value] : null);
const maxSeats = computed(() => Math.max(1, ...(panel.value?.rows ?? []).map(row => row.seats)));
const restIds = new Set(model.value.regions.filter(row => row.constituency === 'rest').map(row => row.id));

function regionFor(target: EventTarget | null) {
  let node = target instanceof Element ? target : null;
  const svg = root.value?.querySelector('svg');
  while (node && node !== svg) {
    const region = model.value.regions.find(row => row.id === node?.id);
    if (region) return region;
    node = node.parentElement;
  }
  return null;
}
function select(event: MouseEvent) {
  const region = regionFor(event.target);
  if (region) selected.value = region.constituency;
}
function hover(event: MouseEvent) {
  hoveredRest.value = regionFor(event.target)?.constituency === 'rest';
}
function paint() {
  const svg = root.value?.querySelector('svg');
  if (!svg) return;
  for (const region of model.value.regions) {
    const node = svg.querySelector<SVGElement>(`#${region.id}`);
    if (!node) continue;
    const targets = node.querySelectorAll<SVGElement>('path, polygon, rect, circle');
    const fill = region.winner ? ink(region.winner) : '#d9d9d9';
    for (const target of targets.length ? [...targets] : [node]) {
      target.style.fill = fill;
      target.style.fillOpacity = region.winner ? '.9' : '.45';
    }
    node.classList.add('election-region');
    const term = partyTerm(region.winner);
    if (term) node.setAttribute('data-term', term.id);
    else node.removeAttribute('data-term');
    node.classList.toggle('term-hoverable', !!term?.tooltip);
    node.classList.toggle('selected', selected.value === region.constituency);
    node.classList.toggle('rest-hover', hoveredRest.value && restIds.has(region.id));
  }
}
watch([markup, model, selected, hoveredRest], async () => { await nextTick(); paint(); }, { immediate: true });
</script>

<template>
  <div ref="root" class="congreso-results-map" data-test="congreso-results-map">
    <div v-if="markup" class="map-svg" v-html="markup" @click="select" @mousemove="hover" @mouseleave="hoveredRest = false" />
    <p v-else-if="failed" class="map-message">Map unavailable</p>
    <div class="result-panel" aria-live="polite">
      <template v-if="panel">
        <div class="panel-title"><b>{{ panel.name }}</b><small>{{ panel.seats }} seats</small></div>
        <div v-if="panel.rows.length" class="bars" role="img" :aria-label="`Seat distribution in ${panel.name}`">
          <div v-for="row in panel.rows" :key="row.party" class="bar-column">
            <strong>{{ row.seats }}</strong>
            <span class="bar" :style="{ height: `${Math.max(3, Math.round(row.seats / maxSeats * 70))}%`, background: ink(row.party) }" />
            <small><span :data-term="partyTerm(row.party)?.id" :class="{ 'term-hoverable': !!partyTerm(row.party)?.tooltip }">{{ row.party === 'up' ? 'UP' : row.party === 'te' ? '¡TE!' : row.party === 'mes' ? 'MES' : row.party.toUpperCase() }}</span></small>
          </div>
        </div>
        <p v-else>No results yet.</p>
      </template>
      <p v-else>Click a region to see its seat distribution.</p>
    </div>
  </div>
</template>

<style scoped>
.congreso-results-map { width: 100%; color: #2e2a22; }
.map-svg :deep(svg) { width: 100%; height: auto; display: block; }
.map-svg :deep(.election-region) { cursor: pointer; stroke: #181713 !important; stroke-width: .4 !important; }
.map-svg :deep(.election-region:hover), .map-svg :deep(.rest-hover) { opacity: .72; stroke-width: 1.1 !important; }
.map-svg :deep(.selected) { stroke-width: 1.7 !important; }
.result-panel { min-height: 70px; margin-top: 6px; font: 11px/1.3 var(--font-title); }
.result-panel p { margin: 0; color: #8a8273; text-align: right; }
.panel-title { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; }
.panel-title small { color: #8a8273; }
.bars { display: flex; flex-direction: row-reverse; justify-content: flex-end; align-items: end; gap: 3px; height: 100px; border-bottom: 1px solid #c6bda8; }
.bar-column { min-width: 20px; flex: 1 1 0; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; font: 9px/1 var(--font-title); }
.bar-column .bar { display: block; width: 80%; min-height: 3px; border-radius: 2px 2px 0 0; }
.bar-column strong { margin-bottom: 3px; }.bar-column small { margin-top: 3px; font-size: 8px; }
.map-message { min-height: 120px; display: grid; place-items: center; }
</style>
