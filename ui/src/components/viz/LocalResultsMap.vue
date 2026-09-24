<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { gameLib } from '../../game-bindings';
import { usePartyInk, usePartyTerm } from './usePartyInk';
import { useElectionSvg } from './useElectionSvg';

const props = withDefaults(defineProps<{ q?: Record<string, unknown> }>(), { q: () => ({}) });
const root = ref<HTMLElement | null>(null);
const picked = ref<{ place: string; party: string } | null>(null);
const { markup, failed } = useElectionSvg('local-results.svg');
const model = computed(() => gameLib.localResultsMap(props.q));
const winners = computed(() => model.value.winners);
const ink = usePartyInk();
const partyTerm = usePartyTerm();
function labelFor(party: string) {
  const term = partyTerm(party);
  return term?.display?.replace(/<[^>]*>/g, '') || party.toUpperCase();
}

function placeFor(target: EventTarget | null) {
  let node = target instanceof Element ? target : null;
  const svg = root.value?.querySelector('svg');
  while (node && node !== svg) {
    if (node.id && winners.value[node.id]) return node.id;
    node = node.parentElement;
  }
  return null;
}
function choose(event: MouseEvent) {
  const id = placeFor(event.target);
  if (!id) return;
  const city = model.value.cities.find(row => row.id === id);
  if (!city) return;
  picked.value = { place: city.name, party: winners.value[id] };
}
function paint() {
  const host = root.value?.querySelector<HTMLElement>('.map-svg');
  if (!host || !markup.value) return;
  // Legacy `initCatLocalMap` rebuilds the outline on every mount. Recreate it
  // before painting Q so a removed winner cannot leave a stale colour/label.
  host.innerHTML = markup.value;
  const svg = host.querySelector('svg');
  if (!svg) return;
  if (picked.value && !model.value.cities.some(city => city.name === picked.value?.place && city.winner === picked.value?.party)) {
    picked.value = null;
  }
  for (const [id, party] of Object.entries(winners.value)) {
    const node = svg.querySelector<SVGElement>(`#${id}`);
    if (!node) continue;
    const term = partyTerm(party);
    if (term) node.setAttribute('data-term', term.id);
    else node.removeAttribute('data-term');
    node.classList.toggle('term-hoverable', !!term?.tooltip);
    const targets = node.querySelectorAll<SVGElement>('path, polygon, rect, circle');
    const city = model.value.cities.find(row => row.id === id);
    for (const target of targets.length ? [...targets] : [node]) {
      target.style.fill = ink(party);
      if (!city) target.style.fillOpacity = '.72';
      else { target.style.stroke = '#fff'; target.style.strokeWidth = '1'; }
    }
    if (city) {
      node.style.cursor = 'pointer';
      node.setAttribute('aria-label', `${city.name}: ${labelFor(party)}`);
      const extra = svg.querySelector<SVGElement>(`#${id}-extras`);
      if (extra) { extra.textContent = labelFor(party); extra.style.fill = ink(party); }
    }
  }
}
watch([markup, winners], async () => { await nextTick(); paint(); }, { immediate: true });
</script>

<template>
  <div ref="root" class="local-results-map" data-test="local-results-map">
    <div v-if="markup" class="map-svg" v-html="markup" @click="choose" />
    <p v-else-if="failed" class="map-message">Map unavailable</p>
    <div class="legend" aria-live="polite">
      <template v-if="picked"><b>{{ picked.place }}</b><span class="swatch" :style="{ background: ink(picked.party) }" /><span :data-term="partyTerm(picked.party)?.id" :class="{ 'term-hoverable': !!partyTerm(picked.party)?.tooltip }">{{ labelFor(picked.party) }}</span></template>
      <template v-else>Click a city to see results</template>
    </div>
  </div>
</template>

<style scoped>
.local-results-map { width: 100%; position: relative; color: #2e2a22; }
.map-svg :deep(svg) { width: 100%; height: auto; display: block; }
.map-svg :deep(circle[aria-label]:hover) { stroke-width: 3 !important; }
.legend { min-height: 24px; display: flex; gap: 6px; align-items: center; font: 10px/1.2 var(--font-title); }
.swatch { width: 12px; height: 12px; display: inline-block; }
.map-message { min-height: 120px; display: grid; place-items: center; }
</style>
