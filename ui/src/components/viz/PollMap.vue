<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { gameLib } from '../../game-bindings';
import { usePartyInk } from './usePartyInk';

defineOptions({ name: 'PollMap' });

interface ProvinceRow {
  id: string;
  label: string;
  value: string;
  party?: string | null;
  population: number;
  seats: number;
}
interface Cell { id: string; label: string; value: number }
interface CrossRow {
  id: string;
  label: string;
  value: number;
  province: string;
  cells: Cell[];
  playerParty?: string | null;
}
interface ProjectionRow {
  id: string;
  label: string;
  value: number;
  province: string;
  party: string;
  share: number;
}

const props = withDefaults(
  defineProps<{
    variant?: 'compact' | 'blank';
    q?: Record<string, unknown>;
  }>(),
  { variant: undefined, q: () => ({}) },
);
const selected = ref('barcelona');
const mapSvg = ref('');
const mapFailed = ref(false);
const partyInk = usePartyInk();
const builders = gameLib.brief as Record<string, (q: Record<string, unknown>) => unknown[]>;
const derive = <T>(name: string): T[] => {
  try {
    return (builders[name]?.(props.q) ?? []) as T[];
  } catch (err) {
    console.warn(`poll-map: ${name} derivation threw`, err);
    return [];
  }
};

const provinces = computed(() => derive<ProvinceRow>('provinces'));
const crossRows = computed(() =>
  derive<CrossRow>('crosstab').filter((row) => row.province === selected.value),
);
const projections = computed(() =>
  derive<ProjectionRow>('seatProjection').filter((row) => row.province === selected.value),
);
const parties = computed<Cell[]>(() => {
  const found = new Map<string, Cell>();
  for (const row of crossRows.value) {
    for (const cell of row.cells) if (!found.has(cell.id)) found.set(cell.id, cell);
  }
  return [...found.values()];
});
const selectedProvince = computed(() =>
  provinces.value.find((row) => row.id === selected.value),
);
const full = computed(() => props.variant === undefined);
const painted = computed(() => props.variant !== 'blank');
const mapStyle = computed<Record<string, string>>(() => {
  const style: Record<string, string> = {};
  for (const row of provinces.value) {
    style[`--province-${row.id}`] = painted.value ? partyInk(row.party) : '#a6a6a6';
  }
  return style;
});

function choose(province: string): void {
  if (!full.value || !provinces.value.some((row) => row.id === province)) return;
  selected.value = province;
}
function onMapClick(event: MouseEvent): void {
  const node = (event.target as Element | null)?.closest?.('.province');
  if (node?.id) choose(node.id);
}
function cellFor(row: CrossRow, party: string): number | null {
  return row.cells.find((cell) => cell.id === party)?.value ?? null;
}
function formatPopulation(value = 0): string {
  return Math.round(value).toLocaleString('en-US');
}

onMounted(async () => {
  try {
    const response = await fetch(
      `${import.meta.env.BASE_URL}img/maps/catalonia-provinces.svg`,
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    mapSvg.value = await response.text();
  } catch (err) {
    // Many component tests mount the whole Desk without an HTTP server; the
    // relative asset URL cannot resolve there. Keep production diagnostics,
    // but do not turn an expected jsdom limitation into hundreds of warnings.
    if (import.meta.env.MODE !== 'test') {
      console.warn('poll-map: map asset failed to load', err);
    }
    mapFailed.value = true;
  }
});
</script>

<template>
  <div
    class="poll-map"
    :class="{ 'poll-map-full': full, 'poll-map-blank': !painted }"
    :style="mapStyle"
    data-test="poll-map"
    :data-variant="variant || 'full'"
  >
    <div class="map-shell" :class="{ 'map-failed': mapFailed }" @click="onMapClick">
      <div v-if="mapSvg" class="map-svg" v-html="mapSvg" />
      <span v-else-if="mapFailed" class="map-error">Map unavailable</span>
    </div>

    <template v-if="full">
      <div class="province-tabs" role="tablist" aria-label="Province">
        <button
          v-for="province in provinces"
          :key="province.id"
          type="button"
          role="tab"
          :aria-selected="selected === province.id"
          :class="{ active: selected === province.id }"
          @click="choose(province.id)"
        >{{ province.label }}</button>
      </div>

      <p v-if="selectedProvince" class="province-caption">
        <b>{{ selectedProvince.label }}</b> · {{ selectedProvince.seats }} seats ·
        {{ formatPopulation(selectedProvince.population) }} potential voters
      </p>

      <section class="poll-section">
        <h3>By demographic</h3>
        <div class="crosstab-wrap">
          <table class="crosstab">
            <thead>
              <tr>
                <th scope="col"></th>
                <th
                  v-for="party in parties"
                  :key="party.id"
                  scope="col"
                  :class="{ yours: party.id === q.player_party }"
                  :style="{ color: partyInk(party.id) }"
                >{{ party.label }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in crossRows" :key="row.id">
                <th scope="row">{{ row.label }}</th>
                <td
                  v-for="party in parties"
                  :key="party.id"
                  :class="{ yours: party.id === q.player_party }"
                >{{ cellFor(row, party.id)?.toFixed(1) ?? '–' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="poll-section projection-section">
        <h3>Seats, projected · {{ selectedProvince?.label }}</h3>
        <div class="projection" role="img" :aria-label="`Projected seats for ${selectedProvince?.label ?? selected}`">
          <div v-for="row in projections" :key="row.id" class="projection-party">
            <b>{{ row.value }}</b>
            <span
              class="projection-column"
              :style="{ height: `${Math.max(3, row.share * 72)}px`, background: partyInk(row.party) }"
            />
            <small>{{ row.label }}</small>
          </div>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
.poll-map { color: #2e2a22; }
.map-shell {
  width: 100%;
  height: 155px;
  overflow: hidden;
  background: #f6f2e6;
  border: 1px solid #e2d8bd;
}
.poll-map:not(.poll-map-full) .map-shell { height: 205px; }
.map-svg, .map-svg :deep(svg) { display: block; width: 100%; height: 100%; }
.map-svg :deep(.province) {
  cursor: pointer;
  stroke: #2e2a22;
  stroke-width: .45;
  transition: opacity .15s ease, stroke-width .15s ease;
}
.map-svg :deep(#barcelona) { fill: var(--province-barcelona, #a6a6a6); }
.map-svg :deep(#tarragona) { fill: var(--province-tarragona, #a6a6a6); }
.map-svg :deep(#lleida) { fill: var(--province-lleida, #a6a6a6); }
.map-svg :deep(#girona) { fill: var(--province-girona, #a6a6a6); }
.poll-map-full .map-svg :deep(.province:hover) { opacity: .72; stroke-width: 1; }
.map-svg :deep(.map-province-label) {
  fill: #2e2a22;
  font-family: var(--font-title);
  font-weight: 700;
}
.map-error {
  display: grid;
  height: 100%;
  place-items: center;
  color: #8a8273;
  font: 700 9px/1 var(--font-title);
  letter-spacing: .1em;
  text-transform: uppercase;
}
.province-tabs { display: flex; gap: 3px; margin-top: 7px; }
.province-tabs button {
  flex: 1 1 0;
  min-width: 0;
  padding: 5px 2px 4px;
  border: 1px solid #d8cfba;
  background: #f6f2e6;
  color: #6b655a;
  font: 700 8.5px/1 var(--font-title);
  letter-spacing: .04em;
  text-transform: uppercase;
  cursor: pointer;
}
.province-tabs button.active {
  border-color: #2e2a22;
  background: #2e2a22;
  color: #faf9f5;
}
.province-caption {
  margin: 6px 0 9px;
  color: #8a8273;
  font: 600 9px/1.3 var(--font-title);
  text-align: right;
}
.poll-section h3 {
  margin: 8px 0 5px;
  color: #a89e8c;
  font: 800 9.5px/1 var(--font-title);
  letter-spacing: .14em;
  text-transform: uppercase;
}
.crosstab-wrap { overflow-x: auto; }
.crosstab {
  width: 100%;
  border-collapse: collapse;
  font: 400 8.5px/1.15 var(--font-typed);
  font-variant-numeric: tabular-nums;
}
.crosstab th, .crosstab td { padding: 3px 4px; text-align: right; }
.crosstab thead th { border-bottom: 1px solid #e3dcc9; font-weight: 700; }
.crosstab tbody th { color: #6b655a; font-family: var(--font-title); font-weight: 600; text-align: left; }
.crosstab tbody tr { border-bottom: 1px solid #f0ead9; }
.crosstab .yours { background: #fbf3dc; color: #7c6120 !important; font-weight: 700; }
.projection {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 5px;
  min-height: 100px;
  padding-top: 5px;
  border-bottom: 1px solid #c6bda8;
}
.projection-party {
  display: flex;
  flex: 1 1 0;
  flex-direction: column;
  align-items: center;
  min-width: 0;
  font-family: var(--font-typed);
}
.projection-party b { margin-bottom: 2px; font-size: 9px; }
.projection-column { display: block; width: min(28px, 72%); }
.projection-party small {
  overflow: hidden;
  width: 100%;
  margin-top: 3px;
  font: 700 8px/1 var(--font-title);
  text-align: center;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
