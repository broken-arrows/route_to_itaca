<script setup lang="ts">

import { computed, ref } from 'vue';
import { arc as d3Arc } from 'd3-shape';
import { scaleLinear } from 'd3-scale';
import { range as d3Range, sum as d3Sum } from 'd3-array';
import { useGameStore } from '../../stores/game';

defineOptions({ name: 'Hemicycle' });

interface HemicycleSeat {
  party: string;
  seats: number;
  colour: string;
}

const props = withDefaults(
  defineProps<{
    seats?: HemicycleSeat[];
    majority?: number;
    animate?: boolean;
    q?: Record<string, unknown>;
  }>(),
  {
    seats: () => [],
    majority: 0,
    animate: false,
    // Declared (like AchievementGallery) so WidgetHost's `:q` is consumed as a
    // prop, not leaked onto the root <svg> as a q="[object Object]" attribute.
    q: undefined,
  },
);

const game = useGameStore();
const activeParty = ref<string | null>(null);
const tooltipPosition = ref({ left: '0px', top: '0px' });

// Fixed canvas — same base dimensions as the old shell's own
// `<svg id="parlament" style="width: 500px; height: 250px;">`, and the same
// 500-wide/half-as-tall relationship d3-parliament.js's `parliament(data)`
// computes internally (`height = width / 2`). The SVG scales via CSS
// (`width: 100%`) against a fixed `viewBox`, so this component needs no
// ResizeObserver/DOM measurement and stays trivially testable in jsdom.
const WIDTH = 500;
const HEIGHT = 250;
const INNER_RADIUS_COEF = 0.4;
const OUTER_RADIUS = Math.min(WIDTH / 2, HEIGHT);
const INNER_RADIUS = OUTER_RADIUS * INNER_RADIUS_COEF;

interface SeatPoint {
  x: number;
  y: number;
  theta: number;
  radius: number;
  seatRadius: number;
  party: HemicycleSeat;
}

// Ported row/seat layout from out/html/d3-parliament.js (`parliament(data)`,
// ~lines 56-136): grow concentric rows outward from innerRadius until they
// can hold every seat, place each seat at its row's cartesian coordinate,
// sort left-to-right by angle, then walk the sorted seats assigning parties
// IN THE ORDER `seats` ARRIVES (matching the old renderer's `d[partyIndex]`
// walk) so the leftmost seat belongs to the first party in the prop array.
const seatPoints = computed<SeatPoint[]>(() => {
  const parties = props.seats;
  const nSeats = d3Sum(parties, (p) => Math.max(0, Math.floor(p.seats)));
  if (!nSeats) return [];

  const a = INNER_RADIUS_COEF / (1 - INNER_RADIUS_COEF);
  let nRows = 0;
  let maxSeatNumber = 0;
  let b = 0.5;
  while (maxSeatNumber < nSeats) {
    nRows++;
    b += a;
    // NOTE (ported verbatim): the seats available per row depend on the
    // TOTAL row count, and floor() means adding a row can't just increment
    // the running total — the whole sum is recomputed every growth step.
    maxSeatNumber = d3Sum(d3Range(nRows), (i) => Math.floor(Math.PI * (b + i)));
  }

  const rowRadiusScale = scaleLinear().domain([0, nRows]).range([INNER_RADIUS, OUTER_RADIUS]);
  const rowWidth = (OUTER_RADIUS - INNER_RADIUS) / nRows;
  const seatsToRemove = maxSeatNumber - nSeats;

  const raw: Omit<SeatPoint, 'party'>[] = [];
  for (let i = 0; i < nRows; i++) {
    const rowRadius = rowRadiusScale(i + 0.5);
    const rowSeats =
      Math.floor(Math.PI * (b + i)) -
      Math.floor(seatsToRemove / nRows) -
      (seatsToRemove % nRows > i ? 1 : 0);
    const anglePerSeat = Math.PI / rowSeats;
    for (let j = 0; j < rowSeats; j++) {
      const theta = -Math.PI + anglePerSeat * (j + 0.5);
      raw.push({
        theta,
        radius: rowRadius,
        x: rowRadius * Math.cos(theta),
        y: rowRadius * Math.sin(theta),
        seatRadius: 0.4 * rowWidth,
      });
    }
  }

  // Sort by angle (left → right); tie-break by descending radius, same as
  // the old renderer (`a.polar.teta - b.polar.teta || b.polar.r - a.polar.r`).
  raw.sort((p, q) => p.theta - q.theta || q.radius - p.radius);

  const assigned: SeatPoint[] = [];
  let partyIndex = 0;
  let seatIndex = 0;
  for (const point of raw) {
    let party = parties[partyIndex];
    while (party && seatIndex >= party.seats) {
      partyIndex += 1;
      seatIndex = 0;
      party = parties[partyIndex];
    }
    // Defensive only: `nSeats` is summed from `parties` itself above, so
    // running out of party before running out of seat is not reachable in
    // practice — guards a future caller passing a mismatched seat count.
    if (!party) break;
    assigned.push({ ...point, party });
    seatIndex += 1;
  }
  return assigned;
});

// A seat "dot" is a full-circle arc (innerRadius 0) — d3-shape's arc
// generator, the one piece of chart-library math this component leans on;
// every resulting <path> is still emitted by Vue's own template, not d3.
const seatGlyph = d3Arc<unknown, { r: number }>()
  .innerRadius(0)
  .outerRadius((d) => d.r)
  .startAngle(0)
  .endAngle(Math.PI * 2);

function seatPath(seatRadius: number): string {
  return seatGlyph({ r: seatRadius }) ?? '';
}

function colourVar(token: string): string {
  // Token-OR-hex, matching every other colour path (glossary colourValue, the
  // old shell's cssColour): a raw hex (e.g. a party the palette has no --var
  // for) renders literally; a token resolves to its CSS custom property.
  return token.startsWith('#') ? token : `var(--${token})`;
}

const totalSeats = computed(() => seatPoints.value.length);

const activeSeatGroup = computed(() =>
  props.seats.find((party) => party.party === activeParty.value),
);

const activeTerm = computed(() => {
  const token = activeParty.value?.toLowerCase();
  if (!token) return undefined;
  return game.glossary.find((term) =>
    term.match.some((match) => match.toLowerCase() === token),
  ) ?? game.glossary.find((term) => term.bold && term.colour === token);
});

const tooltipName = computed(() =>
  activeTerm.value?.display ?? activeTerm.value?.tooltip?.title ?? activeParty.value ?? '',
);

const tooltipLogo = computed(() => {
  const path = activeTerm.value?.tooltip?.img;
  return path ? new URL(path, document.baseURI).href : null;
});

function seatStyle(seat: SeatPoint, index: number): Record<string, string> | undefined {
  if (!props.animate) return undefined;
  return {
    '--seat-x': `${seat.x}px`,
    '--seat-y': `${seat.y}px`,
    // The old D3 renderer launches every seat together, with an independent
    // 1–1.8s duration. A deterministic spread retains that lively, irregular
    // arrival without making screenshots and tests random.
    '--seat-duration': `${1000 + ((index * 137) % 800)}ms`,
  };
}

function positionTooltip(event: MouseEvent | FocusEvent): void {
  if (event instanceof MouseEvent) {
    tooltipPosition.value = { left: `${event.clientX + 10}px`, top: `${event.clientY - 50}px` };
    return;
  }
  const rect = (event.currentTarget as SVGCircleElement).getBoundingClientRect();
  tooltipPosition.value = { left: `${rect.right + 10}px`, top: `${rect.top - 20}px` };
}

function showParty(party: string, event: MouseEvent | FocusEvent): void {
  activeParty.value = party;
  positionTooltip(event);
}

function hideParty(): void {
  activeParty.value = null;
}

function isFirstPartySeat(index: number): boolean {
  return index === 0 || seatPoints.value[index - 1]?.party.party !== seatPoints.value[index]?.party.party;
}
</script>

<template>
  <div class="hemicycle-wrap">
    <svg
      class="hemicycle"
      :class="{ 'is-animated': animate }"
      viewBox="0 0 500 250"
      role="img"
      :aria-label="`Hemicycle: ${totalSeats} seats, majority ${majority}`"
    >
      <g :transform="`translate(${WIDTH / 2}, ${OUTER_RADIUS})`">
        <g
          v-for="(seat, i) in seatPoints"
          :key="i"
          class="seat-position"
          :transform="`translate(${seat.x}, ${seat.y})`"
          :style="seatStyle(seat, i)"
        >
          <path
            class="seat"
            :class="[
              seat.party.party,
              {
                'party-hovered': activeParty === seat.party.party,
                'party-nothovered': activeParty && activeParty !== seat.party.party,
              },
            ]"
            :d="seatPath(seat.seatRadius)"
            :fill="colourVar(seat.party.colour)"
          />
          <circle
            class="seat-hit"
            :class="seat.party.party"
            cx="0"
            cy="0"
            :r="seat.seatRadius * 2"
            :tabindex="isFirstPartySeat(i) ? 0 : -1"
            :aria-label="`${seat.party.party}: ${seat.party.seats} seats`"
            @mouseenter="showParty(seat.party.party, $event)"
            @mousemove="positionTooltip"
            @mouseleave="hideParty"
            @focus="showParty(seat.party.party, $event)"
            @blur="hideParty"
          />
        </g>
      </g>
    </svg>
    <Teleport to="body">
      <div
        v-if="activeSeatGroup"
        class="hemicycle-tooltip"
        data-test="hemicycle-tooltip"
        role="tooltip"
        :style="{ ...tooltipPosition, borderColor: colourVar(activeSeatGroup.colour) }"
      >
        <img v-if="tooltipLogo" :src="tooltipLogo" :alt="tooltipName" />
        <div>
          <strong :style="{ color: colourVar(activeSeatGroup.colour) }">{{ tooltipName }}</strong>
          <span>{{ activeSeatGroup.seats }} seat{{ activeSeatGroup.seats === 1 ? '' : 's' }}</span>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.hemicycle-wrap {
  width: 100%;
}
.hemicycle {
  display: block;
  width: 100%;
  height: auto;
}
.is-animated .seat-position {
  animation: seat-enter var(--seat-duration) cubic-bezier(.4, 0, .2, 1) both;
}
@keyframes seat-enter {
  from { transform: translate(0, 0) scale(0); }
  to { transform: translate(var(--seat-x), var(--seat-y)) scale(1); }
}
.seat {
  transition: opacity 140ms ease, filter 140ms ease;
}
.seat.party-hovered {
  filter: saturate(1.2) brightness(1.05);
}
.seat.party-nothovered {
  opacity: .2;
}
.seat-hit {
  fill: transparent;
  pointer-events: all;
  cursor: help;
  outline: none;
}
.seat-hit:focus-visible {
  stroke: var(--ink-0);
  stroke-width: 1.5;
}
.hemicycle-tooltip {
  position: fixed;
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 150px;
  padding: 9px 11px;
  color: var(--ink-0);
  background: var(--paper-0);
  border: 2px solid var(--ink-0);
  border-radius: 3px;
  box-shadow: 0 8px 20px rgba(46, 42, 34, .28);
  font-family: var(--font-body);
  pointer-events: none;
}
.hemicycle-tooltip img {
  width: auto;
  max-width: 120px;
  height: 42px;
  object-fit: contain;
}
.hemicycle-tooltip strong,
.hemicycle-tooltip span {
  display: block;
}
.hemicycle-tooltip strong {
  font-family: var(--font-news);
  font-size: 14px;
}
@media (prefers-reduced-motion: reduce) {
  .is-animated .seat-position { animation: none; }
}
</style>
