<script setup lang="ts">

import { computed } from 'vue';
import { arc as d3Arc, line as d3Line } from 'd3-shape';
import { scaleLinear } from 'd3-scale';
import { range as d3Range, sum as d3Sum } from 'd3-array';

defineOptions({ name: 'Hemicycle' });

interface HemicycleSeat {
  party: string;
  seats: number;
  colour: string;
}

const props = withDefaults(
  defineProps<{ seats?: HemicycleSeat[]; majority?: number; q?: Record<string, unknown> }>(),
  {
    seats: () => [],
    majority: 0,
    // Declared (like AchievementGallery) so WidgetHost's `:q` is consumed as a
    // prop, not leaked onto the root <svg> as a q="[object Object]" attribute.
    q: undefined,
  },
);

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

// The angle marking "one more seat than `majority - 1`" — the boundary
// between the seat that completes a majority and the seat before it. Not
// present in the old shell's chart at all (d3-parliament.js has no majority
// concept; the old page prints "Majority: N" as plain text beside the SVG)
// — a genuine Desk-only addition, not a parity port.
const majorityTheta = computed<number | null>(() => {
  const points = seatPoints.value;
  if (points.length === 0) return null;
  if (points.length === 1) return points[0].theta;
  const idx = Math.min(Math.max(Math.round(props.majority), 1), points.length - 1);
  return (points[idx - 1].theta + points[idx].theta) / 2;
});

const majorityLinePath = computed<string | null>(() => {
  const theta = majorityTheta.value;
  if (theta === null) return null;
  const lineGen = d3Line<[number, number]>();
  return lineGen([
    [INNER_RADIUS * Math.cos(theta), INNER_RADIUS * Math.sin(theta)],
    [OUTER_RADIUS * Math.cos(theta), OUTER_RADIUS * Math.sin(theta)],
  ]);
});

const totalSeats = computed(() => seatPoints.value.length);
</script>

<template>
  <svg
    class="hemicycle"
    viewBox="0 0 500 250"
    role="img"
    :aria-label="`Hemicycle: ${totalSeats} seats, majority ${majority}`"
  >
    <g :transform="`translate(${WIDTH / 2}, ${OUTER_RADIUS})`">
      <path
        v-for="(seat, i) in seatPoints"
        :key="i"
        class="seat"
        :class="seat.party.party"
        :transform="`translate(${seat.x}, ${seat.y})`"
        :d="seatPath(seat.seatRadius)"
        :fill="colourVar(seat.party.colour)"
      >
        <title>{{ seat.party.party }}</title>
      </path>
      <path
        v-if="majorityLinePath"
        class="majority-line"
        :data-majority="majority"
        :d="majorityLinePath"
      />
    </g>
  </svg>
</template>

<style scoped>
.hemicycle {
  display: block;
  width: 100%;
  height: auto;
}
.majority-line {
  fill: none;
  stroke: var(--accent-red);
  stroke-width: 1.5;
  stroke-dasharray: 3 2;
  opacity: 0.75;
}
</style>
