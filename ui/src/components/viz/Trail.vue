<script setup lang="ts">
// The ECONOMY sheet's three full-run trails — brief-frames.md §1 shape 1
// ("A NUMBER THAT MOVES") and §2 frame 14. Content marks
// `<div data-widget="trail" data-props='{"deriveFrom":"trails"}'></div>`;
// WidgetHost resolves the `trails` derivation (source/lib/brief.js) into this
// component's `rows` prop — the widget never learns where they came from
// (docs/design/desk_ui_plan.md §2.5's invariant).
//
// Vue owns every <svg>/<path>/<circle>; d3 supplies pure MATH only — scaleLinear
// for the axes, d3-shape's line() for the path `d` string, d3-array min/max for
// the y-domain. NO chart library, NO d3-select, no DOM manipulation, the same
// contract Hemicycle.vue follows — so the widget stays trivially testable in
// jsdom (a fixed viewBox scaled by CSS, no ResizeObserver).
//
// Shape 1 per row: a header line (key · dotted leader · value + direction arrow)
// over a full-run trail against a dashed zero ground line. The value+arrow and
// the inked current-month dot are GREEN when the move is good for THIS metric,
// RED when bad — `good` decides the colour, never `dir` (unemployment falling is
// good). Negatives read BELOW the line, so the y-domain is
// [min(0,…series), max(0,…series)]; the current month is the last reading.
import { computed } from 'vue';
import { scaleLinear } from 'd3-scale';
import { line as d3Line } from 'd3-shape';
import { max as d3Max, min as d3Min } from 'd3-array';

defineOptions({ name: 'Trail' });

type Dir = 'up' | 'down' | 'flat';
interface TrailRow {
  id: string;
  label: string;
  value: number;
  unit: string;
  dir: Dir;
  good: boolean;
  series: number[];
}

const props = withDefaults(
  defineProps<{ rows?: TrailRow[]; q?: Record<string, unknown> }>(),
  {
    rows: () => [],
    // Declared (as Hemicycle does) so WidgetHost's `:q` is consumed as a prop,
    // not leaked onto the root element as q="[object Object]".
    q: undefined,
  },
);

// §1 shape 1 anchors, shared with BriefSheet's band ramp: good green / bad red.
const GOOD_INK = '#3f8f3f';
const BAD_INK = '#b03030';
// The trail's own marks (§1 shape 1): line `#b5ab94`, dashed zero ground line
// `#d9d2c4`. The dot's ring is the sheet surface (--paper-0) so the mark stays
// legible where it crosses the line (dataviz marks-and-anatomy: surface ring).
const LINE_INK = '#b5ab94';
const ZERO_INK = '#d9d2c4';
const SURFACE = '#faf9f5';

// viewBox units — the design's full-run box (§2 frame 14: 450×110). Scaled to
// the sheet by CSS (width:100%), aspect preserved (preserveAspectRatio=none
// would shear the current-month dot into an ellipse); at the Brief's stable
// ~450px inner width the rendered height lands at ~110px.
const W = 450;
const H = 110;
const PAD_X = 6;
const PAD_Y = 8;
const L = PAD_X;
const R = W - PAD_X;
const T = PAD_Y;
const B = H - PAD_Y;

const ARROW: Record<Dir, string> = { up: '▲', down: '▼', flat: '' };
const DIR_WORD: Record<Dir, string> = { up: 'rising', down: 'falling', flat: 'flat' };

// Colour is decided by `good`, NOT `dir`: a rise can be bad (unemployment) and a
// fall can be good, so the trend arrow (`dir`) and the ink (`good`) are
// orthogonal. brief.js has already reconciled them per metric.
function ink(row: TrailRow): string {
  return row.good ? GOOD_INK : BAD_INK;
}

interface Plot {
  points: number; // count of finite readings
  d: string | null; // line path 'd', null for < 2 points (a lone dot instead)
  zeroY: number; // y of the zero ground line, in viewBox units
  zeroFrac: number; // fraction of the box height (from the top) where 0 sits
  dotX: number; // current-month dot (unused when points === 0)
  dotY: number;
}

function plotFor(series: number[]): Plot {
  const clean = (series ?? []).filter((v) => Number.isFinite(v));
  if (clean.length === 0) {
    return { points: 0, d: null, zeroY: B, zeroFrac: 1, dotX: 0, dotY: 0 };
  }
  const lo = Math.min(0, d3Min(clean) ?? 0);
  const hi = Math.max(0, d3Max(clean) ?? 0);
  // Degenerate domain (every reading exactly 0): open it a hair so 0 lands
  // mid-box and the scale never divides by zero.
  const yDomain: [number, number] = lo === hi ? [lo - 1, hi + 1] : [lo, hi];
  const y = scaleLinear().domain(yDomain).range([B, T]);
  const x = scaleLinear()
    .domain(clean.length > 1 ? [0, clean.length - 1] : [0, 1])
    .range([L, R]);
  const gen = d3Line<number>()
    .x((_, i) => x(i))
    .y((v) => y(v))
    .defined((v) => Number.isFinite(v));
  return {
    points: clean.length,
    d: clean.length > 1 ? gen(clean) : null,
    zeroY: y(0),
    zeroFrac: hi === lo ? 0.5 : hi / (hi - lo),
    // Current month = the last reading. A lone reading has no run to sit at the
    // end of, so it centres.
    dotX: clean.length > 1 ? x(clean.length - 1) : (L + R) / 2,
    dotY: y(clean[clean.length - 1]),
  };
}

interface View extends TrailRow {
  ink: string;
  arrow: string;
  dirWord: string;
  plot: Plot;
}

const views = computed<View[]>(() => {
  return (props.rows ?? []).map((r) => ({
    ...r,
    ink: ink(r),
    arrow: ARROW[r.dir] ?? '',
    dirWord: DIR_WORD[r.dir] ?? 'flat',
    plot: plotFor(r.series),
  }));
});
</script>

<template>
  <div class="trail" data-test="trail">
    <div v-for="v in views" :key="v.id" class="trail-block" :data-test="`trail-${v.id}`">
      <!-- Header: shape 1's `key · dotted leader · value+arrow`. `.brief-row`
           borrows BriefSheet's grammar (leader + right-aligned value) when
           mounted inside a sheet; the value's good/bad ink is inline so it wins
           over the grammar's default #2e2a22. -->
      <div class="brief-row trail-head">
        <span class="trail-key">{{ v.label }}</span>
        <span class="trail-value" :style="{ color: v.ink }"
          >{{ v.value }}{{ v.unit
          }}<span v-if="v.arrow" class="trail-arrow" aria-hidden="true"> {{ v.arrow }}</span></span
        >
      </div>

      <svg
        v-if="v.plot.points > 0"
        class="trail-plot"
        :viewBox="`0 0 ${W} ${H}`"
        role="img"
        :aria-label="`${v.label}: full-run trend, now ${v.value}${v.unit}, ${v.dirWord}`"
      >
        <line
          class="trail-zero"
          :x1="L"
          :x2="R"
          :y1="v.plot.zeroY"
          :y2="v.plot.zeroY"
          :data-zero-frac="v.plot.zeroFrac"
          :stroke="ZERO_INK"
        />
        <path v-if="v.plot.d" class="trail-line" :d="v.plot.d" :stroke="LINE_INK" />
        <circle
          class="trail-dot"
          :cx="v.plot.dotX"
          :cy="v.plot.dotY"
          r="4"
          :fill="v.ink"
          :stroke="SURFACE"
        />
      </svg>
      <p v-else class="trail-empty">start of the run</p>

      <!-- Axis captions: the design's "nov 2012 · start of the run → this month".
           The row carries no start month, so the invented date is dropped
           (filler-copy rule) and only the neutral structural ends remain. -->
      <div v-if="v.plot.points > 0" class="trail-axis" aria-hidden="true">
        <span>start of the run</span>
        <span>this month</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.trail {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
/* Header fallback for when the widget is NOT inside a sheet (tests, previews);
   in a sheet, BriefSheet's `:deep(.brief-row)` grammar supplies the dotted
   leader and the same type ramp. Values here match that grammar so the two
   never disagree. */
.trail-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.trail-key {
  font-family: var(--font-title);
  font-weight: 600;
  font-size: 15px;
  color: #3a342c;
}
.trail-value {
  flex: 0 0 auto;
  font-family: var(--font-news);
  font-weight: 700;
  font-size: 15px;
  white-space: nowrap;
}
.trail-arrow {
  font-size: 0.82em;
}
.trail-plot {
  display: block;
  width: 100%;
  height: auto;
  margin-top: 3px;
  overflow: visible;
}
.trail-zero {
  stroke-width: 1;
  stroke-dasharray: 3 3;
}
.trail-line {
  fill: none;
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.trail-dot {
  stroke-width: 2;
}
.trail-axis {
  display: flex;
  justify-content: space-between;
  margin-top: 3px;
}
.trail-axis > span {
  font-family: var(--font-title);
  font-weight: 600;
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #a89e8c;
}
.trail-empty {
  font-family: var(--font-title);
  font-weight: 600;
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #a89e8c;
  margin: 6px 0 0;
}
</style>
