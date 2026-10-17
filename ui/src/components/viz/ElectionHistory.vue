<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { scaleLinear, scaleUtc } from 'd3-scale';
import { line } from 'd3-shape';
import { useGameStore } from '../../stores/game';
import { markGlossary } from '../../glossary/mark';
import Prose from '../Prose.vue';
import { usePartyTerm } from './usePartyInk';
import { useGlossary } from '../../glossary/useGlossary';

const props = withDefaults(defineProps<{
  recordsFrom: string;
  partiesFrom: string;
  eventsFrom?: string;
  partyNames?: Record<string, string>;
  partyColours?: Record<string, string>;
  q?: Record<string, unknown>;
}>(), { partyNames: () => ({}), partyColours: () => ({}), q: () => ({}) });

const game = useGameStore();
const { t, locale } = useI18n();
const active = ref<string | null>(null);
const hoverX = ref<number | null>(null);
const partyTerm = usePartyTerm();
const { colourValue } = useGlossary();
const L = 42, R = 480;
const TOP = 32;
const EVENT_ANGLE = -90;
interface RecordPoint { date: Date; values: Record<string, unknown> }
interface Point { date: Date; value: number }

const records = computed<RecordPoint[]>(() => {
  const raw = props.q[props.recordsFrom];
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((record) => {
    if (!record || typeof record !== 'object' || record.date == null) return [];
    const date = new Date(record.date);
    return Number.isFinite(date.getTime()) ? [{ date, values: record }] : [];
  }).sort((a, b) => +a.date - +b.date);
});
const parties = computed<string[]>(() => {
  const raw = props.q[props.partiesFrom];
  return Array.isArray(raw) ? [...new Set(raw.filter((id): id is string => typeof id === 'string'))] : [];
});
const events = computed(() => {
  void locale.value;
  const raw = props.eventsFrom ? props.q[props.eventsFrom] : null;
  const first = records.value[0]?.date;
  if (!Array.isArray(raw) || !first) return [];
  const grouped = new Map<number, { date: Date; labels: string[] }>();
  for (const event of raw) {
    if (!event || event.date == null || typeof event.label !== 'string') continue;
    const date = new Date(event.date);
    if (!Number.isFinite(+date) || +date < +first) continue;
    const entry = grouped.get(+date) ?? { date, labels: [] };
    const label = game.adapter?.translate(event.label) ?? event.label;
    if (!entry.labels.includes(label)) entry.labels.push(label);
    grouped.set(+date, entry);
  }
  return [...grouped.values()].sort((a, b) => +a.date - +b.date)
    .map(event => ({ ...event, label: event.labels.join(' · ') }));
});
// These histories measure the whole electorate, including abstention. Keep
// the recorded percentages; they are not normalized election vote shares.
function reading(record: RecordPoint, party: string): number | null {
  const value = record.values[party];
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}
const eventLines = computed(() => events.value.map(event => event.labels));
const x = computed(() => {
  const first = records.value[0]?.date ?? new Date(0);
  const last = new Date(Math.max(+(records.value.at(-1)?.date ?? first), +(events.value.at(-1)?.date ?? first)));
  let start = +first === +last ? +first - 86400000 : +first;
  const end = +first === +last ? +last + 86400000 : +last;
  // Keep every label to the left of its marker without crossing the percentage axis.
  events.value.forEach((event, index) => {
    const padding = Math.min(0.75, (eventLines.value[index].length * 14 + 8) / (R - L));
    start = Math.min(start, (+event.date - padding * end) / (1 - padding));
  });
  return scaleUtc().domain([new Date(start), new Date(end)]).range([L, R]);
});
const annotations = computed(() => {
  const lines = eventLines.value;
  return events.value.map((event, index) => ({
    ...event,
    labelX: x.value(event.date) - 6 - (lines[index].length - 1) * 14,
    labelY: TOP + 4,
    lines: lines[index],
  }));
});
const width = 600;
const top = computed(() => TOP);
const visibleParties = computed(() => parties.value.filter(id => records.value.some(record => reading(record, id) !== null)).length);
const bottom = computed(() => TOP + Math.max(243, visibleParties.value * 18 + 12));
const height = computed(() => bottom.value + 45);
const y = computed(() => {
  let max = 0;
  for (const record of records.value) {
    for (const party of parties.value) max = Math.max(max, reading(record, party) ?? 0);
  }
  return scaleLinear().domain([0, max + 10]).nice(5).range([bottom.value, top.value]);
});
const monthFormat = computed(() => new Intl.DateTimeFormat(locale.value, { month: 'short', timeZone: 'UTC' }));
function dateLabel(date: Date): string {
  const month = monthFormat.value.format(date).replace(/\.$/, '');
  return `${month[0].toLocaleUpperCase(locale.value)}${month.slice(1)}. ${date.getUTCFullYear()}`;
}
function nearestRecord(px: number): RecordPoint | undefined {
  const time = +x.value.invert(px);
  return records.value.reduce<RecordPoint | undefined>((best, record) =>
    !best || Math.abs(+record.date - time) < Math.abs(+best.date - time) ? record : best, undefined);
}
const hoverRecord = computed(() => hoverX.value === null ? undefined : nearestRecord(hoverX.value));
const hoverDate = computed(() => {
  const record = hoverRecord.value;
  if (!record) return '';
  const date = dateLabel(record.date);
  const week = record.values.week;
  return typeof week === 'number' && Number.isInteger(week) && week >= 1 && week <= 4
    ? t('widgets.history.weekDate', { date, week }) : date;
});
const hoverDateLines = computed(() => hoverDate.value.split('\n'));
const hoverLabelWidth = computed(() => Math.min(R - L, Math.max(...hoverDateLines.value.map(line => line.length)) * 6.5));
const hoverLabelX = computed(() => Math.max(L + hoverLabelWidth.value / 2, Math.min(R - hoverLabelWidth.value / 2, hoverX.value ?? L)));
function onPlotMove(event: MouseEvent): void {
  if (event.target instanceof Element && event.target.closest('.history-line, .history-line-hit, .history-point')) {
    hoverX.value = null;
    return;
  }
  const box = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
  const px = (event.clientX - box.left) * width / box.width;
  const py = (event.clientY - box.top) * height.value / box.height;
  const record = px >= L && px <= R && py >= top.value && py <= bottom.value ? nearestRecord(px) : undefined;
  hoverX.value = record ? x.value(record.date) : null;
}
const numberFormat = computed(() => new Intl.NumberFormat(locale.value, { maximumFractionDigits: 1 }));
const percentage = (value: number) => `${numberFormat.value.format(value)}%`;
const xTicks = computed(() => {
  if (records.value.length === 1) return [records.value[0].date];
  return x.value.ticks(4);
});
const series = computed(() => parties.value.flatMap((id) => {
  const points: Point[] = records.value.flatMap(record => {
    const value = reading(record, id);
    return value === null ? [] : [{ date: record.date, value }];
  });
  if (!points.length) return [];
  const name = props.partyNames[id] ?? id;
  const term = partyTerm(id) ?? partyTerm(name);
  const last = points[points.length - 1];
  const path = line<RecordPoint>()
    // A zero/missing reading separates party lifetimes, matching Classic.
    .defined(record => reading(record, id) !== null)
    .x(record => x.value(record.date))
    .y(record => y.value(reading(record, id)!))(records.value);
  return [{ id, name, last, path, colour: props.partyColours[id] ?? '#8a8273',
    labelColour: colourValue(term?.colour) ?? props.partyColours[id] ?? '#8a8273',
    label: markGlossary(name, game.glossary),
    summary: `${name}: ${percentage(last.value)} · ${dateLabel(last.date)}` }];
}));
const highlighted = computed(() => series.value.some(row => row.id === active.value) ? active.value : null);
const hoverPoints = computed(() => {
  const record = hoverRecord.value;
  return record ? series.value.flatMap(row => {
    const value = reading(record, row.id);
    return value === null ? [] : [{ id: row.id, value, colour: row.colour }];
  }) : [];
});
const labelledSeries = computed(() => {
  const placed: Array<{ left: number; right: number; y: number }> = [];
  const labels = new Map<string, number>();
  for (const row of [...series.value].sort((a, b) => a.last.value - b.last.value)) {
    const left = x.value(row.last.date) + 9;
    const right = left + row.name.length * 7 + 12;
    let labelY = Math.min(bottom.value - 7, y.value(row.last.value));
    for (;;) {
      const overlap = placed.find(label => left < label.right && right > label.left && Math.abs(labelY - label.y) < 18);
      if (!overlap) break;
      labelY = overlap.y - 18;
    }
    placed.push({ left, right, y: labelY });
    labels.set(row.id, labelY);
  }
  return series.value.map(row => ({ ...row, labelX: x.value(row.last.date) + 9, labelY: labels.get(row.id)! }));
});
</script>

<template>
  <div class="election-history">
    <template v-if="series.length">
      <p class="history-unit">{{ t('widgets.history.electorate') }}</p>
      <div class="history-plot">
      <svg :viewBox="`0 0 ${width} ${height}`" role="img" :aria-label="t('widgets.history.label')"
        @mousemove="onPlotMove" @mouseleave="hoverX = null">
        <rect class="history-hover-surface" :x="L" :y="top" :width="R - L" :height="bottom - top" fill="transparent" />
        <g class="history-axis" aria-hidden="true">
          <line class="history-axis-baseline" :x1="L" :x2="R" :y1="bottom" :y2="bottom" />
          <g v-for="tick in y.ticks(5)" :key="tick">
            <text :x="L - 8" :y="y(tick)" text-anchor="end" dominant-baseline="middle">{{ tick }}%</text>
          </g>
          <text v-for="tick in xTicks" :key="+tick" :x="x(tick)" :y="bottom + 24" text-anchor="middle">{{ dateLabel(tick) }}</text>
        </g>
        <g v-for="event in annotations" :key="+event.date" class="history-event" :data-event-date="event.date.toISOString()">
          <title>{{ event.label }} · {{ dateLabel(event.date) }}</title>
          <line class="history-event-date" :x1="x(event.date)" :x2="x(event.date)" :y1="top" :y2="bottom" />
          <text :x="event.labelX" :y="event.labelY" text-anchor="end"
            :transform="`rotate(${EVENT_ANGLE} ${event.labelX} ${event.labelY})`">
            <tspan v-for="(line, index) in event.lines" :key="index" :x="event.labelX" :dy="index ? 14 : 0">{{ line + ' ' }}</tspan>
          </text>
        </g>
        <g v-for="row in labelledSeries" :key="row.id" :data-party="row.id"
          :opacity="highlighted && highlighted !== row.id ? 0.15 : 1"
          @mouseenter="active = row.id; hoverX = null" @mouseleave="active = null">
          <path class="history-line-hit" :d="row.path ?? undefined" stroke="transparent" />
          <path class="history-line" :d="row.path ?? undefined" :stroke="row.colour"
            :stroke-width="highlighted === row.id ? 3.5 : 2" />
          <circle class="history-point" :cx="x(row.last.date)" :cy="y(row.last.value)"
            :fill="row.colour" :r="highlighted === row.id ? 3.5 : 2" />
        </g>
        <g v-if="hoverX !== null" class="history-date-guide" pointer-events="none" aria-hidden="true">
          <line :x1="hoverX" :x2="hoverX" :y1="top" :y2="bottom" />
          <circle v-for="point in hoverPoints" :key="point.id" class="history-guide-point" :data-party="point.id"
            :cx="hoverX" :cy="y(point.value)" r="3.5" :fill="point.colour" />
          <text :x="hoverLabelX" :y="top - 6 - (hoverDateLines.length - 1) * 14" text-anchor="middle">
            <tspan v-for="(line, index) in hoverDateLines" :key="index" :x="hoverLabelX" :dy="index ? 14 : 0">{{ line }}</tspan>
          </text>
        </g>
      </svg>
        <button v-for="row in labelledSeries" :key="row.id" class="history-line-label" type="button" :data-party="row.id"
          :style="{ color: row.labelColour, left: `${row.labelX / width * 100}%`, top: `${row.labelY / height * 100}%`, fontSize: `${1200 / width}cqw`, opacity: highlighted && highlighted !== row.id ? 0.15 : 1 }"
          :aria-label="row.summary"
          @mouseenter="active = row.id; hoverX = null" @mouseleave="active = null"
          @mouseover.stop @mouseout.stop
          @focus="active = row.id" @blur="active = null">
          <Prose tag="span" :html="row.label" />
        </button>
      </div>
    </template>
    <p v-else class="history-empty">{{ t('widgets.history.empty') }}</p>
  </div>
</template>

<style scoped>
.election-history { width: 100%; min-width: 0; color: #3a342c; font: var(--type-compact); }
.election-history svg { display: block; width: 100%; height: auto; overflow: visible; }
.history-plot { position: relative; container-type: inline-size; }
.history-unit, .history-empty { margin: 8px 0; font: var(--type-compact); color: #6b655a; }
.history-axis { font: var(--type-compact); fill: #6b655a; }
.history-axis-baseline { stroke: #8a8273; stroke-width: 1; }
.history-event { fill: #6b655a; font: 12px var(--type-face-text, sans-serif); }
.history-event text { paint-order: stroke; stroke: var(--paper-0, #faf9f5); stroke-width: 3px; stroke-linejoin: round; }
.history-event line { stroke: #b8b8b8; stroke-width: 1; }
.history-line { fill: none; stroke-linejoin: round; stroke-linecap: round; }
.history-line-hit { fill: none; stroke-width: 12; pointer-events: stroke; }
.history-date-guide line { stroke: #8a8273; stroke-width: 1; stroke-dasharray: 3 4; opacity: .65; }
.history-guide-point { stroke: var(--paper-0, #faf9f5); stroke-width: 1.5; }
.history-date-guide text { fill: #6b655a; font: 12px var(--type-face-text, sans-serif); }
.history-point { stroke: var(--paper-0, #faf9f5); stroke-width: 0.8; }
.history-line-label { position: absolute; transform: translateY(-50%); white-space: nowrap; padding: 1px 2px;
  border: 0; background: var(--paper-0, #faf9f5); color: inherit; font: var(--type-compact); line-height: 1.15; cursor: pointer; }
.history-line-label:focus-visible { outline: 1px solid currentColor; outline-offset: 2px; }
</style>
