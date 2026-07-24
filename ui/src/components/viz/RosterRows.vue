<script setup lang="ts">
import { computed } from 'vue';
import { useBand } from '../brief/useBand';
import { usePartyLogo } from './usePartyLogo';

defineOptions({ name: 'RosterRows' });

interface RosterRow {
  id: string;
  label: string;
  value: number;
  stamp?: number | null;
  stampDisplay?: string | null;
  valueDisplay?: string | null;
  subtitle?: string;
  isPlayer?: boolean;
  flag?: string;
}

const props = withDefaults(
  defineProps<{ rows?: RosterRow[]; q?: Record<string, unknown> }>(),
  { rows: () => [], q: undefined },
);
const { band } = useBand();
const partyLogo = usePartyLogo();
const humanise = (s: string) => s.replace(/_/g, ' ');

const views = computed(() =>
  props.rows.map((row) => {
    const relation = row.isPlayer
      ? { band: 'you', label: 'you' }
      : band(row.stamp, row.stampDisplay);
    const stance = band(row.value, row.valueDisplay);
    return {
      ...row,
      relation,
      stance,
      stamp: relation.label || (stance.band ? humanise(stance.band) : ''),
      detail: row.subtitle || stance.label,
      logo: row.flag ? null : partyLogo(row.id),
    };
  }),
);
</script>

<template>
  <div class="roster-rows" data-test="roster-rows">
    <div v-for="row in views" :key="row.id" class="roster-row">
      <img v-if="row.flag" class="roster-flag" :src="row.flag" alt="" />
      <img v-else-if="row.logo" class="roster-logo" :src="row.logo" :alt="`${row.label} logo`" />
      <span v-else class="roster-logo-missing" aria-hidden="true" />
      <span class="roster-copy">
        <span class="roster-main">
          <b>{{ row.label }}</b>
          <span v-if="!row.flag">· {{ row.value }}</span>
          <span
            v-if="row.stamp"
            class="brief-stamp roster-stamp"
            :data-band="row.relation.band || row.stance.band"
          >{{ row.stamp }}</span>
        </span>
        <i v-if="row.detail">{{ row.detail }}</i>
      </span>
    </div>
  </div>
</template>

<style scoped>
.roster-rows { display: grid; gap: 9px; }
.roster-row {
  display: grid;
  grid-template-columns: 27px minmax(0, 1fr);
  align-items: center;
  gap: 9px;
}
.roster-logo { width: 27px; height: 27px; object-fit: contain; }
.roster-logo-missing { width: 27px; height: 1px; background: #d8d0bf; }
.roster-flag { width: 26px; height: 18px; object-fit: cover; border: 1px solid #e0d9c8; }
.roster-copy { min-width: 0; }
.roster-main { display: flex; align-items: center; gap: 5px; min-width: 0; }
.roster-main b { font: 700 13px/1.15 var(--font-news); color: #2e2a22; }
.roster-main > span:not(.roster-stamp) { font: 600 11px/1 var(--font-title); color: #6b655a; }
.roster-stamp { margin-left: auto; }
.roster-copy > i {
  display: block;
  overflow: hidden;
  margin-top: 2px;
  color: #8a8273;
  font: italic 10.5px/1.25 var(--font-body);
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
