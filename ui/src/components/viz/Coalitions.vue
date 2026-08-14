<script setup lang="ts">
import { computed } from 'vue';
import { useGameStore } from '../../stores/game';
import { markGlossary } from '../../glossary/mark';
import Prose from '../Prose.vue';

defineOptions({ name: 'Coalitions' });

type MemberType = 'government' | 'support' | 'abstain';

interface CoalitionMember {
  party: string;
  type?: MemberType;
  label?: string;
  seats?: number;
  color?: string;
}

interface CoalitionDefinition {
  label: string;
  seatsVar: string;
  members?: CoalitionMember[];
}

const props = withDefaults(defineProps<{
  totalSeats?: number;
  majoritySeats?: number;
  majorityMarkAt?: number;
  seatsKey?: string;
  partyAliases?: Record<string, string>;
  coalitions?: CoalitionDefinition[];
  q?: Record<string, unknown>;
}>(), {
  totalSeats: 0,
  majoritySeats: 0,
  majorityMarkAt: 0.55,
  seatsKey: '{party}_s',
  partyAliases: () => ({}),
  coalitions: () => [],
  q: undefined,
});

const game = useGameStore();
const state = computed(() => props.q ?? game.q);
const TYPE_RANK: Record<MemberType, number> = { government: 0, support: 1, abstain: 2 };

function finite(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function html(value: unknown): string {
  return markGlossary(String(value ?? ''), game.glossary);
}

function displayParty(label: string): string {
  if (!props.seatsKey.includes('congreso')) return label;
  if (label === 'cs' || label === 'Cs') return 'csspa';
  if (label === 'up') return 'UP';
  return label;
}

const entries = computed(() => {
  const q = state.value;
  const majority = Math.max(0, finite(props.majoritySeats));
  const markAt = Math.min(1, Math.max(0, finite(props.majorityMarkAt)));

  return props.coalitions.flatMap((definition, definitionIndex) => {
    if (!(finite(q[definition.seatsVar]) > 0)) return [];

    const members = (definition.members ?? []).map((member, index) => {
      const party = props.partyAliases[member.party] || member.party;
      const quality = props.seatsKey.replace('{party}', party);
      return {
        ...member,
        index,
        party,
        type: member.type ?? 'government' as MemberType,
        label: member.label ?? party,
        seats: Math.max(0, finite(member.seats ?? q[quality])),
        color: member.color ?? `var(--${party})`,
      };
    }).sort((a, b) => TYPE_RANK[a.type] - TYPE_RANK[b.type] || a.index - b.index);

    let government = 0;
    let support = 0;
    let abstain = 0;
    for (const member of members) {
      if (member.type === 'government') government += member.seats;
      else if (member.type === 'support') support += member.seats;
      else abstain += member.seats;
    }
    const yes = government + support;
    const effective = Math.floor(yes + abstain / 2);
    const passing = government >= majority || yes >= majority || (abstain > 0 && effective >= majority);
    const tolerated = government < majority && yes < majority && abstain > 0 && effective >= majority;
    const shownSeats = government >= majority ? government : yes;

    let used = 0;
    const segments = members.filter((member) => member.seats > 0).map((member) => {
      const width = majority > 0
        ? Math.min(member.seats * (markAt / majority) * 100, 100 - used)
        : 0;
      used += width;
      return { ...member, width: Math.max(0, width) };
    });

    return [{
      id: `${definition.seatsVar}-${definitionIndex}`,
      labelHtml: html(definition.label),
      members: members.filter((member) => member.seats > 0).map((member) => ({
        ...member,
        labelHtml: html(displayParty(member.label)),
      })),
      segments,
      passing,
      tolerated,
      shownSeats,
      majority,
      majorityLeft: `${(markAt * 100).toFixed(3)}%`,
    }];
  });
});
</script>

<template>
  <div class="coalitions" data-test="coalitions">
    <article v-for="entry in entries" :key="entry.id" class="coalition-entry">
      <div class="coalition-heading">
        <div class="coalition-title">
          <Prose class="coalition-name" tag="strong" :html="entry.labelHtml" />:
          <template v-for="(member, index) in entry.members" :key="`${member.party}-${index}`">
            <span v-if="index" class="member-join"> + </span>
            <span class="coalition-member">
              <Prose tag="span" :html="member.labelHtml" />
              <span v-if="member.type === 'support'" class="member-role"> (support)</span>
              <span v-else-if="member.type === 'abstain'" class="member-role"> (abst.)</span>
            </span>
          </template>
        </div>
        <div class="coalition-count" :class="{ passing: entry.passing, tolerated: entry.tolerated }">
          <strong v-if="entry.passing">{{ entry.shownSeats }}</strong>
          <span v-else>{{ entry.shownSeats }}</span>
          /
          <strong v-if="!entry.passing">{{ entry.majority }}</strong>
          <span v-else>{{ entry.majority }}</span>
          <span v-if="entry.tolerated"> (tolerated)</span>
        </div>
      </div>
      <div class="coalition-track" :aria-label="`${entry.shownSeats} of ${entry.majority} seats`">
        <div class="coalition-fill">
          <div
            v-for="(segment, index) in entry.segments"
            :key="`${segment.party}-${index}`"
            class="coalition-segment"
            :class="[`segment-${segment.type}`]"
            :style="{ width: `${segment.width.toFixed(3)}%`, backgroundColor: segment.color }"
            :title="`${segment.label}: ${segment.seats} seats`"
          >{{ segment.seats }}</div>
        </div>
        <span class="majority-line" :style="{ left: entry.majorityLeft }" aria-hidden="true" />
      </div>
    </article>
  </div>
</template>

<style scoped>
.coalitions { width: 100%; margin: 14px 0; }
.coalition-entry { margin: 0 0 16px; }
.coalition-heading { display: flex; align-items: baseline; gap: 12px; margin-bottom: 5px; }
.coalition-title { min-width: 0; flex: 1 1 auto; color: var(--ink-1); line-height: 1.45; }
.coalition-name { margin-right: 3px; }
.coalition-member { white-space: nowrap; }
.member-role { color: var(--ink-3); opacity: .75; }
.coalition-count { flex: 0 0 auto; color: var(--ink-3); white-space: nowrap; }
.coalition-count.passing { color: #28733f; }
.coalition-count.tolerated { color: #286b85; }
.coalition-track { position: relative; height: 30px; overflow: hidden; border: 1px solid var(--paper-4); border-radius: 3px; background: var(--paper-2); }
.coalition-fill { position: absolute; inset: 0; display: flex; }
.coalition-segment { display: flex; min-width: 1px; flex: 0 0 auto; align-items: center; justify-content: center; overflow: hidden; color: #fff; font-size: 12px; font-weight: 800; }
.segment-support { opacity: .62; }
.segment-abstain { background-image: repeating-linear-gradient(-45deg, transparent 0, transparent 3px, rgba(0, 0, 0, .3) 3px, rgba(0, 0, 0, .3) 6px); }
.majority-line { position: absolute; z-index: 2; top: -2px; bottom: -2px; width: 2px; background: var(--ink-1); opacity: .72; }
</style>
