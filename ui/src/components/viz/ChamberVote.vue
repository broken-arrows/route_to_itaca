<script setup lang="ts">
import { computed } from 'vue';
import { useGameStore } from '../../stores/game';
import { markGlossary } from '../../glossary/mark';
import Prose from '../Prose.vue';

defineOptions({ name: 'ChamberVote' });

interface VoteParty {
  label: string;
  count?: number | null;
}

interface VoteOutcome {
  kind: 'yes' | 'abstain' | 'no';
  label: string;
  votes: number;
  parties?: VoteParty[];
}

const props = withDefaults(
  defineProps<{ outcomes?: VoteOutcome[]; q?: Record<string, unknown> }>(),
  { outcomes: () => [], q: undefined },
);

const game = useGameStore();

const outcomes = computed(() =>
  props.outcomes
    .map((outcome) => ({
      ...outcome,
      votes: Math.max(0, Number(outcome.votes) || 0),
      parties: (outcome.parties ?? []).map((party) => ({
        ...party,
        // Widget-generated party names must pass through the same glossary
        // marker as authored prose (the Desk equivalent of applyWholesome).
        html: markGlossary(String(party.label ?? ''), game.glossary),
      })),
    }))
    .filter((outcome) => outcome.votes > 0),
);

const layoutStyle = computed(() => {
  const visible = outcomes.value;
  const total = visible.reduce((sum, outcome) => sum + outcome.votes, 0);
  const abstainIndex = visible.findIndex((outcome) => outcome.kind === 'abstain');

  if (total <= 0 || abstainIndex < 0) return {};

  const votesBefore = visible
    .slice(0, abstainIndex)
    .reduce((sum, outcome) => sum + outcome.votes, 0);
  const abstainCenter =
    (votesBefore + visible[abstainIndex].votes / 2) / total;
  const equalColumnCenter = (abstainIndex + 0.5) / visible.length;
  const shift =
    (abstainCenter - equalColumnCenter) * visible.length * 100;

  return { '--chamber-vote-abstain-shift': `${shift}%` };
});
</script>

<template>
  <div class="chamber-vote" data-test="chamber-vote" :style="layoutStyle">
    <div class="chamber-vote-labels">
      <div
        v-for="outcome in outcomes"
        :key="outcome.kind"
        class="chamber-vote-label"
        :class="`chamber-vote-label--${outcome.kind}`"
      >
        {{ outcome.label }}
      </div>
    </div>
    <div class="chamber-vote-bar">
      <div
        v-for="outcome in outcomes"
        :key="outcome.kind"
        class="chamber-vote-outcome"
        :class="`chamber-vote-outcome--${outcome.kind}`"
        :style="{ flexGrow: outcome.votes }"
        :aria-label="`${outcome.label}: ${outcome.votes} votes`"
      >
        {{ outcome.votes }}
      </div>
    </div>
    <div v-if="outcomes.some((outcome) => outcome.parties.length)" class="chamber-vote-breakdowns">
      <div
        v-for="outcome in outcomes"
        :key="outcome.kind"
        class="chamber-vote-breakdown"
        :class="`chamber-vote-breakdown--${outcome.kind}`"
      >
        <ul v-if="outcome.parties.length" class="chamber-vote-parties">
          <li v-for="(party, index) in outcome.parties" :key="`${party.label}-${index}`">
            <Prose :html="party.html" tag="span" class="chamber-vote-party" />
            <span v-if="party.count != null" class="chamber-vote-party-count">
              ({{ party.count }})
            </span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chamber-vote {
  width: 100%;
  margin: 16px 0;
}
.chamber-vote-labels,
.chamber-vote-breakdowns {
  display: flex;
  width: 100%;
}
.chamber-vote-label,
.chamber-vote-breakdown {
  min-width: 0;
  flex: 1 1 0;
}
.chamber-vote-label {
  min-height: 18px;
  padding: 0 5px 5px;
  color: var(--ink-1);
  font-family: inherit;
  font-size: 1em;
  font-weight: 700;
  line-height: 1.15;
  white-space: nowrap;
}
.chamber-vote-label--abstain {
  text-align: center;
  transform: translateX(var(--chamber-vote-abstain-shift, 0));
}
.chamber-vote-label--no {
  text-align: right;
}
.chamber-vote-bar {
  display: flex;
  width: 100%;
}
.chamber-vote-outcome {
  overflow: hidden;
  min-height: 34px;
  min-width: 0;
  flex-basis: 0;
  padding: 8px 4px;
  color: #fff;
  font-family: inherit;
  font-size: 1em;
  font-weight: 800;
  line-height: 18px;
  text-align: center;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.chamber-vote-outcome--yes {
  background: #3f8f3f;
}
.chamber-vote-outcome--abstain {
  background: #b8b2a6;
  color: #2e2a22;
}
.chamber-vote-outcome--no {
  background: #b03030;
}
.chamber-vote-outcome:first-child {
  border-radius: 3px 0 0 3px;
}
.chamber-vote-outcome:last-child {
  border-radius: 0 3px 3px 0;
}
.chamber-vote-parties {
  margin: 0;
  padding: 7px 5px 0;
  list-style: none;
  color: var(--ink-1);
  font-family: inherit;
  font-size: 1em;
  line-height: 1.35;
}
.chamber-vote-breakdown--abstain .chamber-vote-parties {
  text-align: center;
}
.chamber-vote-breakdown--abstain {
  transform: translateX(var(--chamber-vote-abstain-shift, 0));
}
.chamber-vote-breakdown--no .chamber-vote-parties {
  text-align: right;
}
.chamber-vote-party {
  display: inline;
}
.chamber-vote-party-count {
  margin-left: 3px;
  color: var(--ink-2);
}
</style>
