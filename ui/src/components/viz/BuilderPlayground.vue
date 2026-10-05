<script setup lang="ts">
import { computed } from 'vue';
import { useGameStore } from '../../stores/game';
import { gameLib } from '../../game-bindings';
import { markGlossary } from '../../glossary/mark';
import Prose from '../Prose.vue';

interface BuilderRow {
  id: string; name: string; seats: number; slot?: string; variants?: string[];
  column?: number; position?: number; member?: boolean; canJoin?: boolean;
  hidden?: boolean; group?: string; isJoint?: boolean;
}
const props = defineProps<{
  rows?: BuilderRow[];
  chamber?: string;
  q: Record<string, unknown>;
}>();
const game = useGameStore();
const visibleRows = computed(() => {
  // Engine-owned rows mutate in place; the Q snapshot marks each completed edit.
  void props.q;
  return props.rows?.filter(row => !row.hidden) ?? [];
});
const jointRows = computed(() => visibleRows.value.filter(row => row.isJoint || row.slot === 'joint'));
const hasJoint = computed(() => jointRows.value.length > 0);
const groups = computed(() => jointRows.value.filter(row => row.seats > 0).map(row => ({
  slot: row.slot!, start: row.position ?? 1,
  end: Math.max(row.position ?? 1, ...visibleRows.value.filter(member => member.member && (member.group ?? 'joint') === row.slot).map(member => member.position ?? 1)) + 1,
})));
function members(slot: string) {
  return visibleRows.value.filter(row => row.member && row.slot !== 'si' && (row.group ?? 'joint') === slot);
}
function placement(row: BuilderRow) {
  if (row.column === undefined) return undefined;
  const group = groups.value.find(group => group.slot === row.slot);
  return {
    gridColumn: row.column + (hasJoint.value ? (row.column === 0 ? 1 : 2) : 0),
    gridRow: group ? `${group.start} / ${group.end}` : row.position,
  };
}
function cycle(slot: string, direction: number) {
  game.editQualities(q => gameLib.switchCoalitionBuilderParty(q, slot, direction));
}
function join(slot: string) {
  game.editQualities(q => gameLib.toggleCoalitionBuilderMember(q, slot));
}
function edit(id: string, seats: number) {
  game.editQualities(q => gameLib.editCoalitionBuilder(q, id, seats));
}
function input(id: string, event: Event, previous: number) {
  const target = event.target as HTMLInputElement;
  const seats = target.value === '' ? NaN : Number(target.value);
  if (!Number.isSafeInteger(seats) || seats < 0 || seats > Number(game.q.playground_coalitions_needed_seats)) {
    target.value = String(previous);
    return;
  }
  edit(id, seats);
}
</script>

<template>
  <div class="builder-playground" :class="{ 'has-joint': hasJoint }">
    <div v-for="group in groups" :key="group.slot" class="builder-brace" :data-group="group.slot" aria-hidden="true" :style="{ gridRow: `${group.start} / ${group.end}` }" />
    <div v-for="row in visibleRows" :key="row.slot ?? row.id" class="builder-party" :class="{ 'is-muted': row.seats === 0 || row.member, 'is-joint': row.isJoint || row.slot === 'joint' }" :data-party="row.id" :style="placement(row)">
      <div class="builder-name">
        <input v-if="row.canJoin" class="builder-checkbox" type="checkbox" :checked="row.member" :aria-label="`Include ${row.name} in the joint list`" @change="join(row.slot!)">
        <button v-if="(row.variants?.length ?? 0) > 1" class="builder-previous" type="button" :aria-label="`Previous party for ${row.name}`" @click="cycle(row.slot!, -1)">‹</button>
        <span class="builder-label"><Prose :html="markGlossary(row.name, game.glossary)" tag="span" /></span>
        <button v-if="(row.variants?.length ?? 0) > 1" class="builder-next" type="button" :aria-label="`Next party for ${row.name}`" @click="cycle(row.slot!, 1)">›</button>
      </div>
      <div class="builder-controls">
        <button type="button" :aria-label="`Remove a seat from ${row.name}`" :disabled="row.member || row.seats <= 0" @click="edit(row.id, row.seats - 1)">−</button>
        <input type="text" inputmode="numeric" pattern="[0-9]*" :disabled="row.member" :aria-label="`${row.name} seats`" :value="row.seats" @change="input(row.id, $event, row.seats)">
        <button type="button" :aria-label="`Add a seat to ${row.name}`" :disabled="row.member || row.seats >= Number(q.playground_coalitions_needed_seats)" @click="edit(row.id, row.seats + 1)">+</button>
      </div>
      <div v-if="row.isJoint || row.slot === 'joint'" class="builder-members">
        <div v-for="member in members(row.slot!)" :key="member.slot ?? member.id"><Prose :html="markGlossary(member.name, game.glossary)" /></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.builder-playground { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem .5rem; margin: 1rem 0; padding: .35rem 0; }
.builder-playground.has-joint { grid-template-columns: minmax(0, 1fr) 1rem repeat(3, minmax(0, 1fr)); }
.builder-name { display: grid; grid-template-columns: 1.1rem minmax(0, 1fr) 1.1rem; align-items: center; position: relative; min-height: 1.8rem; text-align: center; }
.builder-label { grid-column: 2; grid-row: 1; }
.builder-previous { grid-column: 1; grid-row: 1; }
.builder-next { grid-column: 3; grid-row: 1; }
.builder-checkbox { grid-column: 1; grid-row: 1; appearance: none; width: .85rem; height: .85rem; margin: 0; padding: 0; border: 0; border-radius: 3px; cursor: pointer; box-shadow: inset 0 0 0 1px currentColor; display: grid; place-items: center; position: relative; top: .9rem; z-index: 1; }
.builder-checkbox:checked::after { content: ''; width: .35rem; height: .35rem; border-radius: 50%; background: currentColor; }
.builder-members { display: flex; flex-direction: column; gap: .15em; font-size: .75em; text-align: center; }
.builder-party { display: flex; flex-direction: column; gap: .35rem; min-width: 0; }
.builder-party.is-joint { align-self: center; }
.builder-controls { display: flex; justify-content: center; align-items: center; gap: .1rem; }
button, input { font: inherit; color: inherit; background: transparent; padding: .2rem; }
button { border: 0; border-radius: 0; cursor: pointer; padding: .2rem 0; min-width: 1.1rem; box-shadow: none; }
button:focus-visible, input:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
button:disabled { opacity: .35; cursor: default; }
input[type="text"] { border: 0; border-bottom: 1px solid currentColor; border-radius: 0; width: 2.5rem; min-width: 0; text-align: center; }
.is-muted .builder-label { filter: grayscale(1); opacity: .4; }
.builder-brace { grid-column: 3; position: relative; border: 1px solid currentColor; border-radius: 7px; margin: -.35rem -.3rem; min-width: 0; pointer-events: none; }
.builder-brace::before { content: ''; position: absolute; top: 50%; right: 100%; width: 1.7rem; border-top: 1px solid currentColor; }
</style>
