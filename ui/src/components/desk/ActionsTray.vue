<script setup lang="ts">
import { computed, reactive } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../../stores/game';
import type { CardView } from '../../engine/types';
import Prose from '../Prose.vue';

const props = defineProps<{
  pinned: CardView[];
  disabled?: boolean;
}>();
const emit = defineEmits<{ play: [card: CardView] }>();

const { t } = useI18n();
const game = useGameStore();

const advisors = computed(() => props.pinned.filter((card) => card.role === 'pinned-advisor'));
const parliament = computed(() => props.pinned.find((card) => card.role === 'pinned-parliament') ?? null);
const generic = computed(() => props.pinned.filter(
  (card) => card.role !== 'pinned-advisor' && card.role !== 'pinned-parliament',
));
const advisorReady = computed(() => {
  const timer = game.q.advisor_action_timer;
  return typeof timer === 'number' && timer <= 0;
});

const broken = reactive<Record<string, boolean>>({});
function imgSrc(card: CardView): string | null {
  return card.image ? `${import.meta.env.BASE_URL}${card.image}` : null;
}

function onActivate(card: CardView): void {
  if (props.disabled || card.canChoose === false) return;
  emit('play', card);
}
</script>

<template>
  <div class="actions-tray" :class="{ 'is-disabled': disabled }">
    <p class="actions-title">{{ t('desk.actions.title') }}</p>
    <div v-if="advisors.length" class="advisor-group" data-test="advisor-group">
      <p class="tray-label">
        {{ t('desk.actions.advisors') }}
        <span v-if="advisorReady" class="ready-badge" data-test="advisor-ready" aria-label="Ready">!</span>
      </p>
      <div class="rail">
      <div
        v-for="card in advisors"
        :key="card.id"
        class="advisor"
        data-test="pinned-card"
        role="button"
        :aria-disabled="disabled || card.canChoose === false ? 'true' : 'false'"
        :tabindex="disabled || card.canChoose === false ? -1 : 0"
        @click="onActivate(card)"
        @keydown.enter.prevent="onActivate(card)"
        @keydown.space.prevent="onActivate(card)"
      >
        <div class="portrait">
          <img v-if="imgSrc(card) && !broken[card.id]" :src="imgSrc(card)!" alt="" @error="broken[card.id] = true" />
          <div v-else class="portrait-placeholder" aria-hidden="true"></div>
        </div>
        <!-- card.title reaches this app through convertLine/window.displayText,
             so a title naming a glossary term (4 of the 6 real advisor cards
             do) arrives pre-wrapped in <span data-term=…> — plain {{ }} would
             print the literal markup. Same fix as OpenDossier's cover title. -->
        <p class="advisor-name"><Prose tag="span" :html="card.title" /></p>
      </div>
      </div>
    </div>

    <div v-if="generic.length" class="generic-actions" data-test="generic-actions">
      <button
        v-for="card in generic"
        :key="card.id"
        class="generic-action"
        type="button"
        :disabled="disabled || card.canChoose === false"
        data-test="pinned-card"
        @click="onActivate(card)"
      ><Prose tag="span" :html="card.title" /></button>
    </div>

    <div v-if="parliament" class="institution-divider" aria-hidden="true"></div>
    <div
      v-if="parliament"
      class="parliament-action"
      :class="{ blocked: parliament.canChoose === false }"
      data-test="parliament-action"
      role="button"
      :aria-disabled="disabled || parliament.canChoose === false ? 'true' : 'false'"
      :tabindex="disabled || parliament.canChoose === false ? -1 : 0"
      :title="parliament.subtitle"
      @click="onActivate(parliament)"
      @keydown.enter.prevent="onActivate(parliament)"
      @keydown.space.prevent="onActivate(parliament)"
    >
      <span v-if="parliament.canChoose" class="ready-badge parliament-badge" data-test="parliament-ready" aria-label="Ready">!</span>
      <div class="parliament-icon">
        <img v-if="imgSrc(parliament) && !broken[parliament.id]" :src="imgSrc(parliament)!" alt="" @error="broken[parliament.id] = true" />
        <span v-else class="chamber-mark" aria-hidden="true"><i></i><i></i><i></i></span>
      </div>
      <p class="parliament-name"><Prose tag="span" :html="parliament.title" /></p>
      <p v-if="parliament.canChoose === false && parliament.subtitle" class="blocked-reason">
        <Prose tag="span" :html="parliament.subtitle" />
      </p>
    </div>
  </div>
</template>

<style scoped>
/* Card hexes are design-canvas literals (desk-frames.md §3). */
.actions-tray {
  display: grid;
  grid-template-columns: auto auto auto;
  align-items: center;
  gap: 6px;
  background: rgba(250, 249, 245, 0.72);
  border: 1px solid rgba(90, 70, 40, 0.16);
  border-radius: 10px;
  padding: 10px 16px 9px;
  box-shadow: 0 4px 12px rgba(60, 45, 20, 0.14);
}
.actions-title {
  grid-column: 1 / -1;
  margin: 0;
  font-family: var(--font-title);
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.16em;
  color: #8a7a58;
}
.actions-tray.is-disabled .advisor,
.actions-tray.is-disabled .parliament-action {
  cursor: not-allowed;
  opacity: 0.55;
}
/* Header label — desk-frames §3: 800 7.5px letter-spacing .16em #8a7a58. */
.tray-label {
  margin: 0;
  font-family: var(--font-title);
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #8a7a58;
}
.advisor-group { display: flex; flex-direction: column; gap: 6px; }
.ready-badge {
  display: inline-grid;
  place-items: center;
  width: 19px;
  height: 19px;
  margin-left: 5px;
  border-radius: 50%;
  color: #fff;
  background: var(--accent-red);
  box-shadow: 0 1px 4px rgba(90, 20, 20, 0.32);
  font-family: var(--font-title);
  font-size: 12px;
  font-weight: 900;
  line-height: 1;
  vertical-align: middle;
}
.rail {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.advisor {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  width: 60px;
  cursor: pointer;
}
/* 44px circles, 2px idle ring. Readiness belongs to the group badge, never
   to an individual portrait. */
.portrait {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  overflow: hidden;
  border: 2px solid #c9c0aa;
  background: var(--paper-0);
  transition: transform 0.15s ease, border-color 0.15s ease;
}
.advisor:hover .portrait {
  transform: translateY(-3px);
  border-color: var(--accent-gold);
}
.actions-tray.is-disabled .advisor:hover .portrait {
  transform: none;
  border-color: #c9c0aa;
}
.portrait img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.portrait-placeholder {
  width: 100%;
  height: 100%;
  background-image: repeating-linear-gradient(
    45deg,
    var(--paper-3),
    var(--paper-3) 5px,
    var(--paper-1) 5px,
    var(--paper-1) 10px
  );
}
.advisor-name {
  margin: 0;
  font-family: var(--font-title);
  font-size: 9.5px;
  line-height: 1.15;
  text-align: center;
  color: #6b655a;
  overflow-wrap: break-word;
  max-width: 60px;
}
.generic-actions {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-width: 116px;
}
.generic-action {
  padding: 5px 8px;
  border: 1px solid #c9c0aa;
  border-radius: 3px;
  color: #6b655a;
  background: var(--paper-0);
  font-family: var(--font-title);
  font-size: 9px;
  cursor: pointer;
}
.generic-action:disabled { cursor: not-allowed; opacity: 0.55; }
.institution-divider {
  align-self: stretch;
  width: 1px;
  min-height: 66px;
  margin: 0 4px;
  background: #d3cbb8;
}
.parliament-action {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 84px;
  cursor: pointer;
}
.parliament-action.blocked { cursor: not-allowed; opacity: 0.62; }
.parliament-badge {
  position: absolute;
  z-index: 2;
  top: -6px;
  right: 6px;
  margin: 0;
}
.parliament-icon {
  display: grid;
  place-items: center;
  width: 52px;
  height: 44px;
  overflow: hidden;
  border: 3px solid var(--accent-slate);
  border-radius: 4px;
  background: var(--paper-0);
  transition: translate 0.15s ease;
}
.parliament-action:not(.blocked):hover .parliament-icon { translate: 0 -3px; }
.parliament-icon img { width: 100%; height: 100%; object-fit: cover; }
.chamber-mark { display: flex; gap: 5px; align-items: end; }
.chamber-mark i { display: block; width: 7px; height: 19px; background: var(--accent-slate); }
.chamber-mark i:nth-child(2) { height: 24px; }
.parliament-name {
  margin: 4px 0 0;
  color: var(--accent-slate);
  font-family: var(--font-title);
  font-size: 9.5px;
  font-weight: 800;
  line-height: 1.1;
  text-align: center;
  text-transform: uppercase;
}
.blocked-reason {
  max-width: 110px;
  margin: 3px -13px 0;
  color: #6b655a;
  font-family: var(--font-body);
  font-size: 8px;
  line-height: 1.15;
  text-align: center;
}
</style>
