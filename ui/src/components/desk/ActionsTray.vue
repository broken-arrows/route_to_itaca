<script setup lang="ts">
// Actions tray: top-right card with advisor portraits (pinned cards).
// Geometry/colours: docs/design/reference/desk-frames.md §3 "Actions tray"
// (translucent paper card, header label, 44px ringed portraits with the
// advisor's name below). Phase-4 items drawn in the canvas but deliberately
// NOT built here: the gold "!" ready badge/ring (no readiness signal exists
// on CardView yet) and the Parlament icon + red count badge (the chamber
// surface swap is phase 4's one-move reconciliation — desk_ui_plan.md §11.4).
import { reactive } from 'vue';
import { useI18n } from 'vue-i18n';
import type { CardView } from '../../engine/types';
import Prose from '../Prose.vue';

const props = defineProps<{
  pinned: CardView[];
  disabled?: boolean;
}>();
const emit = defineEmits<{ play: [card: CardView] }>();

const { t } = useI18n();

const broken = reactive<Record<string, boolean>>({});
function imgSrc(card: CardView): string | null {
  return card.image ? `${import.meta.env.BASE_URL}${card.image}` : null;
}

function onActivate(card: CardView): void {
  if (props.disabled) return;
  emit('play', card);
}
</script>

<template>
  <div class="actions-tray" :class="{ 'is-disabled': disabled }">
    <p class="tray-label">{{ t('desk.actions.title') }}</p>
    <div class="rail">
      <div
        v-for="card in pinned"
        :key="card.id"
        class="advisor"
        data-test="pinned-card"
        role="button"
        :aria-disabled="disabled ? 'true' : 'false'"
        :tabindex="disabled ? -1 : 0"
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
</template>

<style scoped>
/* Card hexes are design-canvas literals (desk-frames.md §3). */
.actions-tray {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  background: rgba(250, 249, 245, 0.72);
  border: 1px solid rgba(90, 70, 40, 0.16);
  border-radius: 10px;
  padding: 10px 16px 9px;
  box-shadow: 0 4px 12px rgba(60, 45, 20, 0.14);
}
.actions-tray.is-disabled .advisor {
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
/* 44px circles, 2px ring. #c9c0aa is the canvas's idle ring; the gold
   ready ring (#e7c977) arrives with phase 4's readiness logic. */
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
</style>
