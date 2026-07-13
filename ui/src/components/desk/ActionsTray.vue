<script setup lang="ts">
// Actions tray: top-right, advisor portraits (pinned cards). Spec:
// prototype-draw-to-dossier-NOTES.md "Other desk objects" — "Phase 2:
// minimal version — clickable, no readiness logic": the gold "!" ready
// badge is deliberately NOT built here (no readiness signal exists on
// CardView yet; inventing one would be flavour behaviour, not styling).
import { reactive } from 'vue';
import { useI18n } from 'vue-i18n';
import type { CardView } from '../../engine/types';

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
        class="portrait"
        data-test="pinned-card"
        role="button"
        :aria-disabled="disabled ? 'true' : 'false'"
        :tabindex="disabled ? -1 : 0"
        @click="onActivate(card)"
        @keydown.enter.prevent="onActivate(card)"
        @keydown.space.prevent="onActivate(card)"
      >
        <img v-if="imgSrc(card) && !broken[card.id]" :src="imgSrc(card)!" alt="" @error="broken[card.id] = true" />
        <div v-else class="portrait-placeholder" aria-hidden="true"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.actions-tray {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}
.actions-tray.is-disabled .portrait {
  cursor: not-allowed;
  opacity: 0.55;
}
.tray-label {
  margin: 0;
  font-family: var(--font-typed);
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--ink-0);
  opacity: 0.75;
}
.rail {
  display: flex;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 20px;
  background: var(--paper-3);
  box-shadow: inset 0 2px 5px rgba(46, 42, 34, 0.22);
}
.portrait {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  overflow: hidden;
  border: 2px solid var(--accent-gold);
  cursor: pointer;
  background: var(--paper-0);
  transition: transform 0.15s ease;
}
.portrait:hover {
  transform: translateY(-3px);
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
</style>
