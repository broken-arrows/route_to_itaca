<script setup lang="ts">

import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useSettingsStore } from '../../stores/settings';

export interface AchievementToastPayload {
  name: string;
  image: string;
  stars: number;
}

const props = defineProps<{
  textKey: string | null;
  achievement?: AchievementToastPayload | null;
}>();
const { t } = useI18n();
const settings = useSettingsStore();

const fadeMs = computed(() => `${settings.animations ? 180 : 0}ms`);

// Registry image paths are web-root-relative (`img/...`) — resolve against
// BASE_URL and hide on 404 rather than showing a broken-image icon, same
// convention as HandCard/ActionsTray/GlossaryTerm (spec §9).
const imgBroken = ref(false);
watch(
  () => props.achievement,
  () => {
    imgBroken.value = false;
  },
);
const achievementImgSrc = computed(() =>
  props.achievement?.image ? `${import.meta.env.BASE_URL}${props.achievement.image}` : null,
);
</script>

<template>
  <Transition name="toast">
    <div
      v-if="achievement"
      class="toast toast--achievement"
      data-test="toast-achievement"
      :style="{ '--fade-ms': fadeMs }"
    >
      <div class="toast-achievement-header">{{ t('desk.toast.achievementUnlocked') }}</div>
      <div class="toast-achievement-body">
        <img
          v-if="achievementImgSrc && !imgBroken"
          class="toast-achievement-image"
          :src="achievementImgSrc"
          alt=""
          @error="imgBroken = true"
        />
        <div class="toast-achievement-text">
          <div class="toast-achievement-name">{{ achievement.name }}</div>
          <div class="toast-achievement-stars">
            <span
              v-for="i in 5"
              :key="i"
              :class="i <= achievement.stars ? 'star--filled' : 'star--empty'"
              >★</span
            >
          </div>
        </div>
      </div>
    </div>
    <div v-else-if="textKey" class="toast" data-test="toast" :style="{ '--fade-ms': fadeMs }">
      {{ t(textKey) }}
    </div>
  </Transition>
</template>

<style scoped>
.toast {
  position: absolute;
  left: 50%;
  bottom: 40px;
  transform: translateX(-50%);
  background: var(--ink-0);
  color: var(--paper-0);
  font-family: var(--font-typed);
  font-size: 12px;
  letter-spacing: 0.03em;
  padding: 8px 16px;
  border-radius: 3px;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
  z-index: 100;
}
.toast--achievement {
  bottom: auto;
  top: 24px;
  left: auto;
  right: 24px;
  transform: none;
  width: 18em;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow:
    0 0 0 1px rgba(169, 130, 31, 0.4),
    0 0 14px rgba(169, 130, 31, 0.35),
    0 6px 16px rgba(0, 0, 0, 0.3);
}
.toast-achievement-header {
  font-family: var(--font-typed);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-gold);
}
.toast-achievement-body {
  display: flex;
  align-items: center;
  gap: 12px;
}
.toast-achievement-image {
  flex: 1 0 0;
  aspect-ratio: 3 / 2;
  object-fit: cover;
  border-radius: 2px;
}
.toast-achievement-text {
  flex: 2 0 0;
  min-width: 0;
}
.toast-achievement-name {
  font-family: var(--font-body);
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
}
.toast-achievement-stars .star--filled {
  color: var(--accent-gold);
}
.toast-achievement-stars .star--empty {
  color: var(--paper-3);
}
.toast-enter-active,
.toast-leave-active {
  transition: opacity var(--fade-ms, 180ms) ease, transform var(--fade-ms, 180ms) ease;
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, 8px);
}
.toast--achievement.toast-enter-from,
.toast--achievement.toast-leave-to {
  transform: translateX(calc(100% + 32px));
}
</style>
