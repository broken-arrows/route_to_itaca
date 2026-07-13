<script setup lang="ts">
// Toast — spec: prototype-draw-to-dossier-NOTES.md "Motion sequences" #5
// (slide-up fade .18s). Auto-dismiss + single-slot/latest-wins timing is
// the STORE's job (desk.nudge() already owns the ~1700ms auto-clear); this
// component only owns the fade-in/out VISUAL, gated by settings.animations
// so "animations off" collapses it to instant, same reduced-motion
// contract as the rest of the desk components (there's no DELAYS.toast*
// fade entry to route through animMs() — DELAYS.toast is the dismiss
// delay, a different constant — so this reads settings directly).
// No handwritten/cursive treatment (locked decision, desk_ui_plan.md §8):
// plain typed-ink styling.
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { useSettingsStore } from '../../stores/settings';

const props = defineProps<{ textKey: string | null }>();
const { t } = useI18n();
const settings = useSettingsStore();

const fadeMs = computed(() => `${settings.animations ? 180 : 0}ms`);
</script>

<template>
  <Transition name="toast">
    <div v-if="textKey" class="toast" data-test="toast" :style="{ '--fade-ms': fadeMs }">
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
.toast-enter-active,
.toast-leave-active {
  transition: opacity var(--fade-ms, 180ms) ease, transform var(--fade-ms, 180ms) ease;
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, 8px);
}
</style>
