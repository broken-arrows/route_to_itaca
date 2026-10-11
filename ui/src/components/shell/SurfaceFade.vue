<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue';
import { useSettingsStore } from '../../stores/settings';
import { useShellStore } from '../../stores/shell';

const settings = useSettingsStore();
const shell = useShellStore();
const live = ref<HTMLElement | null>(null);
const snapshot = ref<HTMLElement | null>(null);
const covering = ref(false);
let midpoint: ReturnType<typeof setTimeout> | undefined;
let completion: ReturnType<typeof setTimeout> | undefined;

function finish(): void {
  clearTimeout(midpoint);
  clearTimeout(completion);
  snapshot.value?.replaceChildren();
  covering.value = false;
  shell.transitioning = false;
}

// Called after one engine navigation, before Vue updates the old surface.
// Freeze only its presentation; the fade never invokes or restores the engine.
function start(): void {
  if (!settings.animations || shell.transitioning || !live.value || !snapshot.value) return;
  const copy = live.value.cloneNode(true) as HTMLElement;
  const originals = [live.value, ...live.value.querySelectorAll<HTMLElement>('*')];
  const copies = [copy, ...copy.querySelectorAll<HTMLElement>('*')];
  snapshot.value.replaceChildren(copy);
  originals.forEach((element, index) => {
    copies[index].scrollTop = element.scrollTop;
    copies[index].scrollLeft = element.scrollLeft;
  });
  covering.value = true;
  shell.transitioning = true;
  midpoint = setTimeout(() => { covering.value = false; }, 200);
  completion = setTimeout(finish, 400);
}

watch(() => settings.animations, (enabled) => { if (!enabled) finish(); });
onBeforeUnmount(finish);
defineExpose({ start });
</script>

<template>
  <div class="surface-frame" data-test="surface-frame">
    <div ref="live" class="surface-live" :inert="shell.transitioning || undefined">
      <slot />
    </div>
    <div ref="snapshot" class="surface-snapshot" :class="{ covering }" inert aria-hidden="true" />
    <div v-if="shell.transitioning" class="surface-black" data-test="surface-fade" aria-hidden="true" />
  </div>
</template>

<style scoped>
.surface-frame, .surface-live { position: absolute; inset: 0; }
.surface-snapshot { position: absolute; inset: 0; z-index: 100; visibility: hidden; pointer-events: none; background: var(--paper-3); }
.surface-snapshot.covering { visibility: visible; }
.surface-black { position: absolute; inset: 0; z-index: 101; background: #000; animation: surface-black 400ms linear both; }
@keyframes surface-black { 0% { opacity: 0; } 45%, 55% { opacity: 1; } 100% { opacity: 0; } }
</style>
