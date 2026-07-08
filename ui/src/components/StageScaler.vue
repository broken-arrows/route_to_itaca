<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';
import { computeScale, STAGE_W, STAGE_H } from '../engine/stage';

const scale = ref(1);

function update() {
  scale.value = computeScale(window.innerWidth, window.innerHeight);
}

onMounted(() => {
  update();
  window.addEventListener('resize', update);
});
onUnmounted(() => window.removeEventListener('resize', update));
</script>

<template>
  <div class="stage-viewport">
    <div
      class="stage"
      :style="{
        width: STAGE_W + 'px',
        height: STAGE_H + 'px',
        transform: `scale(${scale})`,
      }"
    >
      <slot />
    </div>
  </div>
</template>

<style scoped>
.stage-viewport {
  position: fixed;
  inset: 0;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--paper-3);
}
.stage {
  flex: none;
  position: relative;
  overflow: hidden;
  transform-origin: center center;
}
</style>
