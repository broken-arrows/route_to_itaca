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
  /* A step darker than the Brief column's --paper-3 so the letterboxed
     stage reads as a surface instead of bleeding into the page. */
  background: #e6dfd0;
}
.stage {
  flex: none;
  position: relative;
  overflow: hidden;
  transform-origin: center center;
  /* The canvas frames carry their own 1px #ddd8cc frame border
     (desk-frames §1); the soft shadow separates the desk from the
     letterbox at window sizes that don't fill the viewport. */
  border: 1px solid #ddd8cc;
  box-shadow: 0 16px 44px rgba(60, 45, 20, 0.16);
}
</style>
