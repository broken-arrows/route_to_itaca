<template>
  <div class="responsive-viewport">
    <slot />
  </div>
</template>

<style scoped>
/*
 * The player UI owns the viewport. Geometry is resolved by the surfaces
 * inside this shell; there is deliberately no global scale transform.
 */
.responsive-viewport {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100dvh;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  background: var(--paper-3);
  isolation: isolate;
}
</style>
