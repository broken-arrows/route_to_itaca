<script setup lang="ts">
import { nextTick, ref, useId } from 'vue';

const props = withDefaults(defineProps<{
  title: string;
  meta?: string;
  animations?: boolean;
  closeLabel?: string;
  showClose?: boolean;
  labelledBy?: string;
  titleHidden?: boolean;
}>(), {
  meta: '',
  animations: true,
  closeLabel: 'Close',
  showClose: true,
  labelledBy: undefined,
  titleHidden: false,
});

const emit = defineEmits<{
  close: [];
}>();

const generatedHeadingId = `step-aside-heading-${useId()}`;
const heading = ref<HTMLElement | null>(null);
const closeButton = ref<HTMLButtonElement | null>(null);

async function focusHeading(): Promise<boolean> {
  await nextTick();
  heading.value?.focus();
  return Boolean(heading.value);
}

async function focusClose(): Promise<boolean> {
  await nextTick();
  closeButton.value?.focus();
  return Boolean(closeButton.value);
}

defineExpose({ focusHeading, focusClose });
</script>

<template>
  <section
    class="step-aside-pane"
    :class="{ 'is-animated': animations }"
    :aria-labelledby="labelledBy ?? generatedHeadingId"
    data-test="step-aside-pane"
  >
    <header class="pane-header" :class="{ 'title-hidden': titleHidden }">
      <h1
        :id="generatedHeadingId"
        ref="heading"
        class="pane-title"
        :class="{ 'visually-hidden': titleHidden }"
        tabindex="-1"
        data-test="pane-heading"
      >
        <slot name="title">{{ title }}</slot>
      </h1>
      <p v-if="meta" class="pane-meta">{{ meta }}</p>
      <slot name="header-actions" />
      <button
        v-if="showClose"
        ref="closeButton"
        type="button"
        class="pane-close"
        :aria-label="closeLabel"
        data-test="pane-close"
        @click="emit('close')"
      >
        <span aria-hidden="true">×</span>
      </button>
    </header>
    <div class="pane-rule" aria-hidden="true" />
    <div class="pane-body">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.step-aside-pane {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  height: 100%;
  padding: clamp(24px, 4vw, 56px);
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--accent-gold) 22%, var(--paper-3));
  border-radius: 5px 1px 4px 2px;
  background: var(--paper-0);
  color: var(--ink-0);
  box-shadow:
    9px 10px 0 color-mix(in srgb, var(--paper-1) 90%, transparent),
    0 18px 38px rgba(46, 42, 34, 0.18);
}

.pane-header {
  display: flex;
  align-items: center;
  gap: 20px;
  min-width: 0;
}

.pane-title {
  flex: 1 1 auto;
  min-width: 0;
  margin: 0;
  font-family: var(--font-title);
  font-size: clamp(1.55rem, 3vw, 2.8rem);
  font-weight: 800;
  letter-spacing: 0.015em;
  line-height: 1;
  text-transform: uppercase;
}

.pane-title:focus-visible {
  outline: 3px solid var(--accent-gold);
  outline-offset: 6px;
}

.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.pane-header.title-hidden { justify-content: flex-end; }

.pane-meta {
  flex: 0 1 auto;
  max-width: 44ch;
  margin: 0;
  color: color-mix(in srgb, var(--ink-0) 62%, transparent);
  font-size: clamp(0.72rem, 1vw, 0.9rem);
  line-height: 1.35;
  text-align: right;
}

.pane-close {
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  width: 40px;
  height: 40px;
  padding: 0;
  border: 2px solid color-mix(in srgb, var(--ink-0) 68%, transparent);
  border-radius: 50%;
  background: transparent;
  color: var(--ink-0);
  font: 400 1.8rem/1 var(--font-body);
  cursor: pointer;
}

.pane-close:hover { background: var(--paper-1); }

.pane-close:focus-visible {
  outline: 3px solid var(--accent-gold);
  outline-offset: 4px;
}

.pane-rule {
  flex: 0 0 auto;
  margin: clamp(18px, 2.5vh, 30px) 0;
  border-top: 2px solid var(--ink-0);
}

.pane-body {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  scrollbar-color: color-mix(in srgb, var(--ink-0) 36%, transparent) transparent;
  scrollbar-width: thin;
}

.is-animated {
  animation: pane-step-aside 260ms cubic-bezier(0.22, 0.72, 0.24, 1) both;
}

@keyframes pane-step-aside {
  from { opacity: 0; transform: translateX(-28px) rotate(-0.2deg); }
  to { opacity: 1; transform: translateX(0) rotate(0); }
}

@media (max-width: 700px) {
  .step-aside-pane { padding: 22px 18px; }
  .pane-header { align-items: flex-start; flex-wrap: wrap; gap: 12px; }
  .pane-title { flex-basis: calc(100% - 56px); }
  .pane-meta { order: 3; flex-basis: 100%; max-width: none; text-align: left; }
}

@media (prefers-reduced-motion: reduce) {
  .is-animated { animation: none; }
}
</style>
