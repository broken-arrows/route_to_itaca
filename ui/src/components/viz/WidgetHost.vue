<script setup lang="ts">
/**
 * Mounts the component a piece of content DECLARED, by name.
 *
 * THE INVARIANT — do not violate it, phase 3 depends on it:
 * WidgetHost is the ONLY thing in the app that parses `data-props`. A widget
 * component takes plain props + Q and has NO IDEA where they came from. That is
 * what makes the Brief's deferred sheet-composition decision cheap: a Vue sheet
 * passing props directly and a marker in content both end up at the same
 * component, unchanged. (Spec §5.3.)
 */
import { computed, onErrorCaptured, ref, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { useGameStore } from '../../stores/game';
import { WIDGETS } from './registry';
import type { WidgetName } from './registry';

const props = defineProps<{ name: string; props?: Record<string, unknown> }>();

const { q } = storeToRefs(useGameStore());
const failed = ref(false);

// The latch is per-render-attempt, not per-host-lifetime: new inputs deserve a
// fresh try (a widget that threw on one Q view-model may render fine on the
// next). Without this, one bad frame would placeholder the widget until the
// whole host remounts.
watch(
  () => [props.name, props.props] as const,
  () => {
    failed.value = false;
  },
);

const component = computed(() =>
  failed.value ? null : (WIDGETS[props.name as WidgetName] ?? null),
);

// A widget that throws must not blank the sheet around it.
onErrorCaptured((err) => {
  console.warn(`widget "${props.name}" failed to render:`, err);
  failed.value = true;
  return false;
});

// `{"configFrom": "someQKey"}` — content computed a view-model into Q and the
// marker points at it, rather than pushing it through a global. This is how
// window._cvParlement dies.
const resolved = computed<Record<string, unknown>>(() => {
  const p = { ...(props.props ?? {}) };
  const from = p.configFrom;
  if (typeof from === 'string') {
    delete p.configFrom;
    Object.assign(p, (q.value[from] as Record<string, unknown>) ?? {});
  }
  return p;
});
</script>

<template>
  <component :is="component" v-if="component" v-bind="resolved" :q="q" />
  <!-- Same striped treatment HandCard uses for missing art: never a broken div. -->
  <div v-else class="widget-placeholder" :data-widget-missing="name" />
</template>

<style scoped>
.widget-placeholder {
  min-height: 80px;
  background: repeating-linear-gradient(
    45deg,
    var(--paper-3, #eee9db),
    var(--paper-3, #eee9db) 8px,
    var(--paper-2, #f3efe4) 8px,
    var(--paper-2, #f3efe4) 16px
  );
  border: 1px solid var(--paper-4, #ded8c6);
}
</style>
