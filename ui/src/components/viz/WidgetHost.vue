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
import { gameLib } from '../../game-bindings';

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

// Two ways content can point at data, and the widget can tell them apart from
// neither: it just receives props.
//   configFrom — content computed real STATE into Q; the marker names the key.
//   deriveFrom — a pure VIEW the game's lib builds on demand (spec §3.2).
//                Never persisted, so it can never go stale in a save.
const resolved = computed<Record<string, unknown>>(() => {
  const p = { ...(props.props ?? {}) };
  const from = p.configFrom;
  if (typeof from === 'string') {
    delete p.configFrom;
    Object.assign(p, (q.value[from] as Record<string, unknown>) ?? {});
  }
  const derive = p.deriveFrom;
  if (typeof derive === 'string') {
    delete p.deriveFrom;
    // `GameLib` (source/lib/index.d.ts) doesn't declare `brief` yet — that
    // module ships in Wave 2 (tasks 3-5, source/lib/brief.js). Double-cast
    // through `unknown` rather than widen the shared interface here on a
    // guess at its eventual shape; `builder?.[derive]` below is what makes
    // this safe when the field is genuinely absent, pre-Wave-2 or on a typo.
    const builder = (gameLib as unknown as Record<string, unknown>).brief as
      | Record<string, (q: Record<string, unknown>) => unknown[]>
      | undefined;
    const fn = builder?.[derive];
    if (typeof fn !== 'function') {
      // Unknown derivation: placeholder, never a throw. The audit guard is what
      // is supposed to catch this at build time; this is the runtime backstop.
      console.warn(`widget "${props.name}": unknown deriveFrom "${derive}"`);
      failed.value = true;
      return p;
    }
    // `onErrorCaptured` below CANNOT see a throw from this call: it walks up
    // from `instance.parent` (runtime-core's `handleError`), which only ever
    // reaches DESCENDANT components' errors, never the throwing instance's
    // own. `fn` is game-lib code invoked directly in our own computed, not a
    // child component — so without this try/catch a bad derivation would
    // escape WidgetHost entirely and blank the whole sheet. Guard it here,
    // same placeholder path as the unknown-name branch above.
    try {
      p.rows = fn(q.value);
    } catch (err) {
      console.warn(`widget "${props.name}": deriveFrom "${derive}" threw:`, err);
      failed.value = true;
      return p;
    }
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
