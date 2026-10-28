<script setup lang="ts">
import { computed } from 'vue';
import { gameLib } from '../../game-bindings';
import type { SceneBarSpec } from '../../../../source/lib';
import { useSceneWidgetText } from './useSceneWidgetText';
const props = withDefaults(defineProps<{ rows?: SceneBarSpec[]; q?: Record<string, unknown> }>(), { rows: () => [], q: () => ({}) });
const views = computed(() => gameLib.sceneResultBars(props.q, props.rows));
const { html, change } = useSceneWidgetText();
</script>

<template>
  <div class="result-bars">
    <div v-for="(row, index) in views" :key="index" class="result-row">
      <div class="result-heading"><span v-html="html(row.label)" /><span><b v-if="row.value !== null">{{ row.value }} · </b>{{ row.share }}%<span v-if="row.change !== null"> ({{ change(row.change, 1, '%') }})</span></span></div>
      <div class="track" aria-hidden="true"><div :style="{ width: row.width + '%', background: row.colour }" /></div>
    </div>
  </div>
</template>

<style scoped>
.result-bars { display: grid; gap: 12px; margin: 14px 0; font: var(--type-compact); font-variant-numeric: tabular-nums; }
.result-heading { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 5px 12px; margin-bottom: 5px; }
.track { background: var(--paper-4, #ded8c6); border-radius: 3px; overflow: hidden; }
.track > div { height: 12px; }
</style>
