<script setup lang="ts">
import { computed } from 'vue';
import { gameLib } from '../../game-bindings';
import type { SceneBarSpec } from '../../../../source/lib';
import { useSceneWidgetText } from './useSceneWidgetText';
const props = withDefaults(defineProps<{ title?: string; groupBelow?: number; rows?: SceneBarSpec[]; q?: Record<string, unknown> }>(), { title: '', rows: () => [], q: () => ({}) });
const views = computed(() => gameLib.sceneResultBars(props.q, props.rows, props.groupBelow));
const { html, label } = useSceneWidgetText();
</script>

<template>
  <figure class="stacked-breakdown">
    <figcaption v-if="title">{{ label(title) }}</figcaption>
    <div class="strip" aria-hidden="true"><i v-for="(row, index) in views" :key="index" :style="{ width: row.width + '%', background: row.colour }" /></div>
    <ul><li v-for="(row, index) in views" :key="index"><i :style="{ background: row.colour }" /><span v-html="html(row.label)" /><b>{{ row.value !== null ? row.value + ' · ' : '' }}{{ row.share }}%</b></li></ul>
  </figure>
</template>

<style scoped>
.stacked-breakdown { margin: 16px 0; font: var(--type-compact); font-variant-numeric: tabular-nums; }
figcaption { margin-bottom: 7px; font-weight: bold; }
.strip { display: flex; height: 20px; border-radius: 3px; overflow: hidden; background: var(--paper-4, #ded8c6); }
.strip i { flex-shrink: 0; }
ul { list-style: none; margin: 8px 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px 16px; }
li { display: flex; align-items: center; gap: 5px; }
li i { width: 9px; height: 9px; }
li b { font-weight: normal; }
</style>
