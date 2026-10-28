<script setup lang="ts">
import { computed } from 'vue';
import { gameLib } from '../../game-bindings';
import { useSceneWidgetText } from './useSceneWidgetText';
const props = withDefaults(defineProps<{ q?: Record<string, unknown> }>(), { q: () => ({}) });
const rows = computed(() => gameLib.proposedCabinetRows(props.q));
const { label, html } = useSceneWidgetText();
const base = import.meta.env.BASE_URL;
</script>

<template>
  <div class="presidency">
    <img :src="base + rows[0].icon" alt="">
    <div>
      <div v-for="row in rows.slice(0, 2)" :key="row.id" :class="row.id">
        <strong>{{ row.label }}</strong> <span class="person" v-html="html(row.name)" />
        <span v-if="row.party"> (<span v-html="html(row.party)" />)</span>
      </div>
    </div>
  </div>
  <dl class="cabinet-proposal">
    <div v-for="row in rows.slice(2)" :key="row.id">
      <dt><img :src="base + row.icon" alt="">{{ label(row.label) }}</dt>
      <dd><span v-html="html(row.name)" /> <span v-if="row.party">(<span v-html="html(row.party)" />)</span></dd>
    </div>
  </dl>
</template>

<style scoped>
.cabinet-proposal { display: grid; gap: 0; margin: 15px 0; font: var(--type-compact); }
.cabinet-proposal > div { display: grid; grid-template-columns: minmax(120px, 1fr) minmax(0, 1.4fr); gap: 8px; padding: 9px 0; border-bottom: 1px solid var(--paper-4, #ded8c6); }
dt { display: flex; align-items: center; gap: 9px; }
dt img { width: 21px; height: 23px; object-fit: contain; }
dd { margin: 0; align-self: center; }
.presidency { display: flex; align-items: center; gap: 1em; margin: 15px 0; font: var(--type-compact); }
.presidency > img { width: 2em; object-fit: contain; }
.president, .vp { white-space: nowrap; }
.president .person { font-weight: bold; }
.vp { font-size: .9em; }
@media (max-width: 480px) { .cabinet-proposal > div { grid-template-columns: 1fr; } dd { padding-left: 30px; } }
</style>
