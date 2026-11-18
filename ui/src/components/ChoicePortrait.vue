<script setup lang="ts">
import { ref } from 'vue';
import { imageUrl } from '../engine/image-url';
import { useSettingsStore } from '../stores/settings';

defineProps<{ path: string }>();
const settings = useSettingsStore();
const failed = ref('');
</script>

<template>
  <img v-if="settings.eventImages && failed !== path" :src="imageUrl(path)" class="choice-portrait" alt="" @error="failed = path" />
</template>

<style scoped>
.choice-portrait { width: 2.8em; height: 3.5em; object-fit: cover; vertical-align: middle; margin-inline-end: .65em; }
</style>
