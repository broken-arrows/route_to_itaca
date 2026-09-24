<script setup lang="ts">
import PollMap from './PollMap.vue';
defineProps<{ q?: Record<string, unknown> }>();
</script>

<template>
  <PollMap :q="q" layout="wide" />
</template>
