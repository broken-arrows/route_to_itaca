<script setup lang="ts">
import { useI18n } from 'vue-i18n';
import { setLocale, type AppLocale } from './i18n';
import DebugPage from './views/DebugPage.vue';

const { t, locale } = useI18n();
const locales: AppLocale[] = ['en', 'ca'];
</script>

<template>
  <header class="app-header">
    <h1>{{ t('app.title') }}</h1>
    <nav>
      <button
        v-for="l in locales"
        :key="l"
        :disabled="locale === l"
        @click="setLocale(l)"
      >
        {{ t(`lang.${l}`) }}
      </button>
    </nav>
  </header>
  <main id="page-root"><DebugPage /></main>
</template>
