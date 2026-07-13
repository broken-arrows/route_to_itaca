<script setup lang="ts">
import { useI18n } from 'vue-i18n';
import type { AppLocale } from './i18n';
import { useSettingsStore } from './stores/settings';
import DebugPage from './views/DebugPage.vue';
import GameView from './views/GameView.vue';

const { t, locale } = useI18n();
const settings = useSettingsStore();
const locales: AppLocale[] = ['en', 'ca'];

// Language changes go through the SETTINGS STORE, never i18n's setLocale
// directly: the store is the single source of truth (it persists the settings
// blob and drives i18n from it). Calling setLocale here would write only the
// legacy `rti:desk:locale` key, which the blob outranks the moment anything
// writes one — so header language changes would silently stop persisting.
// `locale` is still read from vue-i18n for the disabled state: it is the value
// actually being rendered.

// The debug save/load harness stays reachable at ?debug (manual QA path,
// per the Desk UI Phase 2 plan) — GameView is the default player-facing
// route. Read once at setup time; changing ?debug requires a reload, same
// as any other query-param-gated mode.
const isDebug = new URLSearchParams(window.location.search).has('debug');
</script>

<template>
  <header class="app-header">
    <h1>{{ t('app.title') }}</h1>
    <nav>
      <button
        v-for="l in locales"
        :key="l"
        :disabled="locale === l"
        @click="settings.setLanguage(l)"
      >
        {{ t(`lang.${l}`) }}
      </button>
    </nav>
  </header>
  <main id="page-root">
    <DebugPage v-if="isDebug" />
    <GameView v-else />
  </main>
</template>
