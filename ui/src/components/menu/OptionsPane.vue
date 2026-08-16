<script setup lang="ts">
import { useI18n } from 'vue-i18n';
import type { AppLocale } from '../../i18n';
import { useGameStore } from '../../stores/game';
import { useSettingsStore } from '../../stores/settings';

const { t } = useI18n();
const game = useGameStore();
const settings = useSettingsStore();

async function changeLanguage(language: AppLocale): Promise<void> {
  if (settings.language === language) return;
  settings.setLanguage(language);
  await game.setContentLanguage(language);
}
</script>

<template>
  <form class="options-pane" data-test="options-pane" @submit.prevent>
    <fieldset>
      <legend>{{ t('shell.options.language') }}</legend>
      <div class="segmented">
        <label v-for="language in (['en', 'ca'] as const)" :key="language">
          <input
            type="radio"
            name="language"
            :value="language"
            :checked="settings.language === language"
            :data-test="`setting-language-${language}`"
            @change="changeLanguage(language)"
          />
          <span>{{ t(`lang.${language}`) }}</span>
        </label>
      </div>
    </fieldset>

    <label class="setting-row">
      <span>
        <strong>{{ t('shell.options.animations') }}</strong>
        <small>{{ t('shell.options.animationsDescription') }}</small>
      </span>
      <input
        type="checkbox"
        :checked="settings.animations"
        data-test="setting-animations"
        @change="settings.setAnimations(($event.target as HTMLInputElement).checked)"
      />
    </label>

    <label class="setting-row">
      <span>
        <strong>{{ t('shell.options.eventImages') }}</strong>
        <small>{{ t('shell.options.eventImagesDescription') }}</small>
      </span>
      <input
        type="checkbox"
        :checked="settings.eventImages"
        data-test="setting-event-images"
        @change="settings.setEventImages(($event.target as HTMLInputElement).checked)"
      />
    </label>

    <label class="setting-row disabled" aria-disabled="true">
      <span>
        <strong>{{ t('shell.options.music') }}</strong>
        <small>{{ t('shell.options.wip') }}</small>
      </span>
      <input type="checkbox" disabled data-test="setting-music" />
    </label>
  </form>
</template>

<style scoped>
.options-pane { display: grid; gap: 14px; max-width: 720px; }
fieldset { margin: 0; padding: 0 0 18px; border: 0; border-bottom: 1px solid color-mix(in srgb, var(--ink-0) 24%, transparent); }
legend, strong { font-family: var(--font-title); font-size: 1rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
.segmented { display: flex; gap: 8px; margin-top: 12px; }
.segmented label { position: relative; cursor: pointer; }
.segmented input { position: absolute; opacity: 0; }
.segmented span { display: block; min-width: 70px; padding: 9px 18px; border: 1px solid var(--ink-0); text-align: center; }
.segmented input:checked + span { background: var(--ink-0); color: var(--paper-0); }
.segmented input:focus-visible + span { outline: 3px solid var(--accent-gold); outline-offset: 3px; }
.setting-row { display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 14px 0; border-bottom: 1px solid color-mix(in srgb, var(--ink-0) 24%, transparent); cursor: pointer; }
.setting-row > span { display: grid; gap: 5px; }
.setting-row small { color: color-mix(in srgb, var(--ink-0) 62%, transparent); line-height: 1.35; }
.setting-row input { width: 22px; height: 22px; accent-color: var(--ink-0); }
.setting-row.disabled { cursor: not-allowed; opacity: .48; }
</style>
