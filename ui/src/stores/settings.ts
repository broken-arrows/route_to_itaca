import { ref } from 'vue';
import { defineStore } from 'pinia';
import { setLocale, type AppLocale } from '../i18n';

const STORAGE_KEY = 'rti:desk:settings';
// Phase-1 key, still written by i18n.ts's setLocale. Only consulted when no
// rti:desk:settings blob exists yet — once the blob exists it always wins.
const LEGACY_LOCALE_KEY = 'rti:desk:locale';

interface SettingsBlob {
  language: AppLocale;
  animations: boolean;
  eventImages: boolean;
}

function readBlob(): SettingsBlob | null {
  if (typeof localStorage === 'undefined') return null;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<SettingsBlob>;
    return {
      language: parsed.language === 'ca' ? 'ca' : 'en',
      animations: parsed.animations !== false,
      eventImages: parsed.eventImages !== false,
    };
  } catch {
    return null;
  }
}

function initialSettings(): SettingsBlob {
  const blob = readBlob();
  if (blob) return blob;
  const legacyLocale =
    typeof localStorage !== 'undefined' ? localStorage.getItem(LEGACY_LOCALE_KEY) : null;
  return {
    language: legacyLocale === 'ca' ? 'ca' : 'en',
    animations: true,
    eventImages: true,
  };
}

export const useSettingsStore = defineStore('settings', () => {
  const initial = initialSettings();
  const language = ref<AppLocale>(initial.language);
  const animations = ref<boolean>(initial.animations);
  const eventImages = ref<boolean>(initial.eventImages);

  // This store is the single source of truth for the UI language, so it must
  // APPLY its resolved initial value, not just hold it: i18n.ts boots from its
  // own legacy `rti:desk:locale` key, which the settings blob is allowed to
  // outrank (see readBlob) — so without this a blob saying 'ca' booted the UI
  // in English. Idempotent when the two already agree.
  setLocale(language.value);

  function persist(): void {
    if (typeof localStorage === 'undefined') return;
    const blob: SettingsBlob = {
      language: language.value,
      animations: animations.value,
      eventImages: eventImages.value,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(blob));
  }

  function setLanguage(l: AppLocale): void {
    language.value = l;
    setLocale(l);
    persist();
  }
  function setAnimations(b: boolean): void {
    animations.value = b;
    persist();
  }
  function setEventImages(b: boolean): void {
    eventImages.value = b;
    persist();
  }

  return { language, animations, eventImages, setLanguage, setAnimations, setEventImages };
});
