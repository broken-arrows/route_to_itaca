import { ref } from 'vue';
import { defineStore } from 'pinia';
import { setLocale, type AppLocale } from '../i18n';

interface SettingsBlob {
  language: AppLocale;
  animations: boolean;
  eventImages: boolean;
}

function readBlob(key: string): SettingsBlob | null {
  if (typeof localStorage === 'undefined') return null;
  const raw = localStorage.getItem(key);
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

const DEFAULTS: SettingsBlob = { language: 'en', animations: true, eventImages: true };

export const useSettingsStore = defineStore('settings', () => {
  const language = ref<AppLocale>(DEFAULTS.language);
  const animations = ref<boolean>(DEFAULTS.animations);
  const eventImages = ref<boolean>(DEFAULTS.eventImages);
  let storageKey: string | null = null;

  // The reusable UI cannot know its game's localStorage namespace before the
  // manifest loads. Configuration is explicit and idempotent; it also applies
  // the persisted language through the one i18n seam. There is deliberately no
  // second loose locale key.
  function configure(storageId: string): void {
    const nextKey = `${storageId}:settings`;
    if (storageKey === nextKey) return;
    storageKey = nextKey;
    const stored = readBlob(nextKey);
    // A host may set provisional preferences before the asynchronously loaded
    // manifest reveals its namespace. A persisted blob wins when present; in
    // its absence, keep those in-memory values instead of resetting them.
    if (stored) {
      language.value = stored.language;
      animations.value = stored.animations;
      eventImages.value = stored.eventImages;
    }
    setLocale(language.value);
  }

  function persist(): void {
    if (typeof localStorage === 'undefined' || storageKey === null) return;
    const blob: SettingsBlob = {
      language: language.value,
      animations: animations.value,
      eventImages: eventImages.value,
    };
    localStorage.setItem(storageKey, JSON.stringify(blob));
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

  return {
    language,
    animations,
    eventImages,
    configure,
    setLanguage,
    setAnimations,
    setEventImages,
  };
});
