import { ref } from 'vue';
import { defineStore } from 'pinia';
import { setLocale, type AppLocale } from '../i18n';

// `dnt:` prefix (the library, not the game) — see i18n.ts's STORAGE_KEY
// comment for the naming rule and the phase-5 per-game discriminator plan.
const STORAGE_KEY = 'dnt:settings';
// Pre-rename blob key (phases 1–2.5). Read-only fallback; never written again.
const LEGACY_STORAGE_KEY = 'rti:desk:settings';
// Phase-1 locale keys, still written by i18n.ts's setLocale (current name) and
// possibly present under the pre-rename name. Only consulted when no settings
// blob exists yet — once a blob exists it always wins.
const LOCALE_KEYS = ['dnt:locale', 'rti:desk:locale'];

interface SettingsBlob {
  language: AppLocale;
  animations: boolean;
  eventImages: boolean;
}

function readBlob(): SettingsBlob | null {
  if (typeof localStorage === 'undefined') return null;
  const raw = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY);
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
  const looseLocale =
    typeof localStorage !== 'undefined'
      ? LOCALE_KEYS.map((k) => localStorage.getItem(k)).find((v) => v !== null) ?? null
      : null;
  return {
    language: looseLocale === 'ca' ? 'ca' : 'en',
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
  // own loose locale key (`dnt:locale`, or the pre-rename `rti:desk:locale`),
  // which the settings blob is allowed to outrank (see readBlob) — so without
  // this a blob saying 'ca' booted the UI in English. Idempotent when the two
  // already agree.
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
