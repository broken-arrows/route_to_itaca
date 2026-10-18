import { createI18n } from 'vue-i18n';
import en from '../../source/locales/en/ui.json';
import ca from '../../source/locales/ca/ui.json';

export type AppLocale = 'en' | 'ca';
export const i18n = createI18n({
  legacy: false,
  // Persistence belongs to the settings store's manifest-scoped blob. i18n
  // starts from the language-independent default, then settings.configure()
  // applies the saved locale once game.info.ifid is available.
  locale: 'en',
  fallbackLocale: 'en',
  messages: { en, ca },
});

export function setLocale(locale: AppLocale): void {
  i18n.global.locale.value = locale;
  syncDocument();
}

export function syncDocument(): void {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = i18n.global.locale.value;
  document.title = i18n.global.t('app.title');
}

// Both the bundled startup messages and published runtime catalogs come from
// source/locales. Fetching at boot allows updated published catalogs to win;
// a network failure retains the bundled copy of those same canonical files.
const GAME_LOCALES: AppLocale[] = ['en', 'ca'];
let gameLocaleLoad: Promise<void> | null = null;

async function fetchAndMergeGameLocale(locale: AppLocale): Promise<void> {
  try {
    const base = import.meta.env.BASE_URL;
    const res = await fetch(`${base}locales/${locale}/ui.json`);
    if (!res.ok) return; // 404 = no catalog shipped for this locale — bundled messages stand
    const messages = await res.json();
    i18n.global.mergeLocaleMessage(locale, messages);
  } catch {
    // Network error, or fetch unavailable in this environment — bundled messages stand.
  }
}

/**
 * Boot-time load of the game's locale messages for every supported locale.
 * Idempotent: repeated calls (e.g. re-mounting the app shell) return the same
 * in-flight/settled promise instead of re-fetching.
 */
export function initGameLocale(): Promise<void> {
  if (!gameLocaleLoad) {
    gameLocaleLoad = Promise.all(GAME_LOCALES.map(fetchAndMergeGameLocale)).then(() => {
      syncDocument(); // the active locale's title may have just changed
    });
  }
  return gameLocaleLoad;
}
