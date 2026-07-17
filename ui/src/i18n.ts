import { createI18n } from 'vue-i18n';
import en from './locales/en.json';
import ca from './locales/ca.json';

export type AppLocale = 'en' | 'ca';
const STORAGE_KEY = 'rti:desk:locale';

function initialLocale(): AppLocale {
  const stored = typeof localStorage !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
  return stored === 'ca' ? 'ca' : 'en';
}

export const i18n = createI18n({
  legacy: false,
  locale: initialLocale(),
  fallbackLocale: 'en',
  messages: { en, ca },
});

export function setLocale(locale: AppLocale): void {
  i18n.global.locale.value = locale;
  if (typeof localStorage !== 'undefined') localStorage.setItem(STORAGE_KEY, locale);
  syncDocument();
}

export function syncDocument(): void {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = i18n.global.locale.value;
  document.title = i18n.global.t('app.title');
}

// -----------------------------------------------------------------------
// Game-owned overrides (§3.4 of the content/UI decoupling design):
// `ui/src/locales/<loc>.json` (imported above) is a FALLBACK LAYER, not a
// fence. `source/locales/<loc>/ui.json` — this game's chrome (tray captions,
// the app title, anything naming Catalan institutions) — wins on any key
// collision. Served by `vite-plugin-game-assets.ts`'s `/locales` middleware
// (dev) / `dist/locales` copy (build); see that file for why a miss must
// 404 rather than fall through to Vite's SPA fallback.
//
// A missing/404 catalog is NOT an error: it means the game ships no
// override, and the bundled `ui/src/locales/<loc>.json` defaults above
// simply stand. Fetched ONCE for both locales at boot (not per setLocale()
// call): the catalogs are tiny and pre-loading both means a later language
// switch is instant, with no network dependency on that hot path, and
// `setLocale` itself stays a plain, side-effect-free locale swap (it is
// called at every settings-store construction — see stores/settings.ts —
// so it must not fan out into a network call every time).
// -----------------------------------------------------------------------
const GAME_LOCALES: AppLocale[] = ['en', 'ca'];
let gameLocaleLoad: Promise<void> | null = null;

async function fetchAndMergeGameLocale(locale: AppLocale): Promise<void> {
  try {
    const base = import.meta.env.BASE_URL;
    const res = await fetch(`${base}locales/${locale}/ui.json`);
    if (!res.ok) return; // 404 = no override shipped for this locale — defaults stand
    const overrides = await res.json();
    i18n.global.mergeLocaleMessage(locale, overrides);
  } catch {
    // Network error, or fetch unavailable in this environment — defaults stand.
  }
}

/**
 * Boot-time load of the game's locale overrides for every supported locale.
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
