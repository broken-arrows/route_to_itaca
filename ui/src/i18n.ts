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
