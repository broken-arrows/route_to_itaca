import type { AppLocale } from '../i18n';

const catalogs = new Map<AppLocale, Promise<Record<string, string>>>();

export function loadContentCatalog(locale: AppLocale): Promise<Record<string, string>> {
  const cached = catalogs.get(locale);
  if (cached) return cached;

  const request = fetch(`${import.meta.env.BASE_URL}locales/${locale}/content.json`)
    .then(async (response) => {
      if (!response.ok) return {};
      const value = await response.json() as unknown;
      if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
      return Object.fromEntries(
        Object.entries(value).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
      );
    })
    .catch(() => ({}));
  catalogs.set(locale, request);
  return request;
}

export function clearContentCatalogCacheForTest(): void {
  catalogs.clear();
}
