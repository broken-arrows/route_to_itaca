import type { AppLocale } from '../i18n';
import type { CatalogEntry, ContentCatalog } from '../engine/types';

const catalogs = new Map<AppLocale, Promise<ContentCatalog>>();

function isEntry(value: unknown): value is CatalogEntry {
  if (typeof value === 'string') return true;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const node = value as Record<string, unknown>;
  const selector = node.select ?? node.plural;
  if (typeof selector !== 'string' || !/^[A-Za-z_][\w]*$/.test(selector)
    || (node.select !== undefined && node.plural !== undefined)) return false;
  if (!node.cases || typeof node.cases !== 'object' || Array.isArray(node.cases)) return false;
  const cases = node.cases as Record<string, unknown>;
  return Object.prototype.hasOwnProperty.call(cases, 'other') && Object.values(cases).every(isEntry);
}

/** Ignore unsupported catalog entries while retaining complete selector trees. */
export function parseContentCatalog(value: unknown): ContentCatalog {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, CatalogEntry] => isEntry(entry[1])));
}

export function loadContentCatalog(locale: AppLocale): Promise<ContentCatalog> {
  const cached = catalogs.get(locale);
  if (cached) return cached;

  const request = fetch(`${import.meta.env.BASE_URL}locales/${locale}/content.json`)
    .then(async (response) => {
      if (!response.ok) return {};
      const value = await response.json() as unknown;
      return parseContentCatalog(value);
    })
    .catch(() => ({}));
  catalogs.set(locale, request);
  return request;
}

export function clearContentCatalogCacheForTest(): void {
  catalogs.clear();
}
