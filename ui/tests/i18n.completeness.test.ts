import { describe, it, expect } from 'vitest';
import en from '../../source/locales/en/ui.json';
import ca from '../../source/locales/ca/ui.json';

function leaves(value: unknown, path: string[] = []): [string, unknown][] {
  if (!value || typeof value !== 'object') return [[JSON.stringify(path), value]];
  return Object.entries(value).flatMap(([key, child]) => leaves(child, [...path, key]));
}

describe('canonical UI catalogs in source/locales', () => {
  it('exposes the same keys in both locales, including Classic labels', () => {
    expect(leaves(ca).map(([key]) => key).sort()).toEqual(leaves(en).map(([key]) => key).sort());
  });
  it.each([['en', en], ['ca', ca]])('%s contains string messages', (_locale, catalog) => {
    // A locale may deliberately omit a decorative suffix. Validate the runtime
    // data shape without prescribing copy or requiring every label to be visible.
    expect(leaves(catalog).filter(([, value]) => typeof value !== 'string')).toEqual([]);
  });
});
