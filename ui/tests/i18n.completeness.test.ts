import { describe, it, expect } from 'vitest';
import en from '../src/locales/en.json';
import ca from '../src/locales/ca.json';

// Flatten a nested message object into its full set of leaf key paths.
// Only leaves count: an EN leaf that CA turned into a nested object (or vice
// versa) is drift, and comparing leaf paths catches that as well as a plain
// missing/extra key.
function leafKeys(obj: unknown, prefix = ''): string[] {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) return [prefix];
  return Object.entries(obj as Record<string, unknown>).flatMap(([k, v]) =>
    leafKeys(v, prefix ? `${prefix}.${k}` : k),
  );
}

describe('i18n completeness', () => {
  it('en.json and ca.json expose exactly the same key set', () => {
    const enKeys = leafKeys(en).sort();
    const caKeys = leafKeys(ca).sort();

    // Report the two directions separately: a bare set comparison tells you
    // "they differ", these tell you which file to edit.
    const missingInCa = enKeys.filter((k) => !caKeys.includes(k));
    const missingInEn = caKeys.filter((k) => !enKeys.includes(k));
    expect(missingInCa, 'keys present in en.json but missing from ca.json').toEqual([]);
    expect(missingInEn, 'keys present in ca.json but missing from en.json').toEqual([]);
    expect(caKeys).toEqual(enKeys);
  });

  it('every key resolves to a non-empty string in both locales', () => {
    // CA values are allowed to equal EN (translation is a later pass), but a
    // key that exists with an empty value renders as blank UI, which is worse
    // than an untranslated string.
    for (const src of [
      { name: 'en', json: en },
      { name: 'ca', json: ca },
    ]) {
      const empties = leafKeys(src.json).filter((path) => {
        const value = path
          .split('.')
          .reduce<unknown>((acc, k) => (acc as Record<string, unknown>)[k], src.json);
        return typeof value !== 'string' || value.trim() === '';
      });
      expect(empties, `empty or non-string values in ${src.name}.json`).toEqual([]);
    }
  });
});
