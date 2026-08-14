import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';
import en from '../src/locales/en.json';
import ca from '../src/locales/ca.json';

// -----------------------------------------------------------------------
// The source catalogs are read via fs, not a static import: they live
// outside ui/'s tsconfig `include` (ui/tsconfig.json only names
// src/**/*.ts and tests/**/*.ts), and — critically — a MISSING file is a
// legitimate, expected state (no override authored yet for that locale; see
// i18n.ts's initGameLocale). A plain `import` can't express "may not exist"
// without a build-time error, so this reads the same way ui/vite-plugin-
// game-assets.ts's /locales middleware does: check existsSync first.
// -----------------------------------------------------------------------

const here = path.dirname(fileURLToPath(import.meta.url));
const SOURCE_LOCALES_DIR = path.resolve(here, '../../source/locales');

function readGameCatalog(locale: 'en' | 'ca'): Record<string, unknown> {
  const file = path.join(SOURCE_LOCALES_DIR, locale, 'ui.json');
  if (!existsSync(file)) return {}; // no override shipped — NOT an error
  return JSON.parse(readFileSync(file, 'utf-8'));
}

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

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

// Mirrors vue-i18n's own mergeLocaleMessage semantics (@intlify/shared's
// deepCopy: the override wins recursively at every leaf; keys that exist
// only on the base survive untouched) — see i18n.ts. Re-implemented here
// rather than imported so this test does not depend on vue-i18n's internals
// to catch drift in the CATALOGS; ui.game-locale-merge.test.ts is what
// exercises the real vue-i18n merge path end to end.
function deepMerge(base: unknown, override: unknown): unknown {
  if (!isPlainObject(base) || !isPlainObject(override)) return override;
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(override)) {
    out[k] = k in base ? deepMerge(base[k], v) : v;
  }
  return out;
}

function valueAt(obj: unknown, keyPath: string): unknown {
  return keyPath
    .split('.')
    .reduce<unknown>((acc, k) => (acc as Record<string, unknown> | undefined)?.[k], obj);
}

function emptyLeaves(obj: unknown): string[] {
  return leafKeys(obj).filter((p) => {
    const value = valueAt(obj, p);
    return typeof value !== 'string' || value.trim() === '';
  });
}

const gameEn = readGameCatalog('en');
const gameCa = readGameCatalog('ca');
const mergedEn = deepMerge(en, gameEn);
const mergedCa = deepMerge(ca, gameCa);

describe('i18n completeness (merged: ui/ defaults + source/locales/<loc>/ui.json overrides)', () => {
  it('en and ca expose exactly the same key set after the game-layer merge', () => {
    const enKeys = leafKeys(mergedEn).sort();
    const caKeys = leafKeys(mergedCa).sort();

    // Report the two directions separately: a bare set comparison tells you
    // "they differ", these tell you which file to edit.
    const missingInCa = enKeys.filter((k) => !caKeys.includes(k));
    const missingInEn = caKeys.filter((k) => !enKeys.includes(k));
    expect(
      missingInCa,
      'keys present in the merged EN set (ui/src/locales/en.json + source/locales/en/ui.json) ' +
        'but missing from the merged CA set — add them to ui/src/locales/ca.json or ' +
        'source/locales/ca/ui.json',
    ).toEqual([]);
    expect(
      missingInEn,
      'keys present in the merged CA set (ui/src/locales/ca.json + source/locales/ca/ui.json) ' +
        'but missing from the merged EN set — add them to ui/src/locales/en.json or ' +
        'source/locales/en/ui.json',
    ).toEqual([]);
    expect(caKeys).toEqual(enKeys);
  });

  it('every key resolves to a non-empty string in both merged locales', () => {
    // CA values are allowed to equal EN (translation is a later pass), but a
    // key that exists with an empty value renders as blank UI, which is
    // worse than an untranslated string.
    expect(
      emptyLeaves(mergedEn),
      'empty or non-string values in the merged EN set (ui/src/locales/en.json + ' +
        'source/locales/en/ui.json)',
    ).toEqual([]);
    expect(
      emptyLeaves(mergedCa),
      'empty or non-string values in the merged CA set (ui/src/locales/ca.json + ' +
        'source/locales/ca/ui.json)',
    ).toEqual([]);
  });

  it('the game-owned catalogs are non-empty on their own (names the exact file to edit)', () => {
    // A stricter, single-file version of the check above: if the merged
    // check ever fails, this one narrows the blame to source/ specifically
    // rather than leaving ui/ under suspicion too.
    expect(emptyLeaves(gameEn), 'empty or non-string values in source/locales/en/ui.json').toEqual(
      [],
    );
    expect(emptyLeaves(gameCa), 'empty or non-string values in source/locales/ca/ui.json').toEqual(
      [],
    );
  });
});
