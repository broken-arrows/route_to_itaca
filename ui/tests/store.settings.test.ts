import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useSettingsStore } from '../src/stores/settings';
import { i18n } from '../src/i18n';

describe('settings store', () => {
  beforeEach(() => {
    localStorage.clear();
    // i18n is a module singleton: pin it back to the default so the
    // "applies its language at creation" assertions below are real.
    i18n.global.locale.value = 'en';
    setActivePinia(createPinia());
  });
  afterEach(() => {
    i18n.global.locale.value = 'en';
  });

  it('defaults to en/animations-on/eventImages-on when nothing is stored', () => {
    const store = useSettingsStore();
    expect(store.language).toBe('en');
    expect(store.animations).toBe(true);
    expect(store.eventImages).toBe(true);
  });

  it('round-trips the settings blob across a fresh pinia instance', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');
    store.setAnimations(false);
    store.setEventImages(false);

    setActivePinia(createPinia());
    const restored = useSettingsStore();
    expect(restored.language).toBe('ca');
    expect(restored.animations).toBe(false);
    expect(restored.eventImages).toBe(false);
  });

  it('falls back to a loose locale key (dnt:locale) when no settings blob exists yet', () => {
    localStorage.setItem('dnt:locale', 'ca');
    const store = useSettingsStore();
    expect(store.language).toBe('ca');
    // the fallback only concerns language; the other two keep their defaults.
    expect(store.animations).toBe(true);
    expect(store.eventImages).toBe(true);
  });

  it('falls back to the pre-rename rti:desk:locale key too', () => {
    localStorage.setItem('rti:desk:locale', 'ca');
    const store = useSettingsStore();
    expect(store.language).toBe('ca');
  });

  it('reads a pre-rename rti:desk:settings blob when no dnt:settings blob exists', () => {
    localStorage.setItem(
      'rti:desk:settings',
      JSON.stringify({ language: 'ca', animations: false, eventImages: true }),
    );
    const store = useSettingsStore();
    expect(store.language).toBe('ca');
    expect(store.animations).toBe(false);
  });

  it('the loose locale key is ignored once a real settings blob exists', () => {
    localStorage.setItem('dnt:locale', 'ca');
    const store = useSettingsStore();
    store.setAnimations(false); // forces a persisted blob with language 'ca'
    store.setLanguage('en');

    setActivePinia(createPinia());
    const restored = useSettingsStore();
    expect(restored.language).toBe('en'); // blob wins, not the stale loose key
  });

  it('persists all three keys as one JSON blob at dnt:settings', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');

    const raw = localStorage.getItem('dnt:settings');
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw!)).toEqual({ language: 'ca', animations: true, eventImages: true });
  });

  it('every setter re-persists the whole blob, not just the changed key', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');
    store.setAnimations(false);

    expect(JSON.parse(localStorage.getItem('dnt:settings')!)).toEqual({
      language: 'ca',
      animations: false,
      eventImages: true,
    });

    store.setEventImages(false);
    expect(JSON.parse(localStorage.getItem('dnt:settings')!)).toEqual({
      language: 'ca',
      animations: false,
      eventImages: false,
    });
  });

  it('setLanguage also drives the i18n setLocale seam (loose locale key updates too)', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');
    expect(localStorage.getItem('dnt:locale')).toBe('ca');
  });

  // REGRESSION (I2): the store is the single source of truth for the UI
  // language, but it never APPLIED its own resolved value to i18n — so a
  // persisted blob saying 'ca' booted the UI in English (i18n read only its
  // own legacy key, which the blob is allowed to outrank).
  it('applies its persisted language to i18n at creation (a ca blob boots in Catalan)', () => {
    localStorage.setItem(
      'dnt:settings',
      JSON.stringify({ language: 'ca', animations: true, eventImages: true }),
    );
    expect(i18n.global.locale.value).toBe('en'); // before the store exists

    const store = useSettingsStore();
    expect(store.language).toBe('ca');
    expect(i18n.global.locale.value).toBe('ca'); // the store drove i18n
  });

  it('applies the loose-key fallback language to i18n at creation too', () => {
    localStorage.setItem('dnt:locale', 'ca');
    const store = useSettingsStore();
    expect(store.language).toBe('ca');
    expect(i18n.global.locale.value).toBe('ca');
  });
});

// The old "achievements module (cross-save, rti:desk:achievements)" describe
// block that used to live here tested `recordUnlocks`/`listUnlocked` — a
// SECOND, duplicate achievement ledger next to the engine's own
// (localStorage[game.title + '_achievements'], engine.js:1141). Deleted in
// phase 2.5 Task 8, per the task brief's Step 4 (see
// docs/design/LEARNINGS.md 2026-07-13 §7-8: the engine already IS the
// cross-save ledger — a second one duplicated it and had gotten its own two
// qualities backwards). The replacement is a stateless diff, not a ledger —
// see ui/tests/achievements.test.ts (the pure `newlyUnlocked` predicate) and
// ui/tests/store.desk.achievements.test.ts (the real wiring, through the
// desk store).
