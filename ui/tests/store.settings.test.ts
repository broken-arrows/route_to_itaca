import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useSettingsStore } from '../src/stores/settings';
import { recordUnlocks, listUnlocked } from '../src/stores/achievements';
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

  it('falls back to the legacy rti:desk:locale key when no settings blob exists yet', () => {
    localStorage.setItem('rti:desk:locale', 'ca');
    const store = useSettingsStore();
    expect(store.language).toBe('ca');
    // the fallback only concerns language; the other two keep their defaults.
    expect(store.animations).toBe(true);
    expect(store.eventImages).toBe(true);
  });

  it('the legacy key is ignored once a real settings blob exists', () => {
    localStorage.setItem('rti:desk:locale', 'ca');
    const store = useSettingsStore();
    store.setAnimations(false); // forces a persisted blob with language 'ca'
    store.setLanguage('en');

    setActivePinia(createPinia());
    const restored = useSettingsStore();
    expect(restored.language).toBe('en'); // blob wins, not the stale legacy key
  });

  it('persists all three keys as one JSON blob at rti:desk:settings', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');

    const raw = localStorage.getItem('rti:desk:settings');
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw!)).toEqual({ language: 'ca', animations: true, eventImages: true });
  });

  it('every setter re-persists the whole blob, not just the changed key', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');
    store.setAnimations(false);

    expect(JSON.parse(localStorage.getItem('rti:desk:settings')!)).toEqual({
      language: 'ca',
      animations: false,
      eventImages: true,
    });

    store.setEventImages(false);
    expect(JSON.parse(localStorage.getItem('rti:desk:settings')!)).toEqual({
      language: 'ca',
      animations: false,
      eventImages: false,
    });
  });

  it('setLanguage also drives the i18n setLocale seam (legacy key updates too)', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');
    expect(localStorage.getItem('rti:desk:locale')).toBe('ca');
  });

  // REGRESSION (I2): the store is the single source of truth for the UI
  // language, but it never APPLIED its own resolved value to i18n — so a
  // persisted blob saying 'ca' booted the UI in English (i18n read only its
  // own legacy key, which the blob is allowed to outrank).
  it('applies its persisted language to i18n at creation (a ca blob boots in Catalan)', () => {
    localStorage.setItem(
      'rti:desk:settings',
      JSON.stringify({ language: 'ca', animations: true, eventImages: true }),
    );
    expect(i18n.global.locale.value).toBe('en'); // before the store exists

    const store = useSettingsStore();
    expect(store.language).toBe('ca');
    expect(i18n.global.locale.value).toBe('ca'); // the store drove i18n
  });

  it('applies the legacy-key fallback language to i18n at creation too', () => {
    localStorage.setItem('rti:desk:locale', 'ca');
    const store = useSettingsStore();
    expect(store.language).toBe('ca');
    expect(i18n.global.locale.value).toBe('ca');
  });
});

describe('achievements module (cross-save, rti:desk:achievements)', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('records newly-unlocked achievement_* qualities with both clocks and returns their ids', () => {
    const q = { achievement_foo: true, achievement_bar: false, unrelated_quality: true };
    const newly = recordUnlocks(q, { year: 2015, month: 3 });
    expect(newly).toEqual(['achievement_foo']);

    const stored = listUnlocked();
    expect(Object.keys(stored)).toEqual(['achievement_foo']);
    expect(stored.achievement_foo.inGame).toEqual({ year: 2015, month: 3 });
    expect(() => new Date(stored.achievement_foo.unlockedAt).toISOString()).not.toThrow();
  });

  it('a second call for an already-recorded id returns [] and does not overwrite it', () => {
    const q = { achievement_foo: true };
    recordUnlocks(q, { year: 2015, month: 3 });
    const second = recordUnlocks(q, { year: 2015, month: 4 });
    expect(second).toEqual([]);
    expect(listUnlocked().achievement_foo.inGame).toEqual({ year: 2015, month: 3 }); // unchanged
  });

  it('respects a pre-existing stored set seeded before recordUnlocks is ever called', () => {
    localStorage.setItem(
      'rti:desk:achievements',
      JSON.stringify({
        achievement_preexisting: {
          unlockedAt: '2020-01-01T00:00:00.000Z',
          inGame: { year: 2011, month: 1 },
        },
      }),
    );
    const newly = recordUnlocks(
      { achievement_preexisting: true, achievement_new: true },
      { year: 2016, month: 6 },
    );
    expect(newly).toEqual(['achievement_new']);
    const stored = listUnlocked();
    expect(stored.achievement_preexisting.inGame).toEqual({ year: 2011, month: 1 }); // untouched
    expect(stored.achievement_new.inGame).toEqual({ year: 2016, month: 6 });
  });
});
