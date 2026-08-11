import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useSettingsStore } from '../src/stores/settings';
import { i18n } from '../src/i18n';

describe('settings store', () => {
  beforeEach(() => {
    localStorage.clear();
    i18n.global.locale.value = 'en';
    setActivePinia(createPinia());
  });

  afterEach(() => {
    i18n.global.locale.value = 'en';
  });

  it('uses safe defaults before the manifest storage id is configured', () => {
    const store = useSettingsStore();
    expect(store.language).toBe('en');
    expect(store.animations).toBe(true);
    expect(store.eventImages).toBe(true);
  });

  it('loads and applies the manifest-scoped settings blob on configuration', () => {
    localStorage.setItem(
      'rti:settings',
      JSON.stringify({ language: 'ca', animations: false, eventImages: true }),
    );
    const store = useSettingsStore();
    store.configure('rti');

    expect(store.language).toBe('ca');
    expect(store.animations).toBe(false);
    expect(store.eventImages).toBe(true);
    expect(i18n.global.locale.value).toBe('ca');
  });

  it('round-trips one settings blob across a fresh pinia instance', () => {
    const store = useSettingsStore();
    store.configure('rti');
    store.setLanguage('ca');
    store.setAnimations(false);
    store.setEventImages(false);

    setActivePinia(createPinia());
    const restored = useSettingsStore();
    restored.configure('rti');
    expect(restored.language).toBe('ca');
    expect(restored.animations).toBe(false);
    expect(restored.eventImages).toBe(false);
  });

  it('persists every setter as the complete rti:settings blob', () => {
    const store = useSettingsStore();
    store.configure('rti');
    store.setLanguage('ca');
    store.setAnimations(false);

    expect(JSON.parse(localStorage.getItem('rti:settings')!)).toEqual({
      language: 'ca',
      animations: false,
      eventImages: true,
    });

    store.setEventImages(false);
    expect(JSON.parse(localStorage.getItem('rti:settings')!)).toEqual({
      language: 'ca',
      animations: false,
      eventImages: false,
    });
  });

  it('does not write an origin-global setting before configuration', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');
    expect(localStorage.length).toBe(0);
    expect(i18n.global.locale.value).toBe('ca');
  });

  it('keeps provisional in-memory preferences when the scoped blob is absent', () => {
    const store = useSettingsStore();
    store.setLanguage('ca');
    store.setAnimations(false);
    store.configure('rti');

    expect(store.language).toBe('ca');
    expect(store.animations).toBe(false);
    expect(localStorage.length).toBe(0);
  });

  it('ignores every provisional and legacy settings or locale key', () => {
    localStorage.setItem(
      'dnt:settings',
      JSON.stringify({ language: 'ca', animations: false, eventImages: false }),
    );
    localStorage.setItem('dnt:locale', 'ca');
    localStorage.setItem('rti:desk:locale', 'ca');
    const store = useSettingsStore();
    store.configure('rti');

    expect(store.language).toBe('en');
    expect(store.animations).toBe(true);
    expect(store.eventImages).toBe(true);
  });

  it('keeps settings for different storage namespaces isolated', () => {
    localStorage.setItem(
      'other:settings',
      JSON.stringify({ language: 'ca', animations: false, eventImages: false }),
    );
    const store = useSettingsStore();
    store.configure('rti');
    expect(store.language).toBe('en');
  });
});
