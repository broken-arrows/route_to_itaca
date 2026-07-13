import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import App from '../src/App.vue';
import { useSettingsStore } from '../src/stores/settings';
import { i18n, setLocale } from '../src/i18n';

describe('app shell', () => {
  beforeEach(() => {
    localStorage.clear();
    i18n.global.locale.value = 'en';
  });
  afterEach(() => {
    i18n.global.locale.value = 'en';
  });

  it('renders the i18n title and switches locale', async () => {
    // DebugPage's onMounted fetches game data; without a fetch stub jsdom
    // rejects the request, which logs a real console.error and flips the
    // store's loadError — assert that path is handled cleanly instead of
    // letting it leak as incidental stderr noise.
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('no network in test'))));

    const wrapper = mount(App, { global: { plugins: [createPinia(), i18n] } });
    await flushPromises();

    expect(wrapper.text()).toContain('Route to Ítaca');
    expect(errSpy).toHaveBeenCalledWith('game data load failed:', expect.any(Error));

    setLocale('ca');
    expect(i18n.global.locale.value).toBe('ca');
    expect(document.documentElement.lang).toBe('ca');
    expect(document.title).toContain('Route to Ítaca');
    setLocale('en');

    errSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  // REGRESSION (I2): the header switcher called i18n's setLocale() DIRECTLY,
  // bypassing the settings store entirely. settings.language went stale, and
  // setLocale only writes the legacy `rti:desk:locale` key — which the
  // settings blob outranks the moment anything writes one, so header language
  // changes would silently stop persisting. One source of truth: the store.
  it('the header language switcher goes through the settings store', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('no network in test'))));

    const pinia = createPinia();
    const wrapper = mount(App, { global: { plugins: [pinia, i18n] } });
    await flushPromises();
    setActivePinia(pinia);
    const settings = useSettingsStore();
    expect(settings.language).toBe('en');

    const ca = wrapper.findAll('button').find((b) => b.text() === 'CA');
    expect(ca).toBeDefined();
    await ca!.trigger('click');

    expect(settings.language).toBe('ca'); // the store is the source of truth
    expect(i18n.global.locale.value).toBe('ca'); // ...and it drove i18n
    // ...and it persisted as the settings BLOB, not just the legacy key.
    expect(JSON.parse(localStorage.getItem('rti:desk:settings')!).language).toBe('ca');

    errSpy.mockRestore();
    vi.unstubAllGlobals();
  });
});
