import { describe, it, expect, vi } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import { createPinia } from 'pinia';
import App from '../src/App.vue';
import { i18n, setLocale } from '../src/i18n';

describe('app shell', () => {
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
});
