import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// initGameLocale() caches its fetch behind a module-level singleton promise
// (see i18n.ts's `gameLocaleLoad`) — BY DESIGN, so a re-mount of the app
// shell never refetches. That means testing several different fetch
// scenarios against a single static import would only ever exercise the
// FIRST one. vi.resetModules() + a fresh dynamic import per test gives each
// test its own i18n module instance — fresh bundled defaults, fresh cache —
// which is what actually lets each scenario drive its own fetch.
describe('game-locale merge (source/locales/<loc>/ui.json wins over ui/ defaults)', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.resetModules();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('a key the game overrides wins over the ui/ bundled default', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({ ok: true, json: () => Promise.resolve({ app: { title: 'GAME WINS' } }) }),
      ),
    );
    const { i18n, initGameLocale } = await import('../src/i18n');

    expect(i18n.global.t('app.title')).toBe('The Desk (beta)'); // ui/'s own default, pre-merge
    await initGameLocale();
    expect(i18n.global.t('app.title')).toBe('GAME WINS'); // the game's override wins
  });

  it('a key the game does NOT mention is untouched — the ui/ default stands', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        // This override touches only `app.title` — no `debug.*` key at all.
        Promise.resolve({ ok: true, json: () => Promise.resolve({ app: { title: 'GAME WINS' } }) }),
      ),
    );
    const { i18n, initGameLocale } = await import('../src/i18n');

    await initGameLocale();
    expect(i18n.global.t('app.title')).toBe('GAME WINS');
    expect(i18n.global.t('debug.newGame')).toBe('New game'); // ui/'s own default, untouched
  });

  it('a 404 (no catalog shipped for a locale) is NOT an error — ui/ defaults stand', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: false, status: 404 })));
    const { i18n, initGameLocale } = await import('../src/i18n');

    await expect(initGameLocale()).resolves.toBeUndefined(); // does not throw/reject
    expect(i18n.global.t('app.title')).toBe('The Desk (beta)');
  });

  it('a network error fetching the catalog is NOT an error — ui/ defaults stand', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('offline'))));
    const { i18n, initGameLocale } = await import('../src/i18n');

    await expect(initGameLocale()).resolves.toBeUndefined(); // does not throw/reject
    expect(i18n.global.t('app.title')).toBe('The Desk (beta)');
  });

  it('deep-merges a nested override without discarding untouched sibling keys', async () => {
    // Only desk.tray.government is overridden; desk.tray.out (a ui/-owned
    // sibling one level down) must survive the merge untouched — this is
    // what distinguishes a real deep merge from a shallow Object.assign that
    // would wipe out the rest of `desk.tray` the moment the game overrides
    // any one key in it.
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ desk: { tray: { government: 'GAME-OWNED LABEL' } } }),
        }),
      ),
    );
    const { i18n, initGameLocale } = await import('../src/i18n');

    await initGameLocale();
    expect(i18n.global.t('desk.tray.government')).toBe('GAME-OWNED LABEL');
    expect(i18n.global.t('desk.tray.out')).toBe('OUT'); // sibling key, untouched
  });
});
