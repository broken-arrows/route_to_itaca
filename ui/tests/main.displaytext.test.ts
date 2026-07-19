import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// THE regression this whole task fixes: vendor/dendrynexus-ten/lib/ui/content/
// html.js:14 calls window.displayText(text) on every rendered text run, if it
// exists. The old shell has always defined it; the Vue app never had — which
// silently stripped every dossier of party colours and the entire glossary
// since phase 2 (see docs/design/LEARNINGS.md 2026-07-13, finding #1). Every
// other test in this task exercises markGlossary/the store/the components in
// isolation; per LEARNINGS's own "browser-only bugs" lesson (2026-07-13), NONE
// of that proves the real entrypoint actually wires them together. This test
// imports the real ui/src/main.ts and checks window.displayText for real.
describe('main.ts wiring: window.displayText', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '<div id="app"></div>';
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('no network in test'))));
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    delete (window as { displayText?: unknown }).displayText;
    document.body.innerHTML = '';
  });

  it('installs window.displayText at boot', async () => {
    await import('../src/main');
    expect(typeof window.displayText).toBe('function');
  });

  it('is a safe no-op before any game is loaded — never throws on an empty glossary', async () => {
    await import('../src/main');
    expect(window.displayText!('CiU governs.')).toBe('CiU governs.');
  });
});
