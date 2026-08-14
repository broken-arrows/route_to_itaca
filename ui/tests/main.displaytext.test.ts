import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

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
