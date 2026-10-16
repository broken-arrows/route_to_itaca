import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

describe('main.ts renderer isolation', () => {
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

  it('does not install a global formatter at boot', async () => {
    await import('../src/main');
    expect((window as { displayText?: unknown }).displayText).toBeUndefined();
  });

  it('is a safe no-op before any game is loaded — never throws on an empty glossary', async () => {
    await import('../src/main');
    const { createRenderer } = await import('dendrynexus-ten/lib/ui/content/html.js');
    expect(createRenderer().convertLine('CiU governs.')).toBe('CiU governs.');
  });
});
