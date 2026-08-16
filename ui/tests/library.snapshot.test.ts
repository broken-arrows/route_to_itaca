import { describe, expect, it } from 'vitest';
import { captureLibraryUnderlay } from '../src/components/library/librarySnapshot';

describe('Library presentation snapshot', () => {
  it('keeps the underlying content but removes the previous Brief chrome', () => {
    const surface = document.createElement('div');
    surface.innerHTML = `
      <section class="desk-shell" data-test="underlying-desk">Desk</section>
      <aside class="clipboard-frame"><h1>Overview</h1></aside>
    `;

    const snapshot = captureLibraryUnderlay(surface);

    expect(snapshot).toContain('data-test="underlying-desk"');
    expect(snapshot).not.toContain('clipboard-frame');
    expect(snapshot).not.toContain('Overview');
    expect(surface.querySelector('.clipboard-frame')).not.toBeNull();
  });
});
