import { describe, expect, it } from 'vitest';
import { captureLibraryUnderlay } from '../src/components/library/librarySnapshot';

describe('Library presentation snapshot', () => {
  it('keeps the underlying content but removes the previous Brief chrome', () => {
    const surface = document.createElement('div');
    surface.innerHTML = `
      <div class="desk-view">
        <aside class="clipboard-frame"><h1>Overview</h1></aside>
        <section class="desk-region" data-test="underlying-desk">Desk</section>
      </div>
    `;

    const snapshot = captureLibraryUnderlay(surface);
    const frozen = document.createElement('div');
    frozen.innerHTML = snapshot;
    const columns = frozen.querySelector('.desk-view')?.children;

    expect(snapshot).toContain('data-test="underlying-desk"');
    expect(snapshot).not.toContain('clipboard-frame');
    expect(snapshot).not.toContain('Overview');
    expect(columns).toHaveLength(2);
    expect(columns?.[0].className).toBe('library-brief-placeholder');
    expect(columns?.[1].className).toBe('desk-region');
    expect(surface.querySelector('.clipboard-frame')).not.toBeNull();
  });
});
