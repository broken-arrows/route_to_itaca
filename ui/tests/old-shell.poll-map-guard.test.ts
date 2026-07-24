import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

/**
 * The old shell's poll map must survive being rendered BEFORE the game starts.
 *
 * `mountWidgets` scans the whole document on every render, and `Q.parties` /
 * `Q.parlament_demographics` / `Q.parlament_seats` are only assigned in
 * `@start_menu` (`source/scenes/root.scene.dry:127,455`). So the widget can
 * genuinely be called with an empty Q, and it must degrade to the bare map
 * instead of throwing and taking the page down with it. Before the guard,
 * `buildVoteDataFromQ` did `demographics.forEach(...)` on `undefined`.
 *
 * Kept 2026-07-23 when the shared status scene was split (the shell went back
 * to `status.scene.dry`, whose poll map is behind a tab click rather than on
 * the default sheet). The guard is cheap and the crash is fatal, so it stays
 * defensively; the `variant` parameter it shipped alongside did NOT — that
 * existed only to let one scene serve both UIs, and its test case went with it.
 */
const CAT_POLLS = join(__dirname, '..', '..', 'out', 'html', 'cat_polls.js');

describe('old shell poll-map: missing data must not throw', () => {
  beforeAll(() => {
    // `applyWholesome` (game.js:282) and `glossary` (game.js:283) are real
    // shell globals, and index.html loads game.js BEFORE cat_polls.js — so
    // both are always present in the browser. This suite loads cat_polls.js in
    // isolation, so it must supply them. Harness gap, NOT a product defect
    // being papered over. `glossary()` is consumed as `.terms.find(...)`
    // (cat_polls.js:482-484), so an empty term list is a valid stub.
    Object.assign(window, {
      applyWholesome: (s: string) => s,
      glossary: () => ({ terms: [] as unknown[] }),
    });
    // The file is an IIFE that assigns window.initCataloniaPolls.
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    new Function(readFileSync(CAT_POLLS, 'utf8'))();
  });

  function mountInto(id: string): HTMLElement {
    const el = document.createElement('div');
    el.id = id;
    document.body.appendChild(el);
    return el;
  }

  const init = () =>
    (window as unknown as {
      initCataloniaPolls: (id: string, q: unknown, wide?: boolean) => void;
    }).initCataloniaPolls;

  it('survives a boot-time Q with no demographics and no parties', () => {
    const el = mountInto('boot-empty');
    expect(() => init()('boot-empty', {})).not.toThrow();
    // It must still have drawn something — a silent empty div would hide the
    // regression rather than fix it.
    expect(el.querySelector('#map-container')).not.toBeNull();
  });

  it('survives demographics present but parties missing, and vice versa', () => {
    mountInto('boot-half-a');
    mountInto('boot-half-b');
    expect(() =>
      init()('boot-half-a', { parlament_demographics: ['middle'] }),
    ).not.toThrow();
    expect(() => init()('boot-half-b', { parties: ['erc'] })).not.toThrow();
  });

  it('still renders the real map when the data IS present', () => {
    // Guards against "fixed" by making the widget always bail out.
    const el = mountInto('booted');
    // All THREE inputs the guard checks must be present, or this silently
    // takes the blank path and stops testing what it claims to test.
    const Q = {
      parlament_demographics: ['middle'],
      parties: ['erc'],
      parlament_seats: { barcelona: 85, girona: 17, lleida: 15, tarragona: 18 },
      parlament_barcelona_middle_pop: 1000,
      erc_parlament_barcelona_middle_support: 40,
    };
    expect(() => init()('booted', Q)).not.toThrow();
    expect(el.querySelector('#map-container')).not.toBeNull();
    expect(el.querySelector('#bars')).not.toBeNull();
  });

  it('degrades to the map alone — no bars — when data is missing', () => {
    // The complement of the test above: proves the guard takes the reduced
    // path rather than the full one, so "no throw" isn't passing vacuously.
    const el = mountInto('degraded');
    init()('degraded', {});
    expect(el.querySelector('#map-container')).not.toBeNull();
    expect(el.querySelector('#bars')).toBeNull();
  });
});
