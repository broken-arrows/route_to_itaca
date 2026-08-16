import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');
const HUB = 'status_new';
const SHEETS = ['overview', 'party', 'chamber', 'economy', 'world', 'polls'];

describe('Brief sheet scenes', () => {
  let a: DendryAdapter;
  beforeAll(() => {
    a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
  });

  it('the hub carries role: status', () => {
    expect(a.engine.game.scenes['status_new'].role).toBe('status');
  });

  it('every sheet exists and carries role: info-tab', () => {
    for (const s of SHEETS) {
      const scene = a.engine.game.scenes[`${HUB}.${s}`];
      expect(scene, `status.${s} missing`).toBeDefined();
      expect(scene.role, `status.${s} role`).toBe('info-tab');
    }
  });

  it('no sheet has an on-arrival — rendering a tab must not mutate state', () => {
    for (const s of SHEETS) {
      expect(a.engine.game.scenes[`${HUB}.${s}`].onArrival).toBeUndefined();
    }
    expect(a.engine.game.scenes['status_new'].onArrival).toBeUndefined();
  });

  it('the hub declares the six sheets followed by the authored Library entry', () => {
    // Measured against the real compiled artifact: scene.options is a plain
    // ARRAY of {id} objects (e.g. main.main_easy's options), not an object
    // with its own .options property — the brief's own test dereferenced
    // this wrongly (`(scene.options as any).options`).
    const opts = a.engine.game.scenes['status_new'].options as { id: string }[];
    expect(opts.map((o) => o.id)).toEqual([
      ...SHEETS.map((s) => `@${HUB}.${s}`),
      '@library',
    ]);
  });

  it('every sheet renders to non-empty HTML with no inline flex styling left', () => {
    for (const s of SHEETS) {
      const html = a.renderView(`${HUB}.${s}`);
      expect(html.length, `status.${s} rendered empty`).toBeGreaterThan(0);
      expect(html, `status.${s} still hand-styles`).not.toMatch(/style="display:\s*flex/);
    }
  });

  it('rendering every sheet leaves Q untouched', () => {
    const before = JSON.stringify(a.qualities);
    for (const s of SHEETS) a.renderView(`${HUB}.${s}`);
    expect(JSON.stringify(a.qualities)).toBe(before);
  });

  // Strengthened beyond the brief: "renders non-empty HTML" alone is satisfied
  // by a sheet that rendered nothing but its own title header — it does not
  // distinguish a real sheet from an empty one. Assert each sheet's HTML
  // actually carries content specific to that sheet (a widget marker or a
  // known insert), so a sheet that silently lost its body during the split
  // fails loudly instead of passing on title text alone.
  // EVERY marker a sheet carries is listed, not just one: a sheet that lost its
  // second or third widget during the split would pass a single-fragment check.
  it('every sheet renders content specific to that sheet, not just a title', () => {
    const expectedFragments: Record<string, RegExp[]> = {
      overview: [/data-widget="poll-map"/, /"deriveFrom":"standing"/],
      party: [/"deriveFrom":"factions"/, /"deriveFrom":"street"/],
      chamber: [
        /"deriveFrom":"composition"/, /"deriveFrom":"benches"/, /"deriveFrom":"cabinet"/,
      ],
      economy: [/"deriveFrom":"trails"/],
      world: [/"deriveFrom":"chancelleries"/, /"deriveFrom":"control"/],
      polls: [/data-widget="poll-map"/, /Street sentiment|Social dissent/],
    };
    for (const s of SHEETS) {
      const html = a.renderView(`${HUB}.${s}`);
      for (const fragment of expectedFragments[s]) {
        expect(html, `status.${s} missing ${fragment}`).toMatch(fragment);
      }
    }
  });

  // Task 8b: OVERVIEW's map marker must carry a `variant` prop (so the old
  // shell's poll-map handler renders the reduced/blank map instead of a full
  // duplicate of POLLS), and the variant must track `historical_mode` — the
  // game's own rule, not something the widget decides on its own.
  describe('OVERVIEW poll-map variant (Task 8b)', () => {
    function countPollMapMarkers(html: string): number {
      return (html.match(/data-widget="poll-map"/g) || []).length;
    }

    it('renders exactly one poll-map marker, carrying a variant prop', () => {
      const html = a.renderView('status_new.overview');
      expect(countPollMapMarkers(html)).toBe(1);
      expect(html).toMatch(/data-widget="poll-map"\s+data-props='\{"variant":"(compact|blank)"\}'/);
    });

    it('variant is "compact" when historical_mode is falsy, "blank" when truthy', () => {
      const before = a.qualities.historical_mode;
      try {
        a.qualities.historical_mode = 0;
        const normalHtml = a.renderView('status_new.overview');
        expect(countPollMapMarkers(normalHtml)).toBe(1);
        expect(normalHtml).toMatch(/data-widget="poll-map"\s+data-props='\{"variant":"compact"\}'/);
        expect(normalHtml).not.toContain('"variant":"blank"');

        a.qualities.historical_mode = 1;
        const historicalHtml = a.renderView('status_new.overview');
        expect(countPollMapMarkers(historicalHtml)).toBe(1);
        expect(historicalHtml).toMatch(/data-widget="poll-map"\s+data-props='\{"variant":"blank"\}'/);
        expect(historicalHtml).not.toContain('"variant":"compact"');
      } finally {
        a.qualities.historical_mode = before;
      }
    });

    it('POLLS keeps its full, prop-less poll-map marker, unaffected by historical_mode', () => {
      const before = a.qualities.historical_mode;
      try {
        a.qualities.historical_mode = 0;
        const html = a.renderView('status_new.polls');
        expect(countPollMapMarkers(html)).toBe(1);
        expect(html).toMatch(/<div id="cat-polls-widget" data-widget="poll-map"><\/div>/);
        expect(html).not.toContain('data-props');
      } finally {
        a.qualities.historical_mode = before;
      }
    });
  });
});
