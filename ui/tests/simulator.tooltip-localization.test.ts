import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

const root = resolve(import.meta.dirname, '../..');
const game = readFileSync(resolve(root, 'out/game.json'), 'utf8');
const catalan = JSON.parse(readFileSync(resolve(root, 'source/locales/ca/content.json'), 'utf8'));
const cases = [
  ['ciu', 'parlament', { ciu_parlament_showviz: 1, ciu_parlament_s: 50 }, 'ciu', '50/68', "<span style='color: var(--ciu); font-style: italic;'>minority government</span>"],
  ['jxsi', 'parlament', { jxsi_parlament_showviz: 1, jxsi_parlament_s: 62 }, 'jxsi', '62/68', "<span style='color: var(--jxsi); font-style: italic;'>Sovereignty government</span>"],
  ['jxcat', 'parlament', { jxcat_parlament_showviz: 1, jxcat_parlament_s: 34, erc_parlament_s: 31 }, 'jxcat erc', '65/68', "<span style='color: var(--jxcat); font-style: italic;'>sovereignty</span> <span style='color: var(--erc); font-style: italic;'>government</span>"],
  ['pp', 'congreso', { pdcat_congreso_showviz: 1, pp_congreso_s: 137 }, 'pp', '137/176', "<span style='color: #1d84ce;'>minority government</span>"],
  ['psoe_up', 'congreso', { jxcat_congreso_showviz: 1, psoe_congreso_s: 120, up_congreso_s: 35 }, 'psoe up', '155/176', "<span style='color: #da291c; font-weight: bold;'>progressist</span> <span style='color: #5a205a; font-weight: bold;'>government</span>"],
] as const;

describe('simulator structured coalition tooltips', () => {
  it.each(cases)('translates %s without touching its technical attributes or replaying arrivals', (_id, house, setup, parties, seats, labelKey) => {
    const adapter = DendryAdapter.fromJSONText(game);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      ciu_parlament_showviz: 0, jxsi_parlament_showviz: 0, jxcat_parlament_showviz: 0,
      pdcat_congreso_showviz: 0, jxcat_congreso_showviz: 0, fr_congreso_showviz: 0,
      ...setup,
    });
    const engine = adapter.engine as any;
    const english = adapter.goToScene('election_simulation.post_election_sim' + (house === 'congreso' ? '_congreso' : ''));
    const host = document.createElement('div');
    host.innerHTML = english.html;
    const selector = house === 'congreso' ? '.gobierno-coalition' : '.generalitat-coalition';
    expect(host.querySelector(selector)?.getAttribute('data-summary')).toBe('Minority government - ' + seats);
    const saved = JSON.stringify(engine.getExportableState()), rng = engine.random.getState();
    const actions = vi.spyOn(engine, '_runActions');
    try {
      host.innerHTML = adapter.refreshLocale('ca', catalan)!.html;
      const tip = host.querySelector(selector)!;
      expect(tip).not.toBeNull();
      const summaryTemplate = catalan['Minority government - {seats}/{majority}'];
      expect(typeof summaryTemplate).toBe('string');
      const [seatCount, majority] = seats.split('/');
      expect(tip.getAttribute('data-summary')).toBe(summaryTemplate
        .replace('{seats}', seatCount).replace('{majority}', majority));
      expect(tip.getAttribute('data-parties')).toBe(parties);
      // Compare to the authored catalog, not a duplicate of its current wording/capitalization.
      expect(typeof catalan[labelKey]).toBe('string');
      const expectedLabel = document.createElement('div');
      expectedLabel.innerHTML = catalan[labelKey];
      expect(tip.textContent).toBe(expectedLabel.textContent);
      expect(tip.textContent).not.toBe((() => {
        const fallback = document.createElement('div');
        fallback.innerHTML = labelKey;
        return fallback.textContent;
      })());
      expect(tip.querySelector('[style]')).not.toBeNull();
      expect(host.textContent).not.toContain("CUP's");
      expect(actions).not.toHaveBeenCalled();
      expect(JSON.stringify(engine.getExportableState())).toBe(saved);
      expect(engine.random.getState()).toEqual(rng);
      expect(adapter.refreshLocale('en', {})!.html).toBe(english.html);
    } finally { actions.mockRestore(); }
  });
});
