import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

const gameJs = readFileSync(resolve(import.meta.dirname, '../../out/html/game.js'), 'utf8');
const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');

describe('old-shell coalition tooltip', () => {
  it('shares the delegated tooltip lifecycle and renders title, summary, and formal-member logos', () => {
    document.body.replaceChildren();
    const terms = [
      {
        id: 'erc', match: ['ERC', 'erc'], colour: 'erc',
        tooltip: { title: 'ERC', img: 'img/parties/logo_erc.png' },
      },
      {
        id: 'psc', match: ['PSC', 'psc'], colour: 'psc',
        tooltip: { title: 'PSC', img: 'img/parties/logo_psc.svg' },
      },
    ];
    Object.assign(window, {
      dendryUI: {
        game: { data: { glossary: { terms } } },
        dendryEngine: { state: { qualities: {} } },
        loadSettings: vi.fn(),
        dark_mode: false,
      },
      mountWidgets: vi.fn(),
      syncTabLocks: vi.fn(),
      RTI_GAME_LIB: { allegiances: {} },
    });

    new Function(gameJs)();
    window.onload?.(new Event('load'));

    const trigger = document.createElement('span');
    trigger.className = 'generalitat-coalition';
    trigger.dataset.parties = 'erc psc';
    trigger.dataset.summary = 'Majority government - 74/68';
    trigger.textContent = 'coalition government';
    document.body.appendChild(trigger);
    trigger.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));

    const tip = document.querySelector('.mytooltiptext')!;
    expect(tip.classList.contains('visible')).toBe(true);
    expect(tip.textContent).toContain('Generalitat de Catalunya');
    expect(tip.textContent).toContain('Majority government - 74/68');
    expect(tip.textContent).not.toContain('seats');
    expect(Array.from(tip.querySelectorAll('[data-party]')).map((el) => el.getAttribute('data-party')))
      .toEqual(['erc', 'psc']);
    expect(tip.querySelector('.coalition-tooltip-logos')?.nextElementSibling)
      .toBe(tip.querySelector('.mytooltip-main-text'));
    expect(trigger.style.getPropertyValue('--mytooltip-color')).toBe('var(--erc)');
  });

  it.each([
    ['Catalonia 2015', 'election_simulation.post_election_sim', {
      ciu_parlament_showviz: 1, jxsi_parlament_showviz: 0, jxcat_parlament_showviz: 0,
      ciu_parlament_s: 50, parlament_s_majority: 68,
    }, 'ciu'],
    ['Catalonia 2017', 'election_simulation.post_election_sim', {
      ciu_parlament_showviz: 0, jxsi_parlament_showviz: 0, jxcat_parlament_showviz: 1,
      jxcat_parlament_s: 34, erc_parlament_s: 31, parlament_s_majority: 68,
    }, 'jxcat erc'],
    ['Congreso 2016', 'election_simulation.post_election_sim_congreso', {
      dl_congreso_showviz: 0, pdcat_congreso_showviz: 1, jxcat_congreso_showviz: 0,
      pp_congreso_s: 137, congreso_s_majority: 176,
    }, 'pp'],
    ['Congreso 2019', 'election_simulation.post_election_sim_congreso', {
      dl_congreso_showviz: 0, pdcat_congreso_showviz: 0, jxcat_congreso_showviz: 1,
      fr_congreso_showviz: 0, psoe_congreso_s: 120, up_congreso_s: 35,
      congreso_s_majority: 176,
    }, 'psoe up'],
  ])('does not let applyWholesome corrupt fragmented %s tooltip attributes', (_label, sceneId, q, parties) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, q);
    Object.assign(window, {
      dendryUI: {
        game: adapter.engine.game,
        dendryEngine: adapter.engine,
        loadSettings: vi.fn(),
        dark_mode: false,
      },
      mountWidgets: vi.fn(),
      syncTabLocks: vi.fn(),
      RTI_GAME_LIB: { allegiances: {} },
    });
    new Function(gameJs)();

    const host = document.createElement('div');
    host.innerHTML = adapter.renderView(sceneId);
    const marker = host.querySelector('.generalitat-coalition, .gobierno-coalition');
    expect(marker?.getAttribute('data-parties')).toBe(parties);
    expect(marker?.getAttribute('data-parties')).not.toContain('mytooltip');
    expect(marker?.querySelectorAll('.mytooltip')).toHaveLength(0);
    for (const party of parties.split(' ')) {
      const term = adapter.glossary.find((entry) =>
        entry.match.some((match) => match.toLowerCase() === party),
      );
      expect(term?.tooltip?.img, `${party} must resolve to a real logo`).toBeTruthy();
    }
  });

  it('marks both valid party labels in the real status government roster', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(window, {
      dendryUI: {
        game: adapter.engine.game,
        dendryEngine: adapter.engine,
        loadSettings: vi.fn(),
        dark_mode: false,
      },
      mountWidgets: vi.fn(),
      syncTabLocks: vi.fn(),
      RTI_GAME_LIB: { allegiances: {} },
    });
    new Function(gameJs)();

    const host = document.createElement('div');
    host.innerHTML = adapter.renderView('status.government');
    const seal = host.querySelector('img[src="img/cat_seal.svg"]');
    const officeholders = seal?.parentElement?.parentElement;
    const markedParties = Array.from(
      officeholders?.querySelectorAll('[data-term="ciu"]') ?? [],
    ).map((node) => node.textContent);

    expect(markedParties).toEqual(['CiU', 'CiU']);
    expect(officeholders?.textContent).toContain('Joana Ortega');
  });

  it('keeps an existing glossary marker opaque across streamed fragments', () => {
    Object.assign(window, {
      dendryUI: {
        game: {
          data: {
            glossary: {
              terms: [{ id: 'ciu', match: ['CiU'], colour: 'ciu' }],
            },
          },
        },
      },
      RTI_GAME_LIB: { allegiances: {} },
    });
    new Function(gameJs)();
    const applyWholesome = (window as unknown as {
      applyWholesome: (text: string) => string;
    }).applyWholesome;

    expect(applyWholesome('<span data-term="ciu">')).toBe('<span data-term="ciu">');
    expect(applyWholesome('CiU')).toBe('CiU');
    expect(applyWholesome('</span>')).toBe('</span>');
  });
});
