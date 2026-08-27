import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { convertLine } from 'dendrynexus-ten/lib/ui/content/html.js';
import { DendryAdapter } from '../src/engine/adapter';

const root = resolve(import.meta.dirname, '..', '..');
const gameText = readFileSync(resolve(root, 'out/game.json'), 'utf8');

function renderTitle(sceneId: string, qualities: Record<string, unknown>) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  Object.assign(adapter.qualities, qualities);
  const scene = adapter.engine.game.scenes[sceneId];
  return convertLine(adapter.engine._makeDisplayContent(scene.title, true));
}

describe('authored coalition tooltip metadata', () => {
  it('resolves Dendry arithmetic and party ids inside option-title attributes', () => {
    const html = renderTitle('congreso_coalition_right.pp_cs', {
      pp_congreso_s: 120,
      cs_congreso_s: 30,
      congreso_s_majority: 176,
    });
    expect(html).toContain('class="gobierno-coalition"');
    expect(html).toContain('data-parties="pp cs"');
    expect(html).toContain('data-summary="Minority government - 150/176"');
    expect(html).toMatch(/<\/span> \(PP - csspa\)$/);
  });

  it('keeps legacy colour spans nested inside semantic coalition triggers', () => {
    const directories = [
      resolve(root, 'source/scenes/events/elections'),
      resolve(root, 'source/scenes/events/congreso_elections'),
    ];
    const titleLines = directories.flatMap((directory) =>
      readdirSync(directory)
        .filter((name) => /^(parlament|congreso)_coalition.*\.scene\.dry$/.test(name))
        .flatMap((name) => readFileSync(resolve(directory, name), 'utf8').split(/\r?\n/))
        .filter((line) => line.startsWith('title:') && line.includes('<span style=')),
    );

    expect(titleLines.length).toBeGreaterThan(0);
    for (const line of titleLines) {
      const semantic = line.search(/class="(?:generalitat|gobierno)-coalition"/);
      expect(semantic, line).toBeGreaterThanOrEqual(0);
      expect(semantic, line).toBeLessThan(line.indexOf('<span style='));
    }
  });

  it('migrates election-simulation government labels too', () => {
    const source = readFileSync(resolve(root, 'source/scenes/election_simulation.scene.dry'), 'utf8');
    expect(source).toContain('data-parties="jxsi"');
    expect(source).toContain('data-parties="jxcat erc"');
    expect(source).toContain('data-parties="psoe up"');
  });

  it('authors concise summaries instead of cleaning them up in the renderer', () => {
    const sourceRoot = resolve(root, 'source/scenes');
    const collect = (directory: string): string[] => readdirSync(directory, { withFileTypes: true })
      .flatMap((entry) => entry.isDirectory()
        ? collect(resolve(directory, entry.name))
        : entry.name.endsWith('.dry') ? [resolve(directory, entry.name)] : []);
    const source = collect(sourceRoot).map((file) => readFileSync(file, 'utf8')).join('\n');

    expect(source).not.toMatch(/data-summary="[^"]*\bseats"/);
    expect(source).not.toContain(' + " seats"');
  });

  it.each([
    ['Catalonia 2015', 'election_simulation.post_election_sim', {
      ciu_parlament_showviz: 1, jxsi_parlament_showviz: 0, jxcat_parlament_showviz: 0,
      ciu_parlament_s: 50, parlament_s_majority: 68,
    }, 'ciu', 'Minority government - 50/68'],
    ['Catalonia 2017', 'election_simulation.post_election_sim', {
      ciu_parlament_showviz: 0, jxsi_parlament_showviz: 0, jxcat_parlament_showviz: 1,
      jxcat_parlament_s: 34, erc_parlament_s: 32, parlament_s_majority: 68,
    }, 'jxcat erc', 'Minority government - 66/68'],
    ['Congreso 2016', 'election_simulation.post_election_sim_congreso', {
      dl_congreso_showviz: 0, pdcat_congreso_showviz: 1, jxcat_congreso_showviz: 0,
      pp_congreso_s: 137, congreso_s_majority: 176,
    }, 'pp', 'Minority government - 137/176'],
    ['Congreso 2019', 'election_simulation.post_election_sim_congreso', {
      dl_congreso_showviz: 0, pdcat_congreso_showviz: 0, jxcat_congreso_showviz: 1,
      fr_congreso_showviz: 0, psoe_congreso_s: 120, up_congreso_s: 35,
      congreso_s_majority: 176,
    }, 'psoe up', 'Minority government - 155/176'],
  ])('renders the %s simulation tooltip as valid resolved markup', (_label, sceneId, q, parties, summary) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, q);
    const host = document.createElement('div');
    host.innerHTML = adapter.renderView(sceneId);
    const marker = host.querySelector('.generalitat-coalition, .gobierno-coalition');
    expect(marker?.getAttribute('data-parties')).toBe(parties);
    expect(marker?.getAttribute('data-summary')).toBe(summary);
    expect(marker?.outerHTML).not.toContain('[+');
  });

  it('uses the old shell\'s expanding solid underline, with no resting dotted line', () => {
    const css = readFileSync(resolve(root, 'out/html/game.css'), 'utf8');
    expect(css).not.toContain('text-decoration: underline dotted currentColor 1px');
    expect(css).toContain('.generalitat-coalition:hover::after');
    expect(css).toContain('width: calc(100% - 10px)');
    expect(css).toContain('background: var(--mytooltip-color, currentColor)');
    expect(css).toMatch(/\.coalition-tooltip-logos\s*\{[^}]*justify-content:\s*center/s);
    expect(css).toMatch(/\.coalition-tooltip-logos\s*\{[^}]*margin-bottom:\s*12px/s);
  });

  it('centres and separates coalition logos in the new UI too', () => {
    const css = readFileSync(resolve(root, 'ui/src/styles/tooltips.css'), 'utf8');
    expect(css).toMatch(/\.coalition-logos\s*\{[^}]*justify-content:\s*center/s);
    expect(css).toMatch(/\.coalition-logos\s*\{[^}]*margin-bottom:\s*12px/s);
  });
});
