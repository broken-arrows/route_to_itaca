import { beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const ROOT = path.join(__dirname, '..', '..');
const WIDGETS_JS = readFileSync(path.join(ROOT, 'out', 'html', 'widgets.js'), 'utf8');

const results = [
  {
    id: 'cat-polls-widget-wide',
    name: 'parlament-results-map',
    renderer: 'initCataloniaPolls',
    scenes: ['election_simulation.scene.dry', 'events/elections/parlament_election.scene.dry'],
  },
  {
    id: 'congreso-map-widget',
    name: 'congreso-results-map',
    renderer: 'initCongresoMap',
    scenes: ['election_simulation.scene.dry', 'events/congreso_elections/congreso_election.scene.dry'],
  },
  {
    id: 'catalonia-local-map',
    name: 'local-results-map',
    renderer: 'initCatLocalMap',
    scenes: ['election_simulation.scene.dry', 'events/local_elections/local_elections.scene.dry'],
  },
  {
    id: 'congreso-party-tour-widget',
    name: 'congreso-party-tour',
    renderer: 'initCongresoPartyTour',
    scenes: ['events/congreso_elections/congreso_party_tour.scene.dry', 'erc_party_affairs/erc_reachout.scene.dry'],
  },
] as const;

describe('result widget protocol in the old shell', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    (window as any).initCataloniaPolls = vi.fn();
    (window as any).initCongresoMap = vi.fn();
    (window as any).initCatLocalMap = vi.fn();
    (window as any).initCongresoPartyTour = vi.fn();
    window.eval(WIDGETS_JS);
  });

  it.each(results)('renders marked $name through its old renderer exactly once', ({ id, name, renderer }) => {
    document.body.innerHTML = `<main><div id="${id}" data-widget="${name}"></div></main>`;
    const qualities = { marker: 'current state' };

    (window as any).mountWidgets(document.querySelector('main'), qualities);

    const call = (window as any)[renderer] as ReturnType<typeof vi.fn>;
    expect(call).toHaveBeenCalledTimes(1);
    expect(call).toHaveBeenCalledWith(id, qualities, ...(name === 'parlament-results-map' ? [true] : []));
  });

  it.each(results)('still renders an unmarked $id from older compiled content', ({ id, renderer }) => {
    document.body.innerHTML = `<main><div id="${id}"></div></main>`;
    const qualities = { marker: 'current state' };

    (window as any).mountWidgets(document.querySelector('main'), qualities);

    const call = (window as any)[renderer] as ReturnType<typeof vi.fn>;
    expect(call).toHaveBeenCalledTimes(1);
    expect(call).toHaveBeenCalledWith(id, qualities, ...(id === 'cat-polls-widget-wide' ? [true] : []));
  });

  it.each(results)('marks every authored $name placeholder without changing its ID', ({ id, name, scenes }) => {
    for (const scene of scenes) {
      const source = readFileSync(path.join(ROOT, 'source', 'scenes', scene), 'utf8');
      expect(source).toContain(`id="${id}" data-widget="${name}"`);
    }
  });
});
