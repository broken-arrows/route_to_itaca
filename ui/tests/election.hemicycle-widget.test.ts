import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const repo = resolve(import.meta.dirname, '..', '..');
const scene = (relative: string) =>
  readFileSync(resolve(repo, 'source', 'scenes', relative), 'utf8');
const gameText = readFileSync(resolve(repo, 'out', 'game.json'), 'utf8');

const expectRebuiltOnDisplay = (source: string, config: string) => {
  const assignment = source.indexOf(`Q.${config} = {`);
  expect(assignment).toBeGreaterThan(-1);
  expect(source.lastIndexOf('on-display:', assignment)).toBeGreaterThan(
    source.lastIndexOf('on-arrival:', assignment),
  );
};

describe.each([
  {
    chamber: 'Parlament',
    file: 'events/elections/parlament_election.scene.dry',
    elementId: 'parlament',
    config: 'parlament_election_hemicycle',
  },
  {
    chamber: 'Congreso',
    file: 'events/congreso_elections/congreso_election.scene.dry',
    elementId: 'congreso',
    config: 'congreso_election_hemicycle',
  },
])('$chamber election results', ({ file, elementId, config }) => {
  it('publishes a hemicycle view-model and opts the result surface into animation', () => {
    const source = scene(file);
    expect(source).toContain(`Q.${config} = {`);
    expectRebuiltOnDisplay(source, config);
    expect(source).toContain('seats: data.map(function (p) {');
    expect(source).toContain(`data-widget="hemicycle"`);
    expect(source).toContain(`"configFrom":"${config}"`);
    expect(source).toContain('"animate":true');
  });

  it('does not leave an empty legacy SVG or a direct D3 hemicycle mount behind', () => {
    const source = scene(file);
    expect(source).not.toContain(`<svg id="${elementId}"`);
    expect(source).not.toContain(`d3.select("#${elementId}").datum(data).call(parliament)`);
  });
});

describe.each([
  ['parlament_election.post_election', 'parlament_election_hemicycle'],
  ['congreso_election.post_election', 'congreso_election_hemicycle'],
  ['election_simulation.post_election_sim', 'election_simulation_parlament_hemicycle'],
  ['election_simulation.post_election_sim_congreso', 'election_simulation_congreso_hemicycle'],
  ['election_simulation.post_election_sim_local', 'election_simulation_local_hemicycle'],
])('saved election result %s', (sceneId, config) => {
  it('preserves its JSON-safe display model when restored', () => {
    const source = DendryAdapter.fromJSONText(gameText);
    source.beginGame([1, 2, 3, 4]);
    source.goToScene(sceneId);
    const expected = JSON.parse(JSON.stringify(source.qualities[config]));
    expect((expected as { seats: unknown[] }).seats.length).toBeGreaterThan(0);

    const restored = DendryAdapter.fromJSONText(gameText);
    restored.beginGame([1, 2, 3, 4]);
    restored.importStateJSON(source.exportStateJSON());
    expect(restored.qualities[config]).toEqual(expected);
  });
});

describe.each([
  ['congreso_election.post_election', 'congreso_election_hemicycle'],
  ['election_simulation.post_election_sim', 'election_simulation_parlament_hemicycle'],
  ['election_simulation.post_election_sim_congreso', 'election_simulation_congreso_hemicycle'],
  ['election_simulation.post_election_sim_local', 'election_simulation_local_hemicycle'],
])('restored current result scene %s', (sceneId, config) => {
  it('reconstructs a missing display model through on-display', () => {
    const source = DendryAdapter.fromJSONText(gameText);
    source.beginGame([1, 2, 3, 4]);
    source.goToScene(sceneId);
    const saved = JSON.parse(source.exportStateJSON()) as {
      sceneId: string;
      qualities: Record<string, unknown>;
    };
    expect(saved.sceneId).toBe(sceneId);
    delete saved.qualities[config];

    const restored = DendryAdapter.fromJSONText(gameText);
    restored.beginGame([1, 2, 3, 4]);
    restored.importStateJSON(JSON.stringify(saved));
    expect((restored.qualities[config] as { seats: unknown[] }).seats.length).toBeGreaterThan(0);
  });
});

describe.each([
  {
    chamber: 'simulated Parlament',
    elementId: 'parlament',
    config: 'election_simulation_parlament_hemicycle',
  },
  {
    chamber: 'simulated Congreso',
    elementId: 'congreso',
    config: 'election_simulation_congreso_hemicycle',
  },
  {
    chamber: 'simulated Barcelona council',
    elementId: 'barcelonalocal',
    config: 'election_simulation_local_hemicycle',
  },
])('$chamber results', ({ elementId, config }) => {
  const source = scene('election_simulation.scene.dry');

  it('publishes an animated hemicycle through the dual-UI widget protocol', () => {
    expect(source).toContain(`Q.${config} = {`);
    expectRebuiltOnDisplay(source, config);
    expect(source).toContain(`"configFrom":"${config}"`);
    expect(source).toContain('"animate":true');
  });

  it('does not retain its legacy direct-D3 mount', () => {
    expect(source).not.toContain(`<svg id="${elementId}"`);
    expect(source).not.toContain(`d3.select("#${elementId}").datum(data).call(parliament)`);
  });
});
