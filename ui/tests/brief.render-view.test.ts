import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

// Real game.json, not a fixture: the phase-2 rule — a fixture written from the
// same mental model as the code cannot falsify that model.
const GAME = join(__dirname, '..', '..', 'out', 'game.json');

describe('adapter.renderView', () => {
  let a: DendryAdapter;
  beforeAll(() => {
    a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
  });

  it('renders a scene that is not the current scene', () => {
    const html = a.renderView('status');
    expect(typeof html).toBe('string');
    expect(html.length).toBeGreaterThan(0);
  });

  it('does not touch engine state', () => {
    const sceneBefore = a.engine.state.sceneId;
    const qBefore = JSON.stringify(a.qualities);
    a.renderView('status');
    expect(a.engine.state.sceneId).toBe(sceneBefore);
    expect(JSON.stringify(a.qualities)).toBe(qBefore);
  });

  it('returns empty string for a missing scene rather than throwing', () => {
    expect(a.renderView('no_such_scene_at_all')).toBe('');
  });

  it('translate is identity with no catalog installed', () => {
    expect(a.translate('Airports')).toBe('Airports');
  });

  it('translate uses the catalog when one is installed', () => {
    a.setLocale('ca', { Airports: 'Aeroports' });
    expect(a.translate('Airports')).toBe('Aeroports');
    expect(a.translate('Not In Catalog')).toBe('Not In Catalog');
    a.setLocale(null, null);
  });
});
