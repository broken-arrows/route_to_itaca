import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');

function configAt(sceneId: string, key: string) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene(sceneId);
  const config = adapter.qualities[key];
  return { adapter, config };
}

function containsFunction(value: unknown, seen = new Set<unknown>()): boolean {
  if (typeof value === 'function') return true;
  if (!value || typeof value !== 'object' || seen.has(value)) return false;
  seen.add(value);
  return Object.values(value).some((child) => containsFunction(child, seen));
}

describe('real coalition config persistence contract', () => {
  it.each([
    ['parlament_coalition', 'parlament_coalition_config'],
    ['congreso_coalition', 'congreso_coalition_config'],
  ])('%s emits a populated, function-free config that survives save/load', (sceneId, key) => {
    const { adapter, config } = configAt(sceneId, key);
    expect(config).toBeTruthy();
    expect((config as { coalitions: unknown[] }).coalitions.length).toBeGreaterThan(0);
    expect(containsFunction(config)).toBe(false);

    const serializedConfig = JSON.parse(JSON.stringify(config));
    expect(serializedConfig).toEqual(config);

    const restored = DendryAdapter.fromJSONText(gameText);
    restored.beginGame([1, 2, 3, 4]);
    restored.importStateJSON(adapter.exportStateJSON());
    expect(restored.qualities[key]).toEqual(serializedConfig);
  });
});
