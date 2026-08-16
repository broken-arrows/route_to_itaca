import { beforeAll, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');
let gameText = '';

beforeAll(() => {
  gameText = readFileSync(GAME, 'utf8');
});

function boot(): DendryAdapter {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  return adapter;
}

describe('Phase 5C legacy-shell authored navigation', () => {
  it('boots through the real authored title hub role', () => {
    const adapter = boot();
    expect(adapter.engine.game.scenes['root.start_menu'].role).toBe('title-hub');
    expect(adapter.goToScene('root.start_menu').effectiveRole).toBe('title-hub');
  });

  it('keeps Achievements textual Back wired to the engine prevScene destination', () => {
    const adapter = boot();
    adapter.goToScene('root.start');
    const frame = adapter.goToScene('game_over.achievements');
    const back = frame.choices.findIndex((choice) => choice.id === 'prevScene');

    expect(back).toBeGreaterThanOrEqual(0);
    expect(frame.choices[back].title).toContain('Back');
    expect(adapter.choose(back).sceneId).toBe('root.start');
  });

  it('keeps Library special-scene Exit wired to the exact engine origin', () => {
    const adapter = boot();
    adapter.goToScene('root.start');
    const frame = adapter.goToScene('library');
    const exit = frame.choices.findIndex((choice) => choice.id === 'backSpecialScene');

    expect(exit).toBeGreaterThanOrEqual(0);
    expect(frame.choices[exit].title).toContain('Exit library');
    expect(adapter.choose(exit).sceneId).toBe('root.start');
  });

  it('keeps Credits visibly headed and its textual Back inside About', () => {
    const adapter = boot();
    const frame = adapter.goToScene('credits');
    const back = frame.choices.findIndex((choice) => choice.id === 'about');

    expect(frame.html).toContain('<h1>Credits</h1>');
    expect(back).toBeGreaterThanOrEqual(0);
    expect(frame.choices[back].title).toContain('Back');
    expect(adapter.choose(back).sceneId).toBe('about.force_new_page_about');
  });
});
