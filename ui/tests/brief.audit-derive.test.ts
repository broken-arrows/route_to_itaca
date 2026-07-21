import { describe, it, expect } from 'vitest';
import { execFileSync } from 'node:child_process';
import { writeFileSync, readFileSync, copyFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = join(__dirname, '..', '..');
const GAME = join(ROOT, 'out', 'game.json');

// Falsify the guard before trusting it (the standing rule from phase 2).
describe('audit-globals deriveFrom check', () => {
  it('fails the build on an unknown deriveFrom name', () => {
    const backup = GAME + '.bak';
    copyFileSync(GAME, backup);
    try {
      const game = JSON.parse(readFileSync(GAME, 'utf8'));
      const victim = Object.keys(game.scenes)[0];
      game.scenes[victim].content = [
        '<div data-widget="poll-map" data-props=\'{"deriveFrom":"totally_bogus"}\'></div>',
      ];
      writeFileSync(GAME, JSON.stringify(game));
      expect(() =>
        execFileSync('node', ['tools/audit-globals.mjs'], { cwd: ROOT }),
      ).toThrow();
    } finally {
      copyFileSync(backup, GAME);
      rmSync(backup);
    }
  });
});
