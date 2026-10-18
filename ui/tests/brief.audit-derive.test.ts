import { describe, it, expect } from 'vitest';
import { execFileSync } from 'node:child_process';
import { writeFileSync, readFileSync, mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const ROOT = join(__dirname, '..', '..');
const GAME = join(ROOT, 'out', 'game.json');

// Falsify the guard before trusting it (the standing rule from phase 2).
describe('audit-globals deriveFrom check', () => {
  it('fails the build on an unknown deriveFrom name', () => {
    const project = mkdtempSync(join(tmpdir(), 'rti-derive-audit-'));
    try {
      const game = JSON.parse(readFileSync(GAME, 'utf8'));
      const victim = Object.keys(game.scenes)[0];
      game.scenes[victim].content = [
        '<div data-widget="poll-map" data-props=\'{"deriveFrom":"totally_bogus"}\'></div>',
      ];
      mkdirSync(join(project, 'out'));
      writeFileSync(join(project, 'out/game.json'), JSON.stringify(game));
      expect(() =>
        execFileSync(process.execPath, ['vendor/dendrynexus-ten/lib/cli/main.js', 'audit', project,
          '--config', join(ROOT, 'dendrynexus.audit.mjs')], { cwd: ROOT, stdio: 'pipe' }),
      ).toThrow(/Unknown derivation: totally_bogus/);
    } finally {
      rmSync(project, { recursive: true, force: true });
    }
  });
});
