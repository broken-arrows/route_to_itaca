import { readFileSync, readdirSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const repo = resolve(import.meta.dirname, '..', '..');
const scenesDir = resolve(repo, 'source', 'scenes');

function sceneFiles(dir = scenesDir): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sceneFiles(path);
    return entry.name.endsWith('.scene.dry') ? [path] : [];
  });
}

describe('countdown collection contract', () => {
  it('initializes as a JSON-safe array', () => {
    const root = readFileSync(join(scenesDir, 'root.scene.dry'), 'utf8');

    expect(root).toMatch(
      /Q\.countdowns = \[\s*"player_roadmap_change"\s*\];/,
    );
    expect(JSON.parse(JSON.stringify(['player_roadmap_change']))).toEqual([
      'player_roadmap_change',
    ]);
  });

  it('guards every countdown insertion against duplicates', () => {
    const unguarded: string[] = [];

    for (const file of sceneFiles()) {
      const lines = readFileSync(file, 'utf8').split(/\r?\n/);
      lines.forEach((line, index) => {
        const push = line.match(/Q\.countdowns\.push\((['"])([^'"]+)\1\)/);
        if (!push) return;

        const expectedGuard = `!Q.countdowns.includes(${push[1]}${push[2]}${push[1]})`;
        // A push can sit inside a guarded block as well as on the guard line.
        const nearby = lines.slice(Math.max(0, index - 3), index + 1).join('\n');
        if (!nearby.includes(expectedGuard)) {
          unguarded.push(`${relative(repo, file)}:${index + 1}`);
        }
      });
    }

    expect(unguarded).toEqual([]);
  });

  it('uses no Set-only APIs or temporary diagnostic insertion', () => {
    const sources = sceneFiles()
      .map((file) => readFileSync(file, 'utf8'))
      .join('\n');
    const postEvent = readFileSync(join(scenesDir, 'post_event.scene.dry'), 'utf8');

    expect(sources).not.toMatch(/Q\.countdowns\.(?:add|has|forEach)\b/);
    expect(sources).not.toMatch(/Q\.countdowns\s*=\s*new Set\b/);
    expect(postEvent).not.toContain('Q.countdowns.push("yes")');
    expect(postEvent).not.toContain('Countdowns type');
  });
});
