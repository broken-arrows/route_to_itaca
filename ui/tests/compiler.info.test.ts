import { describe, it, expect } from 'vitest';
import { join } from 'node:path';
import { compileGame } from 'dendrynexus-ten/lib/parsers/compiler.js';

function compile(files: { name: string; contents: string }[]): Promise<any> {
  return new Promise((resolve, reject) => {
    compileGame(files, (err, game) => (err ? reject(err) : resolve(game)));
  });
}

// path.join uses the native separator: '\' on Windows (exercises the routing-
// regex fix, the actual bug) and '/' on Linux/CI (regression guard).
const info = (contents: string) => ({ name: join('proj', 'source', 'info.dry'), contents });
const scene = { name: join('scenes', 't.scene.dry'), contents: 'title: T\n\nBody.\n' };

describe('info manifest emission', () => {
  it('picks up info.dry even with backslash paths (Windows) and emits a whitelisted info block', async () => {
    const game = await compile([
      info('title: My Game\nauthor: Me\nlanguages: en ca\n'),
      scene,
    ]);
    expect(game.info).toBeDefined();
    expect(game.info.title).toBe('My Game');
    expect(game.info.author).toBe('Me');
    expect(game.info.languages).toEqual(['en', 'ca']);
  });

  it('produces an empty info block when there is no info.dry', async () => {
    const game = await compile([scene]);
    expect(game.info).toEqual({});
  });
});
