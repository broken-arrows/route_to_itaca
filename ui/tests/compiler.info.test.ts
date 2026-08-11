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
      info(
        'title: My Game\nauthor: Me\n' +
        'ifid: 12345678-1234-1234-1234-123456789abc\n' +
        'storage-id: my-game\nversion: 0.2.1-beta\nlanguages: en ca\n',
      ),
      scene,
    ]);
    expect(game.info).toBeDefined();
    expect(game.info.title).toBe('My Game');
    expect(game.info.author).toBe('Me');
    expect(game.info.ifid).toBe('12345678-1234-1234-1234-123456789abc');
    expect(game.info.storageId).toBe('my-game');
    expect(game.info.version).toBe('0.2.1-beta');
    expect(game.info.languages).toEqual(['en', 'ca']);
  });

  it.each(['0.2', '0.2-beta', '0.2.1', '0.2.1-beta.2'])('accepts game version %s', async (version) => {
    const game = await compile([info(`title: T\nauthor: A\nversion: ${version}\n`), scene]);
    expect(game.info.version).toBe(version);
  });

  it.each(['RTI', 'route_to_itaca', '9rti', 'rti:game'])('rejects storage id %s', async (storageId) => {
    await expect(compile([info(`title: T\nauthor: A\nstorage-id: ${storageId}\n`), scene])).rejects.toThrow(
      /not a valid storage id/,
    );
  });

  it.each(['0', '0.1.2.3', 'v0.1', '0.1-'])('rejects game version %s', async (version) => {
    await expect(compile([info(`title: T\nauthor: A\nversion: ${version}\n`), scene])).rejects.toThrow(
      /not a valid game version/,
    );
  });

  it('produces an empty info block when there is no info.dry', async () => {
    const game = await compile([scene]);
    expect(game.info).toEqual({});
  });
});
