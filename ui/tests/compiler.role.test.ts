import { describe, it, expect } from 'vitest';
import { compileGame } from 'dendrynexus-ten/lib/parsers/compiler.js';

function compile(files: { name: string; contents: string }[]): Promise<any> {
  return new Promise((resolve, reject) => {
    compileGame(files, (err, game) => (err ? reject(err) : resolve(game)));
  });
}

const scene = (body: string) => ({ name: 't.scene.dry', contents: body });

describe('role scene attribute', () => {
  it('accepts a valid role and emits it on the scene', async () => {
    const game = await compile([scene('title: T\nrole: card\n\nBody.\n')]);
    expect(game.scenes.t.role).toBe('card');
  });

  it('accepts pinned-action (hyphenated value)', async () => {
    const game = await compile([scene('title: T\nrole: pinned-action\n\nBody.\n')]);
    expect(game.scenes.t.role).toBe('pinned-action');
  });

  it('allows a scene with no role (undefined)', async () => {
    const game = await compile([scene('title: T\n\nBody.\n')]);
    expect(game.scenes.t.role).toBeUndefined();
  });

  it('rejects an unknown role with a helpful error', async () => {
    await expect(compile([scene('title: T\nrole: bogus\n\nBody.\n')])).rejects.toThrow(/role/i);
  });
});
