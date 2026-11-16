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

  it.each([
    ['DeSk', 'desk'],
    ['DECK', 'deck'],
    ['Deck-GOV', 'deck-gov'],
    ['Deck-PARTY', 'deck-party'],
    ['Deck-PARLIAMENT', 'deck-parliament'],
    ['CARD', 'card'],
    ['Card-GOV', 'card-gov'],
    ['Card-PARTY', 'card-party'],
    ['Card-PARLIAMENT', 'card-parliament'],
    ['Pinned-ACTION', 'pinned-action'],
    ['Pinned-ADVISOR', 'pinned-advisor'],
    ['Pinned-PARLIAMENT', 'pinned-parliament'],
    ['NEWSPAPER', 'newspaper'],
    ['Event', 'event'],
    ['Status', 'status'],
    ['Info-TAB', 'info-tab'],
    ['Pause-ITEM', 'pause-item'],
    ['Title-HUB', 'title-hub'],
    ['Main-Menu-ITEM', 'main-menu-item'],
    ['Library-ITEM', 'library-item'],
    ['SETUP', 'setup'],
    ['INTRO', 'intro'],
    ['ENDING', 'ending'],
    ['Default', 'default'],
  ])('compiles role %s to canonical %s', async (authored, expected) => {
    const game = await compile([scene(`title: T\nrole: ${authored}\n\nBody.\n`)]);
    expect(game.scenes.t.role).toBe(expected);
  });

  it.each(['pinned-action', 'pinned-advisor', 'pinned-parliament'])(
    'accepts pinned presentation role %s and preserves it',
    async (role) => {
      const game = await compile([scene(`title: T\nrole: ${role}\n\nBody.\n`)]);
      expect(game.scenes.t.role).toBe(role);
    },
  );

  it.each(['title-hub', 'main-menu-item', 'setup', 'intro', 'ending'])(
    'accepts shell presentation role %s and preserves it',
    async (role) => {
      const game = await compile([scene(`title: T\nrole: ${role}\n\nBody.\n`)]);
      expect(game.scenes.t.role).toBe(role);
    },
  );

  it('allows a scene with no role (undefined)', async () => {
    const game = await compile([scene('title: T\n\nBody.\n')]);
    expect(game.scenes.t.role).toBeUndefined();
  });

  it('rejects an unknown role with a helpful error', async () => {
    await expect(compile([scene('title: T\nrole: bogus\n\nBody.\n')])).rejects.toThrow(/role/i);
  });

  it.each(['card-gov', 'card-party', 'card-parliament'])(
    'accepts skin value %s and emits it',
    async (r) => {
      const game = await compile([scene(`title: T\nrole: ${r}\n\nBody.\n`)]);
      expect(game.scenes.t.role).toBe(r);
    },
  );

  it('still rejects unknown skin-like values', async () => {
    await expect(compile([scene('title: T\nrole: card-ministry\n\nBody.\n')])).rejects.toThrow(/role/i);
  });

  it('rejects unknown roles regardless of case', async () => {
    await expect(compile([scene('title: T\nrole: Card-MINISTRY\n\nBody.\n')])).rejects.toThrow(/role/i);
  });

  it('still rejects unknown pinned presentation variants', async () => {
    await expect(compile([scene('title: T\nrole: pinned-ministry\n\nBody.\n')])).rejects.toThrow(
      /role/i,
    );
  });
});
