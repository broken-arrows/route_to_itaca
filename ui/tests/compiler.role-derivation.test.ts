import { describe, it, expect } from 'vitest';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { DendryAdapter } from '../src/engine/adapter';

function compile(files: { name: string; contents: string }[]): Promise<any> {
  return new Promise((resolve, reject) => {
    compileGame(files, (err: Error | null, game: any) => (err ? reject(err) : resolve(game)));
  });
}

async function adapterFor(files: { name: string; contents: string }[]): Promise<DendryAdapter> {
  const game = await compile(files);
  const json = await new Promise<string>((res, rej) =>
    convertGameToJSON(game, 0, (e: Error | null, out?: string) => (e ? rej(e) : res(out!))),
  );
  return DendryAdapter.fromJSONText(json);
}

const scene = (body: string) => ({ name: 't.scene.dry', contents: body });

describe('role -> mechanical boolean derivation', () => {
  it.each(['card', 'card-gov', 'card-party', 'card-parliament'])(
    'role: %s derives isCard: true when is-card is absent',
    async (role) => {
      const game = await compile([scene(`title: T\nrole: ${role}\n\nBody.\n`)]);
      expect(game.scenes.t.isCard).toBe(true);
    },
  );

  it('role: deck derives isDeck: true when is-deck is absent', async () => {
    const game = await compile([scene('title: T\nrole: deck\n\nBody.\n')]);
    expect(game.scenes.t.isDeck).toBe(true);
  });

  it('role: desk derives isHand: true when is-hand is absent', async () => {
    const game = await compile([scene('title: T\nrole: desk\n\nBody.\n')]);
    expect(game.scenes.t.isHand).toBe(true);
  });

  it('role: pinned-action derives isPinnedCard: true when is-pinned-card is absent', async () => {
    const game = await compile([scene('title: T\nrole: pinned-action\n\nBody.\n')]);
    expect(game.scenes.t.isPinnedCard).toBe(true);
  });

  it('non-mechanical role (event) leaves all mechanical booleans undefined', async () => {
    const game = await compile([scene('title: T\nrole: event\n\nBody.\n')]);
    expect(game.scenes.t.isCard).toBeUndefined();
    expect(game.scenes.t.isDeck).toBeUndefined();
    expect(game.scenes.t.isHand).toBeUndefined();
    expect(game.scenes.t.isPinnedCard).toBeUndefined();
  });

  it('roleless scene leaves all mechanical booleans undefined', async () => {
    const game = await compile([scene('title: T\n\nBody.\n')]);
    expect(game.scenes.t.isCard).toBeUndefined();
    expect(game.scenes.t.isDeck).toBeUndefined();
    expect(game.scenes.t.isHand).toBeUndefined();
    expect(game.scenes.t.isPinnedCard).toBeUndefined();
  });

  it('explicit is-card wins over the role-derived value, even when false', async () => {
    const game = await compile([scene('title: T\nrole: card\nis-card: false\n\nBody.\n')]);
    expect(game.scenes.t.isCard).toBe(false);
  });

  it('derives isDeck on a section (@sub) the same way as a root scene', async () => {
    const game = await compile([
      scene('title: T\nrole: event\n\nRoot body.\n\n@sub\ntitle: Sub\nrole: deck\n\nSub body.\n'),
    ]);
    expect(game.scenes['t.sub'].isDeck).toBe(true);
    // The root scene's own (non-mechanical) role is untouched by this.
    expect(game.scenes.t.isCard).toBeUndefined();
  });

  it('behavioral: _drawFromDeck sees a role-only card (no is-card) as drawable', async () => {
    const files = [
      { name: 'info.dry', contents: 'title: T\nauthor: A\n' },
      { name: 'root.scene.dry', contents: 'title: Root\n\nIntro.\n\n- @hub\n' },
      {
        name: 'hub.scene.dry',
        contents: 'title: Hub\nis-hand: true\nmax-cards: 3\n\nDesk.\n\n- @pool_deck\n',
      },
      {
        name: 'pool_deck.scene.dry',
        contents: 'title: Pool\nis-deck: true\n\n- #pool\n',
      },
      {
        name: 'c1.scene.dry',
        // role-only: no is-card attribute at all.
        contents: 'title: Card One\nrole: card\ntags: pool\n\nCard prose.\n',
      },
    ];
    const a = await adapterFor(files);
    a.beginGame();
    a.choose(0); // -> hub
    const draw = a.drawCard('pool_deck');
    expect(draw.result.id).toBe('c1');
  });
});
