import { describe, it, expect } from 'vitest';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { DendryAdapter } from '../src/engine/adapter';

function compile(files: { name: string; contents: string }[]): Promise<any> {
  return new Promise((res, rej) => compileGame(files, (e, g) => (e ? rej(e) : res(g))));
}
async function adapterFor(files: { name: string; contents: string }[]): Promise<DendryAdapter> {
  const game = await compile(files);
  const json = await new Promise<string>((res, rej) =>
    convertGameToJSON(game, 0, (e: Error | null, out?: string) => (e ? rej(e) : res(out!))),
  );
  return DendryAdapter.fromJSONText(json);
}

const FILES = [
  { name: 'info.dry', contents: 'title: T\nauthor: A\nlanguages: en ca\n' },
  {
    name: 'root.scene.dry',
    contents: 'title: Root\n\nIntro.\n\n- @hub\n',
  },
  {
    name: 'hub.scene.dry',
    contents:
      'title: Hub\nrole: desk\nis-hand: true\nmax-cards: 3\n\nDesk.\n\n- @gov_deck\n',
  },
  {
    name: 'gov_deck.scene.dry',
    contents: 'title: Gov\nrole: deck\nis-deck: true\n\n- #gcard\n',
  },
  {
    name: 'c1.scene.dry',
    contents:
      'title: Card One\nrole: card-party\ntags: gcard\n\nCard prose.\n\n- @c1_next\n',
  },
  {
    name: 'c1_next.scene.dry',
    contents: 'title: After\n\nOutcome.\n\n- @hub: Back\n',
  },
];

describe('effective role tracking', () => {
  it('boot base-case is page; explicit roles override; role-less inherits', async () => {
    const a = await adapterFor(FILES);
    let f = a.beginGame();
    expect(f.effectiveRole).toBe('page'); // root has no role
    f = a.choose(0); // -> hub
    expect(f.role).toBe('desk');
    expect(f.effectiveRole).toBe('desk');
    const draw = a.drawCard('gov_deck');
    expect(draw.result.id).toBe('c1');
    if (draw.result.id) expect(draw.result.role).toBe('card-party');
    f = a.playCard('c1'); // -> c1
    expect(f.effectiveRole).toBe('card-party');
    f = a.choose(0); // -> c1_next, role-less: inherits card-party
    expect(f.role).toBeUndefined();
    expect(f.effectiveRole).toBe('card-party');
    f = a.choose(0); // -> hub: desk resets
    expect(f.effectiveRole).toBe('desk');
  });

  it('load recomputes effective role from the loaded scene alone', async () => {
    const a = await adapterFor(FILES);
    a.beginGame();
    a.choose(0); // hub
    a.drawCard('gov_deck');
    a.playCard('c1');
    const midCard = a.exportStateJSON();
    a.choose(0);
    a.choose(0); // back at hub (desk)
    const atDesk = a.exportStateJSON();
    const b = await adapterFor(FILES);
    b.beginGame();
    expect(b.importStateJSON(atDesk).effectiveRole).toBe('desk'); // scene's own role
    expect(b.importStateJSON(midCard).effectiveRole).toBe('card-party'); // c1's own role
    // and a role-less scene loads as page:
    const c = await adapterFor(FILES);
    const rootState = c.beginGame() && c.exportStateJSON();
    expect(b.importStateJSON(rootState).effectiveRole).toBe('page');
  });

  it('exposes game info with languages', async () => {
    const a = await adapterFor(FILES);
    expect(a.info.languages).toEqual(['en', 'ca']);
    expect(a.beginGame().info.languages).toEqual(['en', 'ca']);
  });
});
