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

const EVENT_FILES = [
  { name: 'info.dry', contents: 'title: T\nauthor: A\nlanguages: en\n' },
  {
    name: 'root.scene.dry',
    contents: 'title: Root\n\nIntro.\n\n- @event\n- @ordinary\n- @paper\n',
  },
  {
    name: 'event.scene.dry',
    contents:
      'title: Event\nrole: event\n\nEvent.\n\n- @child\n\n@child\ntitle: Child\n\nChild.\n\n- @cross_file\n',
  },
  {
    name: 'cross_file.scene.dry',
    contents: 'title: Cross-file continuation\n\nCross-file result.\n\n- @done\n',
  },
  {
    name: 'ordinary.scene.dry',
    contents: 'title: Ordinary\nrole: card-party\n\nCard.\n\n- @shared\n',
  },
  {
    name: 'shared.scene.dry',
    contents: 'title: Shared\n\nShared helper.\n\n- @done\n',
  },
  {
    name: 'paper.scene.dry',
    contents: 'title: El Matí\nrole: newspaper\n\nEdition.\n\n- @done\n',
  },
  { name: 'done.scene.dry', contents: 'title: Done\n\nDone.\n' },
];

function stateAt(a: DendryAdapter, sceneId: string): string {
  a.goToScene(sceneId);
  return a.exportStateJSON();
}

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

  it('load uses explicit scene roles and keeps the page fallback for unrelated role-less scenes', async () => {
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

  it('restores event presentation for namespaced and cross-file continuations', async () => {
    const source = await adapterFor(EVENT_FILES);
    source.beginGame();
    source.choose(0); // event
    expect(source.choose(0).effectiveRole).toBe('event'); // namespaced child
    const child = source.exportStateJSON();
    expect(source.choose(0).effectiveRole).toBe('event'); // cross-file continuation
    const crossFile = source.exportStateJSON();

    const loaded = await adapterFor(EVENT_FILES);
    loaded.beginGame();
    expect(loaded.importStateJSON(child)).toMatchObject({
      sceneId: 'event.child',
      role: undefined,
      effectiveRole: 'event',
    });
    expect(loaded.importStateJSON(crossFile)).toMatchObject({
      sceneId: 'cross_file',
      role: undefined,
      effectiveRole: 'event',
    });
  });

  it('restores newspaper scenes directly but does not guess for shared continuations', async () => {
    const source = await adapterFor(EVENT_FILES);
    source.beginGame();
    const paper = stateAt(source, 'paper');

    // Make the otherwise cross-file event continuation shared with a non-event
    // role. Static provenance is then ambiguous and must keep the page fallback.
    const sharedSource = await adapterFor(
      EVENT_FILES.map((file) =>
        file.name === 'ordinary.scene.dry'
          ? { ...file, contents: file.contents.replace('@shared', '@cross_file') }
          : file,
      ),
    );
    sharedSource.beginGame();
    const shared = stateAt(sharedSource, 'cross_file');

    const loaded = await adapterFor(EVENT_FILES);
    loaded.beginGame();
    expect(loaded.importStateJSON(paper).effectiveRole).toBe('newspaper');

    const ambiguous = await adapterFor(
      EVENT_FILES.map((file) =>
        file.name === 'ordinary.scene.dry'
          ? { ...file, contents: file.contents.replace('@shared', '@cross_file') }
          : file,
      ),
    );
    ambiguous.beginGame();
    expect(ambiguous.importStateJSON(shared).effectiveRole).toBe('page');
  });

  it('restores an authored event face image even though engine setState does not replay it', async () => {
    const withImage = EVENT_FILES.map((file) =>
      file.name === 'event.scene.dry'
        ? { ...file, contents: file.contents.replace('role: event', 'role: event\nface-image: img/event.jpg') }
        : file,
    );
    const source = await adapterFor(withImage);
    source.beginGame();
    source.choose(0);
    const state = source.exportStateJSON();

    const loaded = await adapterFor(withImage);
    loaded.beginGame();
    expect(loaded.importStateJSON(state).faceImage).toBe('img/event.jpg');
  });

  it('exposes game info with languages', async () => {
    const a = await adapterFor(FILES);
    expect(a.info.languages).toEqual(['en', 'ca']);
    expect(a.beginGame().info.languages).toEqual(['en', 'ca']);
  });
});
