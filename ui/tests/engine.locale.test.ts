import { describe, it, expect } from 'vitest';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';

// compileGame() returns a Game object with live compiled functions embedded
// in stateDependencies (insert/predicate fn). Plain JSON.stringify silently
// drops function-valued properties, so any scene with an insert would crash
// (fn undefined) once round-tripped. convertGameToJSON is the same
// serializer the real CLI uses (lib/cli/cmd/compile.js): its custom replacer
// turns each fn into {$code: source}, revived by convertJSONToGame's reviver.
function compileText(files: { name: string; contents: string }[]): Promise<string> {
  return new Promise((resolve, reject) => {
    compileGame(files, (err, game) => {
      if (err) return reject(err);
      convertGameToJSON(game, 0, (err2, json) => (err2 ? reject(err2) : resolve(json)));
    });
  });
}

// A hub scene whose option carries a plain-string title we can translate,
// plus a body sentence with an insert (must be passed through untranslated).
// path.join uses the native separator, matching parseFilename's platform-
// dependent basename split (dry.js:83) on both Windows and Linux/CI.
const hub = {
  name: join('scenes', 'hub.scene.dry'),
  contents:
    'title: Hub\n\nYou have [+ gold +] left.\n\n- @leaf: Renew the Front Bench\n',
};
const leaf = { name: join('scenes', 'leaf.scene.dry'), contents: 'title: Leaf\n\nDone.\n' };
const root = {
  name: join('scenes', 'root.scene.dry'),
  contents: 'title: Root\n\nStart.\n\n- @hub: Go\n',
};

describe('setLocale substitution seam', () => {
  it('localizes a plain-string choice title when a catalog is active', async () => {
    const text = await compileText([root, hub, leaf]);
    const a = DendryAdapter.fromJSONText(text);
    a.engine.setLocale('ca', { 'Renew the Front Bench': 'Renovar la primera línia' });
    a.beginGame();
    const frame = a.goToScene('hub');
    const choice = frame.choices.find((c) => c.id === 'leaf');
    expect(choice?.title).toBe('Renovar la primera línia');
  });

  it('is identity when no locale is set', async () => {
    const text = await compileText([root, hub, leaf]);
    const a = DendryAdapter.fromJSONText(text);
    a.beginGame();
    const frame = a.goToScene('hub');
    expect(frame.choices.find((c) => c.id === 'leaf')?.title).toBe('Renew the Front Bench');
  });

  it('passes a key through unchanged when the catalog lacks it', async () => {
    const text = await compileText([root, hub, leaf]);
    const a = DendryAdapter.fromJSONText(text);
    a.engine.setLocale('ca', { 'Something else': 'X' });
    a.beginGame();
    const frame = a.goToScene('hub');
    expect(frame.choices.find((c) => c.id === 'leaf')?.title).toBe('Renew the Front Bench');
  });

  it('does NOT translate fragments of an insert sentence (deferred transform)', async () => {
    const text = await compileText([root, hub, leaf]);
    const a = DendryAdapter.fromJSONText(text);
    a.engine.setLocale('ca', { ' left.': 'FRAGMENT-SHOULD-NOT-APPLY' });
    a.beginGame();
    const frame = a.goToScene('hub');
    expect(frame.html).toContain('left.');
    expect(frame.html).not.toContain('FRAGMENT-SHOULD-NOT-APPLY');
  });
});
