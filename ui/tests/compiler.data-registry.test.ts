import { describe, expect, it } from 'vitest';
import { compileGame } from 'dendrynexus-ten/lib/parsers/compiler.js';

// NB: no directory component in `name`. dry.js's parseFilename hardcodes the
// platform separator and mis-parses a forward-slash path on Windows — see
// LEARNINGS.md 2026-07-08. Registries are routed on the `data/` prefix, so we
// must use a real separator here; build it with path.sep.
import path from 'node:path';

const compile = (files: { name: string; contents: string }[]) =>
  new Promise<any>((resolve, reject) =>
    compileGame(files, (err: unknown, game: unknown) => (err ? reject(err) : resolve(game))),
  );

const INFO = { name: 'info.dry', contents: 'title: T\nauthor: A\n' };
const SCENE = { name: 'root.scene.dry', contents: 'title: Root\n\nHello.\n' };

describe('source/data/*.json → game.json.data', () => {
  it('attaches a registry under its basename', async () => {
    const game = await compile([
      INFO,
      SCENE,
      {
        name: ['source', 'data', 'glossary.json'].join(path.sep),
        contents: JSON.stringify({ terms: [{ id: 'ciu', match: ['CiU'] }] }),
      },
    ]);
    expect(game.data.glossary.terms[0].id).toBe('ciu');
  });

  it('leaves data undefined when there are no registries', async () => {
    const game = await compile([INFO, SCENE]);
    expect(game.data).toBeUndefined();
  });

  it('fails the compile on malformed registry JSON rather than silently skipping', async () => {
    await expect(
      compile([
        INFO,
        SCENE,
        { name: ['source', 'data', 'broken.json'].join(path.sep), contents: '{ not json' },
      ]),
    ).rejects.toThrow(/broken\.json/);
  });
});
