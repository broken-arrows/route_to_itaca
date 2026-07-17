import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DendryEngine } from 'dendrynexus-ten/lib/engine.js';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { CaptureUI } from '../src/engine/capture-ui';

const compile = (files: { name: string; contents: string }[]) =>
  new Promise<any>((resolve, reject) =>
    compileGame(files, (err: unknown, game: unknown) => (err ? reject(err) : resolve(game))),
  );

// Compiled `$code` only survives a JSON round-trip through the compiler's own
// replacer (it stashes {$code: fn.source} for function values — see
// compiler.js's convertGameToJSON). A plain JSON.stringify(compiledGame) is
// NOT equivalent: JSON.stringify silently turns a function held in an ARRAY
// into `null` and DROPS a function held as a plain object property entirely,
// so on-arrival actions would decompile to `[null]` (a live TypeError at
// runActions) and every predicate would vanish (silently reverting to
// runPredicate's default). Use the real serializer so this fixture hydrates
// $code exactly as the compiled CLI output does.
const toJSON = (game: unknown) =>
  new Promise<string>((resolve, reject) =>
    convertGameToJSON(game, 0, (err: unknown, json: string) => (err ? reject(err) : resolve(json))),
  );

// A game whose root calls into the game lib from on-arrival, and gates a choice
// on a predicate that also uses it — the two invocation paths content actually
// takes.
const SOURCE = [
  { name: 'info.dry', contents: 'title: T\nauthor: A\n' },
  {
    name: 'root.scene.dry',
    contents: [
      'title: Root',
      'on-arrival: {! Q.ticked = G.tick(Q.ticked || 0); !}',
      '',
      'Root.',
      '',
      '- @gated: Gated',
      '',
      '@gated',
      'view-if: {! return G.allowed(); !}',
      '',
      'Gated.',
      '',
    ].join('\n'),
  },
];

// The engine hydrates $code via JSON round-trip, exactly as the real app does.
const boot = async (lib?: object) => {
  const json = await toJSON(await compile(SOURCE));
  const { convertJSONToGame } = await import('dendrynexus-ten/lib/engine.js');
  return new Promise<{ engine: any; ui: CaptureUI }>((resolve, reject) => {
    convertJSONToGame(json, (err, g) => {
      if (err) return reject(err);
      if (!g) return reject(new Error('convertJSONToGame produced no game'));
      const ui = new CaptureUI();
      const engine = new DendryEngine(ui, g);
      if (lib) engine.setGameLib(lib);
      resolve({ engine, ui });
    });
  });
};

describe('engine.setGameLib', () => {
  it('passes G into on-arrival action code', async () => {
    const { engine } = await boot({ tick: (n: number) => n + 41, allowed: () => true });
    engine.beginGame();
    expect(engine.state.qualities.ticked).toBe(41);
  });

  it('passes G into predicates', async () => {
    const { engine, ui } = await boot({ tick: (n: number) => n, allowed: () => false });
    engine.beginGame();
    expect(ui.choices.find((c) => c.id === 'root.gated')).toBeUndefined();

    const { engine: e2, ui: ui2 } = await boot({ tick: (n: number) => n, allowed: () => true });
    e2.beginGame();
    expect(ui2.choices.find((c) => c.id === 'root.gated')).toBeDefined();
  });

  it('warns once when the game lib was never installed', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { engine } = await boot();
    engine.beginGame();
    expect(warn.mock.calls.flat().join(' ')).toMatch(/setGameLib/);
    warn.mockRestore();
  });

  it('leaves G as an empty object, never undefined', async () => {
    const { engine } = await boot();
    engine.beginGame();
    // G.tick is not a function -> swallowed by runActions; the point is that the
    // failure is a MISSING METHOD, not a TypeError on undefined.
    expect(engine.gameLib).toEqual({});
  });
});
