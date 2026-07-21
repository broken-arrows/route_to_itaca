import { describe, expect, it, vi } from 'vitest';
import { DendryEngine } from 'dendrynexus-ten/lib/engine.js';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { CaptureUI } from '../src/engine/capture-ui';

// Scene code that throws at RUNTIME is deliberately swallowed by the engine (a
// single broken block must not crash the game). These tests pin the diagnostic
// that makes it findable: the message must name the scene, the phase, and the
// exact failing STATEMENT — not the whole block, which is what it reported
// before (useless for e.g. congreso_coalition's 556-line on-arrival).
//
// The feature IS a console string, so asserting on the string is the only
// thing that can catch a regression here.

const compile = (files: { name: string; contents: string }[]) =>
  new Promise<any>((resolve, reject) =>
    compileGame(files, (err: unknown, game: unknown) => (err ? reject(err) : resolve(game))),
  );

// Must go through the compiler's own serializer: a plain JSON.stringify drops
// $code entirely. Same reasoning as engine.gamelib.test.ts.
const toJSON = (game: unknown) =>
  new Promise<string>((resolve, reject) =>
    convertGameToJSON(game, 0, (err: unknown, json: string) => (err ? reject(err) : resolve(json))),
  );

const boot = async (files: { name: string; contents: string }[], lib?: object) => {
  const json = await toJSON(await compile(files));
  const { convertJSONToGame } = await import('dendrynexus-ten/lib/engine.js');
  return new Promise<{ engine: any; ui: CaptureUI }>((resolve, reject) => {
    convertJSONToGame(json, (err: unknown, g: unknown) => {
      if (err) return reject(err);
      if (!g) return reject(new Error('convertJSONToGame produced no game'));
      const ui = new CaptureUI();
      const engine = new DendryEngine(ui, g);
      if (lib) engine.setGameLib(lib);
      resolve({ engine, ui });
    });
  });
};

const captureErrors = () => vi.spyOn(console, 'error').mockImplementation(() => {});
const joined = (spy: ReturnType<typeof captureErrors>) => spy.mock.calls.flat().join('\n');

const INFO = { name: 'info.dry', contents: 'title: T\nauthor: A\n' };

describe('scene-code error reporting', () => {
  it('names the failing statement, not the whole block', async () => {
    // The throw is on source line 4 of the block, with distinct decoy lines
    // above and below it so a wrong offset cannot pass by accident.
    const files = [
      INFO,
      {
        name: 'root.scene.dry',
        contents: [
          'title: Root',
          'new-page: true',
          'on-arrival: {!',
          '    Q.first = 1;',
          '    Q.second = 2;',
          '    Q.third = 3;',
          '    Q.fourth = notDeclaredAnywhere.value;',
          '    Q.fifth = 5;',
          '!}',
          '',
          'Root.',
          '',
        ].join('\n'),
      },
    ];
    const err = captureErrors();
    const { engine } = await boot(files);
    engine.beginGame();
    const out = joined(err);
    err.mockRestore();

    expect(out).toContain('Scene action error');
    expect(out).toContain('on-arrival');
    expect(out).toContain('"root"');
    expect(out).toContain('ReferenceError');
    // The statement itself — the whole point of this feature.
    expect(out).toContain('Q.fourth = notDeclaredAnywhere.value;');
    // Block source line 4: "Q.first" is line 1 after trimming.
    expect(out).toMatch(/block source line 4, col \d+/);
    expect(out).toContain('^');
    // Context lines, and NOT the whole block.
    expect(out).toContain('Q.third = 3;');
    expect(out).not.toContain('Q.first = 1;');
  });

  it('drops the engine frames below the block', async () => {
    const files = [
      INFO,
      {
        name: 'root.scene.dry',
        contents: ['title: Root', 'on-arrival: {! boomNotDefined(); !}', '', 'Root.', ''].join(
          '\n',
        ),
      },
    ];
    const err = captureErrors();
    const { engine } = await boot(files);
    engine.beginGame();
    const out = joined(err);
    err.mockRestore();

    expect(out).toContain('boomNotDefined');
    // These are the fixed, information-free frames the old message drowned in.
    expect(out).not.toContain('runActions');
    expect(out).not.toContain('__changeScene');
    expect(out).not.toContain('goToScene');
  });

  it('resolves predicate blocks too', async () => {
    // A view-if takes a different path into makeFunctionFromSource than an
    // on-arrival does (runPredicate, no phase), so it needs its own coverage.
    const files = [
      INFO,
      {
        name: 'root.scene.dry',
        contents: [
          'title: Root',
          'new-page: true',
          '',
          'Root.',
          '',
          '- @gated: Gated',
          '',
          '@gated',
          'view-if: {! return missingPredicateVar > 1; !}',
          '',
          'Gated.',
          '',
        ].join('\n'),
      },
    ];
    const err = captureErrors();
    const { engine } = await boot(files);
    engine.beginGame();
    const out = joined(err);
    err.mockRestore();

    expect(out).toContain('Scene predicate error');
    expect(out).toContain('missingPredicateVar');
    expect(out).toContain('return missingPredicateVar > 1;');
    expect(out).toMatch(/block source line \d+, col \d+/);
  });

  it('resolves expression blocks too', async () => {
    // NB `[+ foo +]` compiles to Q['foo'] (undefined, no throw) and never
    // reaches runExpression's catch — the insert has to be magic to throw.
    //
    // This case ALSO documents a pre-existing engine bug unrelated to this
    // diagnostic: runExpression swallows and returns `default_` (undefined),
    // then _evaluateStateDependencies calls value.toString() on it
    // (engine.js:919) and throws for real. So a throwing insert crashes the
    // game today. The log below fires first; the crash is downstream of it.
    const files = [
      INFO,
      {
        name: 'root.scene.dry',
        contents: [
          'title: Root',
          'new-page: true',
          '',
          'Root [+ {! return missingExpressionVar.value; !} +].',
          '',
        ].join('\n'),
      },
    ];
    const err = captureErrors();
    const { engine } = await boot(files);
    try {
      engine.beginGame();
    } catch (downstream) {
      expect(String(downstream)).toContain('toString');
    }
    const out = joined(err);
    err.mockRestore();

    expect(out).toContain('Scene expression error');
    expect(out).toContain('missingExpressionVar');
    expect(out).toContain('return missingExpressionVar.value;');
    expect(out).toMatch(/block source line \d+, col \d+/);
  });

  it('keeps the frames ABOVE the block when content throws through G.*', async () => {
    const files = [
      INFO,
      {
        name: 'root.scene.dry',
        contents: [
          'title: Root',
          'new-page: true',
          'on-arrival: {!',
          '    Q.before = 1;',
          '    G.explode(Q);',
          '!}',
          '',
          'Root.',
          '',
        ].join('\n'),
      },
    ];
    const err = captureErrors();
    const { engine } = await boot(files, {
      explode: function explodeInGameLib() {
        throw new TypeError('lib blew up');
      },
    });
    engine.beginGame();
    const out = joined(err);
    err.mockRestore();

    expect(out).toContain('TypeError: lib blew up');
    // Where it actually broke...
    expect(out).toContain('explodeInGameLib');
    // ...and which content line reached in.
    expect(out).toContain('reached from block source line 2');
    expect(out).toContain('G.explode(Q);');
  });

  it('falls back without throwing when the throw carries no stack', async () => {
    const files = [
      INFO,
      {
        name: 'root.scene.dry',
        contents: [
          'title: Root',
          "on-arrival: {! throw 'a bare string, no stack'; !}",
          '',
          'Root.',
          '',
        ].join('\n'),
      },
    ];
    const err = captureErrors();
    const { engine } = await boot(files);
    // The swallow must survive a throw the formatter cannot resolve.
    expect(() => engine.beginGame()).not.toThrow();
    const out = joined(err);
    err.mockRestore();

    expect(out).toContain('Scene action error');
    expect(out).toContain('"root"');
    expect(out).toContain('a bare string, no stack');
  });

  it('still swallows: the scene transitions and later chunks still run', async () => {
    const files = [
      INFO,
      {
        name: 'root.scene.dry',
        contents: [
          'title: Root',
          'new-page: true',
          'on-arrival: {! Q.before = 1; alsoNotDefined(); Q.never = 1; !} more = 7 {! Q.after = 1; !}',
          '',
          'Root.',
          '',
        ].join('\n'),
      },
    ];
    const err = captureErrors();
    const { engine } = await boot(files);
    engine.beginGame();
    err.mockRestore();

    const q = engine.state.qualities;
    expect(q.before).toBe(1);
    // The throw kills the REST OF ITS OWN CHUNK...
    expect(q.never).toBeUndefined();
    // ...but each() moves on to the next chunk, and the scene still arrives.
    expect(q.more).toBe(7);
    expect(q.after).toBe(1);
    expect(engine.state.sceneId).toBe('root');
  });
});
