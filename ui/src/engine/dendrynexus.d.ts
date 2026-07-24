declare module 'dendrynexus-ten/lib/engine.js' {
  export interface RawChoice {
    id: string;
    title: string;
    subtitle?: string;
    unavailableSubtitle?: string;
    canChoose: boolean;
    isDeck?: boolean;
    image?: string;
  }
  export interface RawCard {
    id: string;
    title: string;
    image?: string;
  }
  export class DendryEngine {
    constructor(ui: object, game: object);
    state: {
      sceneId: string;
      gameOver: boolean;
      qualities: Record<string, unknown>;
      currentHands: Record<string, RawCard[]>;
      bg: string | null;
      [key: string]: unknown;
    };
    game: {
      scenes: Record<string, Record<string, unknown>>;
      info?: { title?: string; author?: string; languages?: string[] };
      [key: string]: unknown;
    };
    beginGame(rndSeeds?: number[]): void;
    choose(choiceIndex: number): void;
    displayChoices(): void;
    drawCard(deckId: string): RawCard | { id: null; title: string };
    playCard(cardId: string): void;
    playPinnedCard(cardId: string): void;
    goToScene(id: string): void;
    getCurrentScene(): Record<string, unknown>;
    getCurrentChoices(): RawChoice[] | undefined;
    getExportableState(): object;
    setState(state: object): void;
    isGameOver(): boolean;
    setLocale(locale: string | null, catalog: Record<string, string> | null): this;
    // Public catalog lookup for non-content strings (source/lib row labels).
    translate(s: string): string;
    // Public wrapper for the engine's private _getQDisplay (engine.js:901).
    // Classifies a raw quality value through one of the built-in displays
    // ('cardinal'/'ordinal'/'number'/'fudge', each returning a word/ordinal
    // string) or a game-authored qdisplay (source/qdisplays/*.dry), whose
    // `content` ranges likewise resolve to plain-text `output` strings in
    // this codebase (no markup is used in any qdisplay's output lines) —
    // see getUserQDisplay/getCardinalNumber/getOrdinalNumber/getFudgeDisplay,
    // all of which return `string`. Row view-models (source/lib/brief.js)
    // carry {value, qdisplay} rather than a pre-computed band so the
    // qdisplay files stay the only place a threshold is written.
    qdisplay(value: unknown, qDisplayId: string): string;
    // Evaluates a content tree against current state and returns display
    // content. Private by name, but it is the engine's only out-of-band render
    // primitive and the old shell has called it in production since phase 2.5
    // (out/html/game.js:361, updateSidebar). Declared so renderView can use it.
    _makeDisplayContent(content: unknown, useParas: boolean): unknown[];
    // Evaluates a predicate function (a scene/option's compiled `view-if`)
    // against current state, returning `defaultValue` when the predicate is
    // undefined or throws. Private by name (DendryEngine.prototype
    // .`_runPredicate`, engine.js:1114), but — same as `_makeDisplayContent`
    // above — it is the engine's own evaluator and the only correct way to
    // read a `view-if`; do not reimplement it. NOTE: the module also exports
    // a bare `runPredicate` (engine.js:1765), but that is a MODULE-level
    // function taking `(predicate, default_, context, state)`, not a method
    // on `DendryEngine` instances — `engine.runPredicate` is `undefined` at
    // runtime (verified against the compiled engine). Only `_runPredicate`
    // exists as an instance method.
    _runPredicate(predicate: unknown, defaultValue: boolean): boolean;
    // Installs the game's own code namespace (source/lib/*), handed to
    // compiled content as a third `G` parameter alongside `state`/`Q` in
    // on-arrival/on-departure/on-display, predicates, and expression inserts.
    // See vendor/dendrynexus-ten/lib/engine.js DendryEngine.prototype.setGameLib.
    gameLib: Record<string, unknown>;
    setGameLib(lib: object): this;
  }
  export function convertJSONToGame(
    json: string,
    cb: (err: Error | null, game?: object) => void,
  ): void;
}

declare module 'dendrynexus-ten/lib/ui/content/html.js' {
  export function convert(paragraphs: unknown[]): string;
  // Verified against vendor/dendrynexus-ten/lib/ui/content/html.js
  // (module.exports.convertLine = _contentToHTML): the engine's compiled
  // choice/card titles and subtitles are dendry "content" arrays
  // (e.g. ['The Desk']), not plain strings, despite RawChoice/RawCard
  // declaring `title: string` above. convertLine flattens them to text;
  // CaptureUI calls it so everything downstream can trust the declared
  // string types. Also accepts an already-plain string unchanged.
  export function convertLine(content: unknown): string;
}

declare module 'dendrynexus-ten/lib/parsers/compiler.js' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function compileGame(
    files: { name: string; contents: string }[],
    callback: (err: Error | null, game: any) => void
  ): void;
  // Serializes a compiled Game to JSON text, replacing embedded compiled
  // functions (insert/predicate stateDependencies) with {$code: source}
  // markers that convertJSONToGame's reviver turns back into functions.
  // This is what the real CLI uses (lib/cli/cmd/compile.js) — plain
  // JSON.stringify silently drops those function-valued properties.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function convertGameToJSON(
    game: any,
    indent: number,
    callback: (err: Error | null, json: string) => void
  ): void;
}
