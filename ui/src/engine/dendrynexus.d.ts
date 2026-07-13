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
