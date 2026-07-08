declare module 'dendrynexus/lib/engine.js' {
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
    game: { scenes: Record<string, Record<string, unknown>>; [key: string]: unknown };
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
  }
  export function convertJSONToGame(
    json: string,
    cb: (err: Error | null, game?: object) => void,
  ): void;
}

declare module 'dendrynexus/lib/ui/content/html.js' {
  export function convert(paragraphs: unknown[]): string;
  // Verified against vendor/dendrynexus/lib/ui/content/html.js
  // (module.exports.convertLine = _contentToHTML): the engine's compiled
  // choice/card titles and subtitles are dendry "content" arrays
  // (e.g. ['The Desk']), not plain strings, despite RawChoice/RawCard
  // declaring `title: string` above. convertLine flattens them to text;
  // CaptureUI calls it so everything downstream can trust the declared
  // string types. Also accepts an already-plain string unchanged.
  export function convertLine(content: unknown): string;
}
