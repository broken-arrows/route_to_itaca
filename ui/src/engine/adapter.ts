import { DendryEngine, convertJSONToGame } from 'dendrynexus-ten/lib/engine.js';
import { convert as paragraphsToHTML } from 'dendrynexus-ten/lib/ui/content/html.js';
import { CaptureUI, normalizeCard } from './capture-ui';
import type { Frame, DrawResult } from './types';

export class DendryAdapter {
  readonly engine: DendryEngine;
  private ui: CaptureUI;

  private constructor(game: object) {
    this.ui = new CaptureUI();
    this.engine = new DendryEngine(this.ui, game);
  }

  static fromJSONText(text: string): DendryAdapter {
    let out: object | undefined;
    convertJSONToGame(text, (err, game) => {
      if (err) throw err;
      out = game;
    });
    if (!out) throw new Error('convertJSONToGame produced no game');
    return new DendryAdapter(out);
  }

  get qualities(): Record<string, unknown> {
    return this.engine.state.qualities;
  }

  beginGame(seeds?: number[]): Frame {
    this.ui.resetTransient();
    this.engine.beginGame(seeds);
    return this.buildFrame();
  }

  currentFrame(): Frame {
    return this.buildFrame();
  }

  choose(choiceIndex: number): Frame {
    this.ui.resetTransient();
    this.engine.choose(choiceIndex);
    return this.buildFrame();
  }

  goToScene(id: string): Frame {
    this.ui.resetTransient();
    this.engine.goToScene(id);
    return this.buildFrame();
  }

  drawCard(deckId: string): { result: DrawResult; frame: Frame } {
    this.ui.resetTransient();
    const raw = this.engine.drawCard(deckId);
    // For a real card, raw.title arrives as a dendry content-array (e.g.
    // ['Card A']), not a string — same engine behaviour CaptureUI already
    // normalizes for the display path (see capture-ui.ts). The sentinel
    // returns (`no_space_in_hand` / `no_card_in_deck`) are plain strings
    // constructed literally in engine.js, so only the real-card branch
    // needs normalizing. Reuse CaptureUI's exported normalizeCard rather
    // than duplicating convertLine logic here.
    const result: DrawResult =
      raw.id === null
        ? { id: null, title: raw.title as 'no_space_in_hand' | 'no_card_in_deck' }
        : { ...normalizeCard(raw), tags: this.tagsFor(raw.id) };
    // engine.drawCard() (engine.js:474) only re-displays the hand; rebuild
    // the rest of the frame by asking the engine to re-fire its own
    // deck/hand/pinned display for the current scene. displayChoices() is a
    // verified public method (DendryEngine.prototype.displayChoices,
    // engine.js:322) that only rewrites choice/deck/pinned buffers — it does
    // not touch prose. Declared on DendryEngine in dendrynexus.d.ts.
    this.engine.displayChoices();
    return { result, frame: this.buildFrame() };
  }

  playCard(cardId: string): Frame {
    this.ui.resetTransient();
    this.engine.playCard(cardId);
    return this.buildFrame();
  }

  playPinnedCard(cardId: string): Frame {
    this.ui.resetTransient();
    this.engine.playPinnedCard(cardId);
    return this.buildFrame();
  }

  exportStateJSON(): string {
    return JSON.stringify(this.engine.getExportableState());
  }

  importStateJSON(json: string): Frame {
    this.ui.resetTransient();
    this.engine.setState(JSON.parse(json));
    return this.buildFrame();
  }

  private tagsFor(id: string): string[] {
    const s = this.engine.game.scenes[id] as { tags?: string[] } | undefined;
    return s?.tags ?? [];
  }

  protected buildFrame(): Frame {
    const sceneId = this.engine.state.sceneId;
    const scene = (this.engine.game.scenes[sceneId] ?? {}) as Record<string, unknown>;
    return {
      sceneId,
      sceneTags: (scene.tags as string[] | undefined) ?? [],
      html: paragraphsToHTML(this.ui.paragraphs),
      choices: this.ui.choices.map((c) => ({
        id: c.id,
        title: c.title,
        subtitle: c.subtitle ?? c.unavailableSubtitle,
        canChoose: !!c.canChoose,
        tags: this.tagsFor(c.id),
      })),
      isHand: !!scene.isHand,
      decks: this.ui.decks.map((d) => ({
        id: d.id,
        title: d.title,
        subtitle: d.subtitle,
        canChoose: !!d.canChoose,
        image: d.image,
        tags: this.tagsFor(d.id),
      })),
      hand: this.ui.hand.map((c) => ({ id: c.id, title: c.title, image: c.image, tags: this.tagsFor(c.id) })),
      maxCards: this.ui.maxCards,
      pinned: this.ui.pinned.map((c) => ({ id: c.id, title: c.title, image: c.image, tags: this.tagsFor(c.id) })),
      gameOver: this.engine.isGameOver(),
      bg: this.ui.bg,
      signals: [...this.ui.signals],
    };
  }
}
