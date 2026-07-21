import { DendryEngine, convertJSONToGame } from 'dendrynexus-ten/lib/engine.js';
import { convert as paragraphsToHTML } from 'dendrynexus-ten/lib/ui/content/html.js';
import { CaptureUI, normalizeCard } from './capture-ui';
import { installGameLib } from '../game-bindings';
import type { Frame, DrawResult, SceneRole, EffectiveRole, GameInfo, AchievementEntry } from './types';
import type { GlossaryTerm } from '../glossary/mark';

export class DendryAdapter {
  readonly engine: DendryEngine;
  private ui: CaptureUI;
  // Runtime "effective role" tracker: a scene's own non-`default` role becomes
  // the new effective role; role-less (or `default`) scenes inherit the current
  // one. The `desk` reset is just this same assignment — no special case.
  private effective: EffectiveRole = 'page';

  private constructor(game: object) {
    this.ui = new CaptureUI();
    this.engine = new DendryEngine(this.ui, game);
    // The game's own code, handed to the engine — see game-bindings.ts. Must
    // happen BEFORE beginGame: content's on-arrival calls G.* on the very first
    // scene.
    installGameLib(this.engine);
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

  get info(): GameInfo {
    const raw = this.engine.game.info ?? {};
    return { languages: ['en'], ...raw } as GameInfo;
  }

  /** Game registries compiled from `source/data/*.json` (see the §2.5 spec
   *  §3.2 and compiler.data-registry.test.ts). `game.data` is undefined
   *  when source/data/ carries no registries — never throw on that. */
  get data(): Record<string, unknown> {
    return (this.engine.game as { data?: Record<string, unknown> }).data ?? {};
  }

  get glossary(): GlossaryTerm[] {
    const g = this.data.glossary as { terms?: GlossaryTerm[] } | undefined;
    return g?.terms ?? [];
  }

  /** game.json.data.achievements.achievements — see AchievementEntry. Never
   *  throws when source/data/ carries no achievements registry. */
  get achievements(): AchievementEntry[] {
    const a = this.data.achievements as { achievements?: AchievementEntry[] } | undefined;
    return a?.achievements ?? [];
  }

  setLocale(locale: string | null, catalog: Record<string, string> | null): void {
    this.engine.setLocale(locale, catalog);
  }

  setGameLib(lib: object): void {
    this.engine.setGameLib(lib);
  }

  /**
   * Renders a scene's content tree against CURRENT Q, out of band — no scene
   * transition, no choice compilation, no autosave. This is how the Brief's
   * tab sheets render (spec §2): the same three calls the old shell's
   * updateSidebar has made in production since phase 2.5, minus its
   * `_runActions(scene.onArrival)` — that side-effecting re-run is exactly
   * what phase 3b deletes.
   */
  renderView(sceneId: string): string {
    const scene = this.engine.game.scenes[sceneId] as { content?: unknown } | undefined;
    if (!scene || scene.content === undefined) {
      if (!scene) console.warn(`renderView: no scene "${sceneId}"`);
      return '';
    }
    return paragraphsToHTML(this.engine._makeDisplayContent(scene.content, true));
  }

  /** Catalog lookup for strings built outside the content tree — see
   *  source/lib/brief.js's LABELS tables. */
  translate(s: string): string {
    return this.engine.translate(s);
  }

  /** Classify a raw quality value through one of the game's qdisplays. Row
   *  view-models carry {value, qdisplay} rather than a pre-computed band, so
   *  source/qdisplays/*.dry stays the ONLY place a threshold is written. */
  qdisplay(value: unknown, qdisplayId: string): string {
    return this.engine.qdisplay(value, qdisplayId);
  }

  beginGame(seeds?: number[]): Frame {
    this.ui.resetTransient();
    this.effective = 'page';
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
        : { ...normalizeCard(raw), tags: this.tagsFor(raw.id), role: this.roleFor(raw.id) };
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
    // Recompute from the loaded scene alone; dossier context (an inherited
    // effective role from a role-less scene's ancestor) is intentionally NOT
    // restored across load.
    this.effective = this.roleFor(this.engine.state.sceneId) ?? 'page';
    return this.buildFrame();
  }

  private tagsFor(id: string): string[] {
    const s = this.engine.game.scenes[id] as { tags?: string[] } | undefined;
    return s?.tags ?? [];
  }

  private roleFor(id: string): SceneRole | undefined {
    const r = (this.engine.game.scenes[id] as { role?: string } | undefined)?.role;
    return r && r !== 'default' ? (r as SceneRole) : undefined;
  }

  protected buildFrame(): Frame {
    const sceneId = this.engine.state.sceneId;
    const scene = (this.engine.game.scenes[sceneId] ?? {}) as Record<string, unknown>;
    const ownRole = this.roleFor(sceneId);
    if (ownRole) this.effective = ownRole; // 'desk' reset is just this assignment
    return {
      sceneId,
      sceneTags: (scene.tags as string[] | undefined) ?? [],
      role: ownRole,
      effectiveRole: this.effective,
      info: this.info,
      html: paragraphsToHTML(this.ui.paragraphs),
      choices: this.ui.choices.map((c) => ({
        id: c.id,
        title: c.title,
        subtitle: c.subtitle ?? c.unavailableSubtitle,
        canChoose: !!c.canChoose,
        tags: this.tagsFor(c.id),
        role: this.roleFor(c.id),
      })),
      isHand: !!scene.isHand,
      decks: this.ui.decks.map((d) => ({
        id: d.id,
        title: d.title,
        subtitle: d.subtitle,
        canChoose: !!d.canChoose,
        image: d.image,
        tags: this.tagsFor(d.id),
        role: this.roleFor(d.id),
      })),
      hand: this.ui.hand.map((c) => ({
        id: c.id,
        title: c.title,
        image: c.image,
        tags: this.tagsFor(c.id),
        role: this.roleFor(c.id),
      })),
      maxCards: this.ui.maxCards,
      pinned: this.ui.pinned.map((c) => ({
        id: c.id,
        title: c.title,
        image: c.image,
        tags: this.tagsFor(c.id),
        role: this.roleFor(c.id),
      })),
      gameOver: this.engine.isGameOver(),
      bg: this.ui.bg,
      signals: [...this.ui.signals],
    };
  }
}
