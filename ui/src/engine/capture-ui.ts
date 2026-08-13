import type { RawCard, RawChoice } from 'dendrynexus-ten/lib/engine.js';
import { convertLine } from 'dendrynexus-ten/lib/ui/content/html.js';

/**
 * The engine hands us titles/subtitles as dendry "content" arrays
 * (e.g. ['The Desk']) even though RawChoice/RawCard declare `title: string`
 * — verified against vendor/dendrynexus-ten/lib/engine.js
 * (`_makeDisplayContent(title, false)` in `__getChoiceDisplayData`).
 * convertLine flattens content to plain text (and passes an already-plain
 * string through unchanged), so we normalize at this boundary — the one
 * place untyped engine output enters our type system — and everything
 * downstream (adapter, Frame, views) can trust the declared string types.
 */
function flatten(value: unknown): string {
  return convertLine(value);
}
function flattenOptional(value: unknown): string | undefined {
  return value === undefined ? undefined : convertLine(value);
}

function normalizeChoice(choice: RawChoice): RawChoice {
  return {
    ...choice,
    title: flatten(choice.title),
    subtitle: flattenOptional(choice.subtitle),
    unavailableSubtitle: flattenOptional(choice.unavailableSubtitle),
  };
}

/**
 * Exported so callers outside CaptureUI (e.g. DendryAdapter.drawCard, whose
 * result bypasses the UI's displayHand callback) can normalize a raw card's
 * title without duplicating the convertLine logic above.
 */
export function normalizeCard(card: RawCard): RawCard {
  return { ...card, title: flatten(card.title) };
}

/** Records everything the engine "displays". A plain object satisfies the
 *  engine's UI contract (duck typing) — do not subclass UserInterface. */
export class CaptureUI {
  paragraphs: unknown[] = [];
  choices: RawChoice[] = [];
  decks: RawChoice[] = [];
  hand: RawCard[] = [];
  maxCards = 0;
  pinned: RawChoice[] = [];
  bg: string | null = null;
  faceImage: string | null = null;
  signals: unknown[] = [];

  // -- content --
  beginGame(): void {}
  displayContent(paragraphs: unknown[], faceImage?: unknown): void {
    this.paragraphs.push(...paragraphs);
    // A face image belongs to the current page, not to one individual scene
    // transition. Dendry continuations commonly append prose without repeating
    // `face-image`; the old browser UI leaves the existing figure in place.
    // Only replace it when new media is explicitly supplied. `newPage()` is
    // the boundary that clears it.
    if (typeof faceImage === 'string') this.faceImage = faceImage;
  }
  newPage(): void {
    this.paragraphs = [];
    this.faceImage = null;
  }
  // -- choices & cards --
  displayChoices(choices: RawChoice[]): void {
    this.choices = choices.map(normalizeChoice);
  }
  displayDecks(decks: RawChoice[]): void {
    this.decks = decks.map(normalizeChoice);
  }
  displayHand(hand: RawCard[], maxCards?: number): void {
    this.hand = hand.map(normalizeCard);
    if (typeof maxCards === 'number') this.maxCards = maxCards;
  }
  displayPinnedCards(cards: RawChoice[]): void {
    this.pinned = cards.map(normalizeChoice);
  }
  removeChoices(): void {
    this.choices = [];
    this.decks = [];
    this.pinned = [];
  }
  displayGameOver(): void {}
  // -- lifecycle & misc (no-ops or simple records) --
  beginOutput(): void {}
  endOutput(): void {}
  setStyle(_style: unknown): void {}
  signal(data: unknown): void {
    this.signals.push(data);
  }
  setBg(img: string | null): void {
    this.bg = img;
  }
  setSprites(_data: unknown): void {}
  setSpriteStyle(_loc: unknown, _style: unknown): void {}
  audio(_audio: unknown): void {}

  /** Reset per-action volatile state (choices are re-displayed each scene). */
  resetTransient(): void {
    this.choices = [];
    this.decks = [];
    this.hand = [];
    this.pinned = [];
    this.signals = [];
  }
}
