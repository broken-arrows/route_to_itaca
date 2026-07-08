export interface ChoiceView {
  id: string;
  title: string;
  subtitle?: string;
  canChoose: boolean;
  tags: string[];
}

export interface DeckView extends ChoiceView {
  image?: string;
}

export interface CardView {
  id: string;
  title: string;
  image?: string;
  tags: string[];
}

export type DrawResult =
  | CardView
  | { id: null; title: 'no_space_in_hand' | 'no_card_in_deck' };

export interface Frame {
  sceneId: string;
  sceneTags: string[];
  html: string;
  choices: ChoiceView[];
  isHand: boolean;
  decks: DeckView[];
  hand: CardView[];
  maxCards: number;
  pinned: CardView[];
  gameOver: boolean;
  bg: string | null;
  signals: unknown[];
}

export interface SaveMeta {
  slot: string;
  savedAt: string; // ISO
  year: number | null;
  month: number | null;
  playerParty: string | null;
  sceneId: string;
}
