export type SceneRole =
  | 'desk' | 'deck'
  | 'card' | 'card-gov' | 'card-party' | 'card-parlament'
  | 'pinned-action' | 'newspaper' | 'event'
  | 'info-tab' | 'pause-item' | 'main-menu-item' | 'library-item' | 'ending';
export type EffectiveRole = SceneRole | 'page';
export interface GameInfo { title?: string; author?: string; languages: string[] }

export interface ChoiceView {
  id: string;
  title: string;
  subtitle?: string;
  canChoose: boolean;
  tags: string[];
  role?: SceneRole;
}

export interface DeckView extends ChoiceView {
  image?: string;
}

export interface CardView {
  id: string;
  title: string;
  image?: string;
  tags: string[];
  role?: SceneRole;
}

export type DrawResult =
  | CardView
  | { id: null; title: 'no_space_in_hand' | 'no_card_in_deck' };

export interface Frame {
  sceneId: string;
  sceneTags: string[];
  role?: SceneRole;
  effectiveRole: EffectiveRole;
  info: GameInfo;
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
  resources: number | null;
}
