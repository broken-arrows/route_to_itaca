export type SceneRole =
  | 'desk' | 'deck' | 'deck-gov' | 'deck-party' | 'deck-parliament'
  | 'card' | 'card-gov' | 'card-party' | 'card-parliament'
  | 'pinned-action' | 'newspaper' | 'event'
  | 'status' | 'info-tab' | 'pause-item' | 'main-menu-item' | 'library-item' | 'ending';
export type EffectiveRole = SceneRole | 'page';
export interface GameInfo {
  title?: string;
  author?: string;
  ifid?: string;
  storageId?: string;
  version?: string;
  languages: string[];
}

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
  title: string;
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
  faceImage: string | null;
  signals: unknown[];
}

export interface SaveMeta {
  savedAt: string; // ISO
  year: number | null;
  month: number | null;
  playerParty: string | null;
  sceneId: string;
  resources: number | null;
}

export type SaveCompatibility = 'compatible' | 'incompatible' | 'unknown';
export type SaveEntryStatus = 'ready' | 'corrupt' | 'unreadable' | 'unsupported';

export interface SaveSlotEntry extends Partial<SaveMeta> {
  slot: string;
  status: SaveEntryStatus;
  compatibility?: SaveCompatibility;
  meta?: SaveMeta;
  error?: { code: string; message?: string };
}

// game.json.data.achievements = { achievements: AchievementEntry[] } —
// harvested from source/data/achievements.json (phase 2.5 Task 8). `id` is
// the bare name passed to `this.achieve(id)`, matching Q.achievement_<id> /
// Q.game_achievement_<id>.
export interface AchievementEntry {
  id: string;
  name: string;
  description: string;
  stars: number;
  image: string;
}
