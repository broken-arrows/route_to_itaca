// Cross-save achievement ledger. Unlike settings/game saves this is NOT a
// pinia store: it's a plain read-modify-write over one localStorage blob,
// called from the desk store's autosave hook (and, later, wherever
// achievements are displayed). Achievement qualities are `Q.achievement_*`,
// truthy when unlocked, and reset per playthrough — this module is what
// makes an unlock permanent across saves/newGame.

const STORAGE_KEY = 'rti:desk:achievements';
const PREFIX = 'achievement_';

export interface AchievementRecord {
  unlockedAt: string; // ISO
  inGame: { year: number | null; month: number | null };
}

export type AchievementsBlob = Record<string, AchievementRecord>;

function persist(blob: AchievementsBlob): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(blob));
}

export function listUnlocked(): AchievementsBlob {
  if (typeof localStorage === 'undefined') return {};
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return {};
  try {
    return JSON.parse(raw) as AchievementsBlob;
  } catch {
    return {};
  }
}

// Diffs truthy achievement_* keys in `q` against the stored set, records the
// new ones with the current wall-clock + in-game time, and returns the ids
// that were newly unlocked by THIS call (empty when nothing changed).
export function recordUnlocks(
  q: Record<string, unknown>,
  inGame: { year: number | null; month: number | null },
): string[] {
  const stored = listUnlocked();
  const newly: string[] = [];
  for (const key of Object.keys(q)) {
    if (!key.startsWith(PREFIX)) continue;
    if (!q[key]) continue;
    if (stored[key]) continue;
    stored[key] = { unlockedAt: new Date().toISOString(), inGame };
    newly.push(key);
  }
  if (newly.length > 0) persist(stored);
  return newly;
}
