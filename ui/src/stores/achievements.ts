/**
 * Achievement unlock DETECTION. There is no ledger here on purpose.
 *
 * The ENGINE already owns achievements: `achieve(x)` sets Q.achievement_x AND
 * Q.game_achievement_x and persists the cross-playthrough set to the manifest-
 * scoped `<storageId>:achievements` key (`rti:achievements` for this game).
 * Boot and save load restore Q.achievement_* from it. So:
 *
 *   Q.achievement_x       = ever unlocked, ACROSS SAVES (pre-seeded at boot)
 *   Q.game_achievement_x  = unlocked in THIS playthrough
 *
 * Phase 2 built a SECOND ledger (`rti:desk:achievements`) next to the engine's,
 * because dendrynexus_ten_plan.md §8 had the two qualities backwards. Deleted.
 *
 * A notification is therefore a frame-to-frame diff of `achievement_*`:
 * falsy -> truthy WITHIN a session means first time ever, which reproduces
 * content's existing `if (!Q.achievement_game_completed)` guards exactly.
 *
 * The `achievement_` PREFIX guard below already excludes `game_achievement_*`
 * on its own (that key starts with `game_`, not `achievement_`) — there is
 * no separate check needed to keep the per-playthrough ledger out of this
 * diff.
 */
const PREFIX = 'achievement_';

export function newlyUnlocked(
  prev: Record<string, unknown>,
  next: Record<string, unknown>,
): string[] {
  const out: string[] = [];
  for (const key of Object.keys(next)) {
    if (!key.startsWith(PREFIX)) continue;
    if (!next[key]) continue;
    if (prev[key]) continue;
    out.push(key.slice(PREFIX.length));
  }
  return out;
}
