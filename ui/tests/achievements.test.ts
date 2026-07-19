import { describe, expect, it } from 'vitest';
import { newlyUnlocked } from '../src/stores/achievements';

describe('newlyUnlocked', () => {
  it('reports an achievement that went falsy -> truthy', () => {
    expect(newlyUnlocked({}, { achievement_calcotada: 1 })).toEqual(['calcotada']);
  });

  it('reports nothing when it was already set at boot (unlocked in a previous run)', () => {
    // The engine's _loadAchievements() pre-seeds Q.achievement_* from localStorage,
    // so "already truthy in prevQ" IS "unlocked before". No toast.
    expect(newlyUnlocked({ achievement_calcotada: 1 }, { achievement_calcotada: 1 })).toEqual([]);
  });

  it('ignores game_achievement_* — that is per-playthrough, not first-ever', () => {
    expect(newlyUnlocked({}, { game_achievement_calcotada: 1 })).toEqual([]);
  });

  it('reports several at once', () => {
    expect(
      newlyUnlocked({ achievement_a: 1 }, { achievement_a: 1, achievement_b: 1, achievement_c: 1 }),
    ).toEqual(['b', 'c']);
  });
});
