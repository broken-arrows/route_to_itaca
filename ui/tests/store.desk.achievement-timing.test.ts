import { describe, it, expect, beforeEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import path from 'node:path';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';

// Task 8 fix wave 1, Finding 1 (IMPORTANT, behavioral): reproduces the
// stale-`Q` bug in game.ts's `apply()`.
//
// `game.q` is a Vue `computed` memoized on a `version` ref (`void
// version.value` inside the getter). The desk's `flush:'sync'` frame
// watcher (desk.ts:278, `syncFromFrame` -> `checkAchievements`) fires
// SYNCHRONOUSLY inside `frame.value = f`. Before the fix, `apply()` ran
// `frame.value = f; version.value++;` in that order — so if some real
// consumer (DeskView.vue:28-29 reads `gameStore.q.month`/`.year` on every
// render) already read-and-cached `q` since the LAST version bump, the
// computed's cache is still clean when the ACHIEVING transition's frame
// assignment fires the sync watcher. `checkAchievements` then reads the
// PREVIOUS Q snapshot — missing the achievement the CURRENT transition
// just set — and with no further transition to self-correct on (the real
// case: `game_completed`/`barones`, unlocked on `game_over`'s terminal
// frame), the toast never fires at all.
//
// This test drives that exact sequence with the real compiler + real
// game/desk stores (same harness as store.desk.achievements.test.ts, not a
// mock — the bug is about Vue computed memoization ordering, which a mock
// can't exercise honestly): one no-op transition, a `game.q` read standing
// in for DeskView's per-render read, then a SECOND transition whose
// on-arrival calls `this.achieve(...)` with no subsequent action. Only a
// direct `game.choose()` is used deliberately — the desk store's own
// `playFromHand`/`pickPaper` each make an explicit "belt-and-suspenders"
// second `syncFromFrame()` call right after their engine action, which
// would re-read `game.q` post-bump and mask this exact ordering bug.
function compile(files: { name: string; contents: string }[]): Promise<any> {
  return new Promise((res, rej) => compileGame(files, (e, g) => (e ? rej(e) : res(g))));
}
async function jsonFor(files: { name: string; contents: string }[]): Promise<string> {
  const game = await compile(files);
  return new Promise<string>((res, rej) =>
    convertGameToJSON(game, 0, (e: Error | null, out?: string) => (e ? rej(e) : res(out!))),
  );
}

const ACHIEVEMENTS_JSON = {
  name: ['source', 'data', 'achievements.json'].join(path.sep),
  contents: JSON.stringify({
    achievements: [
      { id: 'foo', name: 'Foo Achievement', description: 'Do the foo.', stars: 3, image: 'img/foo.png' },
    ],
  }),
};

const FILES = [
  { name: 'info.dry', contents: 'title: T\nauthor: A\nlanguages: en ca\n' },
  { name: 'root.scene.dry', contents: 'title: Root\n\nIntro.\n\n- @stepA\n' },
  { name: 'stepA.scene.dry', contents: 'title: Step A\n\nWaiting.\n\n- @achieving\n' },
  {
    name: 'achieving.scene.dry',
    contents: 'title: Achieving\non-arrival: {! this.achieve("foo"); !}\n\nDone.\n',
  },
  ACHIEVEMENTS_JSON,
];

async function boot(files: { name: string; contents: string }[]) {
  const game = useGameStore();
  const desk = useDeskStore();
  game.initFromText(await jsonFor(files));
  game.newGame();
  return { game, desk };
}

describe('desk store — achievement toast timing vs game.q memoization (Task 8 fix wave 1, Finding 1)', () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    setAnimationsForTest(false);
  });

  it('toasts an achievement unlocked on a transition even when a consumer cached game.q right before it', async () => {
    const { game, desk } = await boot(FILES);

    game.choose(0); // root -> stepA (no achievement)

    // Simulate a real mounted consumer (DeskView.vue's deskMonth/deskYear
    // computeds) reading game.q AFTER this transition settled — this is
    // exactly what primes/caches the computed and, pre-fix, leaves it
    // clean for the NEXT transition's synchronous watcher read.
    void game.q.month;

    game.choose(0); // stepA -> achieving: this.achieve("foo") fires here, no further action

    expect(desk.achievementToast).toEqual({
      name: 'Foo Achievement',
      image: 'img/foo.png',
      stars: 3,
    });
  });
});
