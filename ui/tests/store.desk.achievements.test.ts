import { describe, it, expect, beforeEach, vi } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import path from 'node:path';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import { DELAYS } from '../src/components/desk/motion';

// Achievement-toast WIRING (phase 2.5 Task 8): the pure diff predicate is
// covered by achievements.test.ts; this file exercises it through the real
// desk store, against a REAL compiled game (compileGame/convertGameToJSON,
// same tool the CLI uses) so `this.achieve(...)` runs through the real
// engine action code — not a hand-rolled stand-in for it — and the
// registry comes through the real source/data/*.json -> game.json.data
// route (compiler.data-registry.test.ts's own pattern; NB path.sep, not a
// literal '/', per that file's own comment on dry.js's parseFilename).
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
      { id: 'bar', name: 'Bar Achievement', description: 'Do the bar.', stars: 1, image: 'img/bar.png' },
    ],
  }),
};

const FILES = [
  { name: 'info.dry', contents: 'title: T\nauthor: A\nstorage-id: test-game\nlanguages: en ca\n' },
  { name: 'root.scene.dry', contents: 'title: Root\n\nIntro.\n\n- @hub\n' },
  {
    name: 'hub.scene.dry',
    contents: 'title: Hub\nrole: desk\nis-hand: true\nmax-cards: 3\n\nDesk.\n\n- @gov_deck\n',
  },
  { name: 'gov_deck.scene.dry', contents: 'title: Gov\nrole: deck\nis-deck: true\n\n- #gcard\n' },
  {
    name: 'c1.scene.dry',
    contents:
      'title: Card One\nrole: card-party\ntags: gcard\non-arrival: {! this.achieve("foo"); !}\n\nCard prose.\n\n- @hub: Back\n',
  },
  ACHIEVEMENTS_JSON,
];

// Variant: a single on-arrival awards TWO achievements at once (mirrors
// game_over.scene.dry's real on-arrival, which can achieve() both
// game_completed and barones in the same block) — the queue must show both,
// one after another, not just the first.
const FILES_DOUBLE = [
  FILES[0],
  FILES[1],
  FILES[2],
  FILES[3],
  {
    name: 'c1.scene.dry',
    contents:
      'title: Card One\nrole: card-party\ntags: gcard\non-arrival: {! this.achieve("foo"); this.achieve("bar"); !}\n\nCard prose.\n\n- @hub: Back\n',
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

describe('desk store — achievement unlock toast (phase 2.5 Task 8)', () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    setAnimationsForTest(false);
  });

  it('does not toast anything on boot (the pre-boot/first-real-frame seed)', async () => {
    const { desk } = await boot(FILES);
    expect(desk.achievementToast).toBeNull();
  });

  it('loads and writes the manifest-scoped achievement ledger without reading the old title key', async () => {
    localStorage.setItem('T_achievements', JSON.stringify({ old: 1 }));
    localStorage.setItem('test-game:achievements', JSON.stringify({ foo: 1 }));

    const { game } = await boot(FILES);
    expect(game.q.achievement_foo).toBe(1);
    expect(game.q.achievement_old).toBeUndefined();

    game.choose(0);
    game.draw('gov_deck');
    game.play(game.frame!.hand[0].id);
    expect(JSON.parse(localStorage.getItem('test-game:achievements')!)).toEqual({ foo: 1 });
    expect(JSON.parse(localStorage.getItem('T_achievements')!)).toEqual({ old: 1 });
  });

  it('retains the title-based achievement key for games without a storage id', async () => {
    localStorage.setItem('T_achievements', JSON.stringify({ foo: 1 }));
    const files = [
      { name: 'info.dry', contents: 'title: T\nauthor: A\nlanguages: en ca\n' },
      ...FILES.slice(1),
    ];

    const { game } = await boot(files);
    expect(game.q.achievement_foo).toBe(1);
  });

  it('toasts the registry name/image/stars when this.achieve() fires for the first time', async () => {
    const { game, desk } = await boot(FILES);
    game.choose(0); // -> hub, idle

    vi.useFakeTimers();
    try {
      desk.drawFrom('gov_deck');
      const card = game.frame!.hand[0];
      desk.playFromHand(card); // c1's on-arrival: this.achieve("foo")

      expect(desk.achievementToast).toEqual({
        name: 'Foo Achievement',
        image: 'img/foo.png',
        stars: 3,
      });

      // Auto-dismisses at the achievement-toast delay (distinct from the
      // ordinary nudge's DELAYS.toast), then clears — no second entry queued.
      vi.advanceTimersByTime(DELAYS.achievementToast - 1);
      expect(desk.achievementToast).not.toBeNull();
      vi.advanceTimersByTime(1);
      expect(desk.achievementToast).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not re-toast the same achievement on a later arrival (falsy->truthy already happened)', async () => {
    const { game, desk } = await boot(FILES);
    game.choose(0);

    vi.useFakeTimers();
    try {
      desk.drawFrom('gov_deck');
      desk.playFromHand(game.frame!.hand[0]);
      expect(desk.achievementToast).not.toBeNull();
      vi.advanceTimersByTime(DELAYS.achievementToast);
      expect(desk.achievementToast).toBeNull();

      // Back at hub, draw c1 again, play it again — this.achieve("foo") runs
      // again (content re-calls it unconditionally every time, same as the
      // real 12 call sites did), but Q.achievement_foo is already truthy, so
      // the falsy->truthy diff finds nothing new.
      desk.drawFrom('gov_deck');
      desk.playFromHand(game.frame!.hand[0]);
      expect(desk.achievementToast).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('queues multiple achievements unlocked in the SAME on-arrival and shows them one at a time', async () => {
    const { game, desk } = await boot(FILES_DOUBLE);
    game.choose(0);

    vi.useFakeTimers();
    try {
      desk.drawFrom('gov_deck');
      desk.playFromHand(game.frame!.hand[0]); // achieve("foo"); achieve("bar") in one block

      expect(desk.achievementToast?.name).toBe('Foo Achievement');
      vi.advanceTimersByTime(DELAYS.achievementToast);
      expect(desk.achievementToast?.name).toBe('Bar Achievement');
      vi.advanceTimersByTime(DELAYS.achievementToast);
      expect(desk.achievementToast).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('an unlock with no matching registry entry is silently skipped (not a crash)', async () => {
    // this.achieve() ids with no gallery entry exist for real (the never-
    // awarded trio); a registry gap here must not throw or wedge the queue.
    const files = [
      FILES[0],
      FILES[1],
      FILES[2],
      FILES[3],
      {
        name: 'c1.scene.dry',
        contents:
          'title: Card One\nrole: card-party\ntags: gcard\non-arrival: {! this.achieve("unregistered"); !}\n\nCard prose.\n\n- @hub: Back\n',
      },
      ACHIEVEMENTS_JSON,
    ];
    const { game, desk } = await boot(files);
    game.choose(0);
    expect(() => {
      desk.drawFrom('gov_deck');
      desk.playFromHand(game.frame!.hand[0]);
    }).not.toThrow();
    expect(desk.achievementToast).toBeNull();
  });
});
