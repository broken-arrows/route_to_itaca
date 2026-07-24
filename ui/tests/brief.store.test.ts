import { describe, it, expect, beforeEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { useGameStore } from '../src/stores/game';
import { useBriefStore } from '../src/stores/brief';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');

describe('brief store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const g = useGameStore();
    g.initFromText(readFileSync(GAME, 'utf8'));
    g.newGame();
  });

  it('finds the hub by ROLE, not by a hardcoded scene id', () => {
    const b = useBriefStore();
    expect(b.tabs.length).toBe(6);
    expect(b.tabs[0].id).toBe('status_new.overview');
  });

  it('tab labels come from each sheet scene title', () => {
    const b = useBriefStore();
    expect(b.tabs.map((t) => t.title)).toEqual([
      'Overview', 'Party', 'Chamber', 'Economy', 'World', 'Polls',
    ]);
  });

  it('defaults to the first tab and renders its HTML', () => {
    const b = useBriefStore();
    expect(b.activeTab).toBe('status_new.overview');
    expect(b.activeHtml.length).toBeGreaterThan(0);
  });

  it('select switches the rendered sheet', () => {
    const b = useBriefStore();
    const first = b.activeHtml;
    b.select('status_new.chamber');
    expect(b.activeTab).toBe('status_new.chamber');
    expect(b.activeHtml).not.toBe(first);
  });

  // STRENGTHENED (see task report): the brief's own version of this test only
  // asserted `all >= 5`, which passes whether view-if filtering works, is
  // broken, or was never implemented at all -- it never puts a tab in a state
  // where it SHOULD disappear. POLLS is the one sheet in the game whose
  // target scene carries a `view-if` (gated on `historical_mode`), so this
  // test puts that predicate through its paces both ways.
  //
  // Two layers, deliberately:
  //  1. `adapter.tabScenes()` directly -- the stateless primitive Step 3 asks
  //     for. Mutating `Q.historical_mode` on the live qualities object and
  //     re-calling it proves the view-if filter itself works in BOTH
  //     directions (hide, then un-hide).
  //  2. `useBriefStore().tabs` through a REAL content-driven Q change (an
  //     actual player choice, not a direct property poke) -- this is the
  //     regression guard for the store's own reactivity. `tabs` needed a
  //     `void game.q` tick dependency added (see task report): as written in
  //     the brief it only tracked `game.adapter` (the ref itself, which never
  //     changes across a session), so it silently never re-derived after the
  //     FIRST access -- not just in this test, in real play too. A direct
  //     `qualities.historical_mode = 1` poke (bypassing the engine's own
  //     `apply()`/version tick) does not exercise that fix, because nothing
  //     in this codebase makes `Q` itself a reactive object; only an actual
  //     engine action (`choose`, here) bumps the version the store computed
  //     depends on.
  it("adapter.tabScenes() hides POLLS when historical_mode is truthy, and restores it", () => {
    const g = useGameStore();
    const scenes = () => g.adapter!.tabScenes();

    expect(scenes().length).toBe(6);
    expect(scenes().some((t) => t.id === 'status_new.polls')).toBe(true);

    (g.adapter!.qualities as Record<string, unknown>).historical_mode = 1;
    expect(scenes().length).toBe(5);
    expect(scenes().some((t) => t.id === 'status_new.polls')).toBe(false);

    (g.adapter!.qualities as Record<string, unknown>).historical_mode = 0;
    expect(scenes().length).toBe(6);
    expect(scenes().some((t) => t.id === 'status_new.polls')).toBe(true);
  });

  it('the store tabs list reactively drops POLLS when a real choice sets historical_mode', () => {
    const g = useGameStore();
    const b = useBriefStore();

    expect(b.tabs.length).toBe(6);
    expect(b.tabs.some((t) => t.id === 'status_new.polls')).toBe(true);

    // root.start_menu_2 -> "Start game" -> root.start's difficulty choice,
    // whose "Historical mode" option sets Q.historical_mode = 1 via a real
    // on-arrival action (source/scenes/root.scene.dry). This is the same
    // `apply()`/version-tick path every ordinary player action takes.
    g.choose(0);
    const historicalIdx = g.frame!.choices.findIndex((c) => /historical/i.test(c.title));
    expect(historicalIdx).toBeGreaterThanOrEqual(0);
    g.choose(historicalIdx);

    expect(b.tabs.length).toBe(5);
    expect(b.tabs.some((t) => t.id === 'status_new.polls')).toBe(false);
  });

  it('selecting a tab does not change the engine scene', () => {
    const g = useGameStore();
    const b = useBriefStore();
    const before = g.frame?.sceneId;
    b.select('status_new.world');
    expect(g.frame?.sceneId).toBe(before);
  });
});
