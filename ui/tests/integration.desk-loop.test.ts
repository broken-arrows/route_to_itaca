import { describe, it, expect, beforeEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
// The game's own code (source/lib/), reached by compiled scene code as
// `G.engineTick` / `G.spaSupportInject`. `DendryAdapter`'s constructor installs
// it on every engine via `installGameLib` (game-bindings.ts), before
// `beginGame` runs — so constructing the adapter (which `game.initFromText`
// below does) wires it; nothing here touches `window` any more.
//
// HISTORY (phase-2.5 Task 3 changed this): this used to be
// `window.engineTick`/`window.spaSupportInject`, reachable only because
// `game-bindings.ts` side-effect-imported `out/html/cat_engine.js` onto
// `window`. The Vue app has no script tags, so without that import
// `window.engineTick` was `undefined` — and dendry's `runActions` SWALLOWS the
// resulting TypeError, so the loop kept working and the calendar kept moving
// while nothing simulated. That is precisely the bug this file guards against;
// only the MECHANISM changed (`window.*` -> `G.*` via `engine.setGameLib`), not
// the property being asserted. See `docs/design/LEARNINGS.md`, 2026-07-13.
import { gameLib } from '../src/game-bindings';
import { mount } from '@vue/test-utils';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import { i18n } from '../src/i18n';
import DeskView from '../src/views/DeskView.vue';
import InTray from '../src/components/desk/InTray.vue';
import type { CardView } from '../src/engine/types';
import uiEn from '../../source/locales/en/ui.json';

// Every DeskView mount below now also mounts Clipboard (phase 3b Task 9;
// formerly the inert ClipboardFrame), which reads `brief.context.*` — GAME
// chrome sourced from source/locales/<loc>/ui.json (see i18n.ts's
// initGameLocale, and the same fix in desk.components.test.ts).
i18n.global.mergeLocaleMessage('en', uiEn as never);

// The REAL compiled game, not a fixture. It is gitignored build output, so the
// test skips when it is absent (fresh clone, no compile yet) rather than
// failing. Regenerate from the repo root with `npm run dendrynexus-ten compile`.
const GAME_JSON = path.join(__dirname, '..', '..', 'out', 'game.json');
const HAVE_GAME = existsSync(GAME_JSON);

// Caps, per the phase-2 acceptance brief. They are deliberately generous: the
// observed real flow needs 8 pre-desk steps and 2 papers, and the first card
// already crosses a month boundary. A cap being HIT is a genuine content
// regression (the desk became unreachable / a card no longer resolves), not a
// number to raise.
const MAX_PRE_DESK_STEPS = 30;
const MAX_PAPERS = 10;
const MAX_TURNS = 20;

const AUTO_SAVE_RE = /^rti:save:auto-/;

function firstChoosable(choices: { canChoose: boolean }[]): number {
  return choices.findIndex((c) => c.canChoose);
}

// Macro-simulation outputs that ONLY `monthPasses` (`out/html/cat_engine.js:137`
// — the function behind `window.engineTick`) can move on the desk→next-month
// path. Verified by grepping every write in `source/`:
//
//   gdp_growth      root.scene.dry:93 (boot init) + election_simulation.scene.dry
//   welfare_index   root.scene.dry:91 (boot init) + election_simulation.scene.dry
//   unemployment    root.scene.dry:98 (boot init) + election_simulation.scene.dry
//                   + post_event.scene.dry:10-11, which are RANGE CLAMPS
//                   (`if (Q.unemployment < 0) …`) and run BEFORE the engineTick
//                   call on line 73 — no-ops for an in-range value.
//
// `root` runs once at boot (long before the snapshot below) and
// `election_simulation` is a standalone sandbox scene that the desk loop never
// enters. So no card, no event and no outcome scene writes any of these: if
// they are unchanged after the calendar has moved, the simulation did not run.
//
// Deliberately EXCLUDED, despite `monthPasses` writing them every tick:
// social_dissent (25 content writes), cat_spa_relations (21) and
// independence_movement (34) are routinely set by ordinary cards and events —
// asserting on them would let a card's own effect mask a dead engine.
//
// Each of the three is a continuous float updated with a gaussian term, and
// none starts near its clamp (see the printed values), so a spurious "unchanged"
// cannot happen by chance. Do not weaken this.
const SIM_OUTPUTS = ['gdp_growth', 'unemployment', 'welfare_index'] as const;

function simState(q: Record<string, unknown>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const k of SIM_OUTPUTS) out[k] = q[k] as number;
  return out;
}

describe('integration: the real game through the desk loop', () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    setAnimationsForTest(false);
  });

  // Compiled scene code calls `G.engineTick` / `G.spaSupportInject` (`G` is
  // `engine.gameLib`, installed by `DendryAdapter`'s constructor — see
  // game-bindings.ts). If this ever stops exporting real functions, every
  // `G.*` call throws a TypeError that dendry's `runActions` SWALLOWS, so the
  // game keeps playing and simulates nothing. This is the cheap canary; the
  // test below is the one that proves the sim actually ran.
  it('the game lib exports the functions content calls as G.*', () => {
    expect(typeof gameLib.engineTick).toBe('function');
    expect(typeof gameLib.spaSupportInject).toBe('function');
  });

  // Not seeded: `newGame()` calls `beginGame()` with no seeds, so the engine
  // uses `Random.fromUnique()` and the drawn card varies run to run. That is
  // intentional coverage here — the starting ERC party deck can surface any of
  // its available cards, and the loop below must hold for whichever one comes
  // up. Every assertion is therefore about the SHAPE of the loop (it reaches
  // the desk, it resolves, it stamps a save), never about a specific card.
  it.skipIf(!HAVE_GAME)('boots, reaches the desk, and resolves a real card end to end', () => {
    const game = useGameStore();
    const desk = useDeskStore();

    // Follow the first available choice off any page the turn produced until we
    // are back at the desk. `game.choose` is what PaperPage does (the desk store
    // is not involved off the desk); the frame watch re-derives the phase.
    const walkToDesk = (): void => {
      let pages = 0;
      while (desk.phase !== 'idle' && pages < MAX_PRE_DESK_STEPS) {
        const i = firstChoosable(game.frame!.choices);
        expect(i, `dead end at ${game.frame!.sceneId}: no choosable option`).toBeGreaterThanOrEqual(
          0,
        );
        game.choose(i);
        pages++;
      }
      expect(pages, 'the turn never routed back to the desk').toBeLessThan(MAX_PRE_DESK_STEPS);
    };

    game.initFromText(readFileSync(GAME_JSON, 'utf8'));
    expect(game.ready).toBe(true);

    game.newGame();
    expect(desk.phase).toBe('page'); // the start menu is an ordinary page

    // --- pre-desk flow: follow the first available choice until we land on a
    // scene the engine calls a desk. This walks start menu -> difficulty ->
    // party -> intro chain in the real content.
    let steps = 0;
    while (desk.phase === 'page' && steps < MAX_PRE_DESK_STEPS) {
      const i = firstChoosable(game.frame!.choices);
      expect(i, `dead end at ${game.frame!.sceneId}: no choosable option`).toBeGreaterThanOrEqual(0);
      game.choose(i);
      steps++;
    }
    expect(steps, 'pre-desk flow did not terminate within the step cap').toBeLessThan(
      MAX_PRE_DESK_STEPS,
    );
    expect(desk.phase).toBe('idle'); // the desk is the terminus of the pre-desk flow

    // --- the desk itself
    const desk0 = game.frame!;
    expect(desk0.effectiveRole).toBe('desk');
    expect(desk0.decks.length).toBeGreaterThanOrEqual(1);
    expect(desk0.isHand).toBe(true);

    const startStamp = `${game.q.year}-${game.q.month}`;
    const simBefore = simState(game.q);
    for (const k of SIM_OUTPUTS) {
      expect(typeof simBefore[k], `${k} is not a number on the starting desk`).toBe('number');
    }

    // --- draw: take from the first deck that actually yields a card (the
    // party deck, for ERC). A deck that is out of cards toasts instead of
    // dealing, which is a legal outcome here, not a failure.
    let drawn: CardView | undefined;
    for (const deck of desk0.decks) {
      const before = game.frame!.hand.length;
      desk.drawFrom(deck.id);
      if (game.frame!.hand.length > before) {
        drawn = game.frame!.hand[game.frame!.hand.length - 1];
        break;
      }
    }
    expect(drawn, 'no deck on the starting desk yielded a card').toBeDefined();
    expect(desk.phase).toBe('idle'); // animations off: the draw commits synchronously
    expect(desk.flying).toBeNull();
    expect(drawn!.role, `drawn card ${drawn!.id} has no role`).toBeDefined();

    // --- play: the card opens as a dossier (committed — there is no cancel)
    desk.playFromHand(drawn!);
    expect(desk.phase).toBe('dossierOpen');
    expect(desk.openCard?.id).toBe(drawn!.id);
    expect(game.effectiveRole.startsWith('card')).toBe(true);

    // --- pick papers until the dossier resolves back to the desk. Real cards
    // are multi-step (the first pick lands on a role-less outcome scene that
    // inherits the card role, so the dossier stays open), hence the loop.
    let papers = 0;
    while (desk.phase === 'dossierOpen' && papers < MAX_PAPERS) {
      const i = firstChoosable(game.frame!.choices);
      expect(i, `dossier dead end at ${game.frame!.sceneId}`).toBeGreaterThanOrEqual(0);
      desk.pickPaper(i);
      papers++;
    }
    expect(papers, 'card did not resolve within the paper cap').toBeLessThan(MAX_PAPERS);

    // --- the resolved card lands in the OUT tray, whatever it resolved INTO.
    // Asserted BEFORE the walk below, because with the simulation actually
    // running the resolution frequently routes off the desk (see next block) —
    // and the OUT-tray/openCard cleanup has to happen on that path too.
    expect(desk.outTray).not.toBeNull();
    expect(desk.outTray!.title).toBe(drawn!.title);
    expect(desk.openCard).toBeNull();
    expect(game.frame!.hand.map((c) => c.id)).not.toContain(drawn!.id); // it left the hand

    // --- back to the desk. The card's outcome go-to's `post_event`, whose tail
    // (timers, roadmap, §2.5 countdowns, the events_choice route) only executes
    // once `window.engineTick` exists: the TypeError from a missing binding
    // aborts the rest of that on-arrival block, and dendry swallows it. So with
    // a dead sim every card fell straight back to the hub, and this walk is
    // literally unreachable. It is reachable now — an event month routes the
    // turn through an event page before the desk returns.
    walkToDesk();
    expect(desk.phase).toBe('idle'); // back at the desk
    expect(game.effectiveRole).toBe('desk');

    // --- autosave: the desk store stamps `${year}-${month}` on every idle
    // entry and writes the auto-1/auto-2 rotation when the stamp moves. In the
    // real game a card's resolution advances the month, so this normally
    // fires on the first card; keep driving turns if a future card does not.
    let turns = 0;
    while (`${game.q.year}-${game.q.month}` === startStamp && turns < MAX_TURNS) {
      const deck = game.frame!.decks[0];
      const before = game.frame!.hand.length;
      desk.drawFrom(deck.id);
      if (game.frame!.hand.length === before) break; // deck dry; nothing more to drive
      const card = game.frame!.hand[game.frame!.hand.length - 1];
      desk.playFromHand(card);
      let guard = 0;
      while (desk.phase === 'dossierOpen' && guard < MAX_PAPERS) {
        const i = firstChoosable(game.frame!.choices);
        if (i < 0) break;
        desk.pickPaper(i);
        guard++;
      }
      walkToDesk(); // the resolution may have routed through an event page
      turns++;
    }
    expect(`${game.q.year}-${game.q.month}`, 'the month never advanced').not.toBe(startStamp);

    // --- THE SIMULATION ACTUALLY RAN.
    // A moving calendar is NOT evidence of a running game: `post_event` bumps
    // `Q.month` *before* it calls `window.engineTick(Q)`, so a missing engine
    // binding leaves the clock ticking over a frozen world — the loop looks
    // alive, autosaves stamp, and nothing simulates. This is the assertion
    // whose absence let that ship for an entire phase. Do not weaken it: if it
    // fails, the macro model is dead, not flaky.
    const simAfter = simState(game.q);
    for (const k of SIM_OUTPUTS) {
      expect(
        simAfter[k],
        `${k} is unchanged after a month passed — window.engineTick never ran, the macro simulation is dead`,
      ).not.toBe(simBefore[k]);
    }

    const autoKeys = Object.keys(localStorage).filter((k) => AUTO_SAVE_RE.test(k));
    expect(autoKeys.length, 'no rti:save:auto-* key after a month advance').toBeGreaterThanOrEqual(1);
    const autoMeta = game.listSlots().filter((s) => s.slot.startsWith('auto-'));
    expect(autoMeta[0].playerParty).toBe('erc'); // the pre-desk walk picks ERC
    expect(autoMeta[0].month).toBe(game.q.month);

    // --- role coverage: everything the real desk loop actually surfaces must
    // carry a role, which is what the Task-2 content sweep was for. Draw once
    // more so the hand is non-empty and the hand check is not vacuous.
    desk.drawFrom(game.frame!.decks[0].id);
    const final = game.frame!;
    expect(final.decks.length).toBeGreaterThanOrEqual(1);
    expect(final.pinned.length).toBeGreaterThanOrEqual(1);
    expect(final.hand.length).toBeGreaterThanOrEqual(1);
    for (const view of [...final.decks, ...final.hand, ...final.pinned]) {
      expect(view.role, `${view.id} surfaced on the desk without a role`).toBeDefined();
    }
    // Task 2 gave the deck scenes themselves a specific deck-* role (gov/
    // party/parliament, or plain `deck` for the neutral-skin fallback like
    // `main.debug_deck`) instead of the uniform `deck` every deck used to
    // carry — assert the family, not the old single literal.
    for (const deck of final.decks) expect(deck.role!.startsWith('deck')).toBe(true);
    for (const pin of final.pinned) expect(pin.role).toBe('pinned-action');
    for (const card of final.hand) expect(card.role!.startsWith('card')).toBe(true);
  });

  // REGRESSION (2026-07-13): the desk rendered ZERO in-trays against the real
  // game, so no card could ever be drawn — while every component test passed.
  // DeskView used to match decks to tray chrome by a hardcoded scene-id table,
  // and dendry prefixes a section scene with its FILE id: main.scene.dry's
  // `@party_erc` compiles to `main.party_erc`, not `party_erc`. The component
  // tests fixtured the bare id — the same wrong id the component assumed — so
  // the fixture encoded the bug and could not catch it. Only the real
  // game.json can. Mount the real thing.
  //
  // Task 2 (2026-07-19) deleted that id table: DeskView now renders one InTray
  // per `deskStore.deskView.decks` entry, generically, in option order — so
  // this test's job shifts from "does the id table know these ids" to "does
  // the real compiled game still carry a role on every deck it offers" (a
  // missing/unmapped role is what would silently produce a neutral-skinned or
  // absent tray now). Kept as a real-game guard, not deleted.
  it.skipIf(!HAVE_GAME)('renders a real in-tray at the desk (real deck ids, not fixtures)', () => {
    const game = useGameStore();
    const desk = useDeskStore();

    game.initFromText(readFileSync(GAME_JSON, 'utf8'));
    game.newGame();
    let pages = 0;
    while (desk.phase !== 'idle' && pages < MAX_PRE_DESK_STEPS) {
      game.choose(firstChoosable(game.frame!.choices));
      pages++;
    }
    expect(desk.phase).toBe('idle');
    expect(game.frame!.decks.length).toBeGreaterThanOrEqual(1);

    const wrapper = mount(DeskView, { global: { plugins: [i18n] } });
    const trays = wrapper.findAllComponents(InTray);
    expect(
      trays.length,
      `the real desk offers decks [${game
        .frame!.decks.map((d) => d.id)
        .join(', ')}] but DeskView rendered ${trays.length} in-trays — ` +
        'the generic deskView.decks -> InTray loop should render exactly one per deck',
    ).toBeGreaterThanOrEqual(1);

    // Every PLAYABLE deck the real desk offers must map to tray chrome, or it is
    // unreachable furniture. `main.debug_deck` is deliberately not a tray (the
    // design has three: GOVERNMENT, PARTY, PARLAMENT) — it rides the `debug`
    // quality and is a developer affordance, not desk furniture.
    for (const deck of game.frame!.decks) {
      if (deck.id === 'main.debug_deck') continue;
      const rendered = trays.some((t) => (t.props('deck') as { id: string }).id === deck.id);
      expect(rendered, `deck ${deck.id} is offered by the desk but has no in-tray`).toBe(true);
    }

    // Tray derivation against the real compiled game: party deck present with
    // its compiled deck-party role driving the skin — no UI-side id table.
    const ercTray = wrapper.find('[data-test="in-tray-main.party_erc"]');
    expect(ercTray.exists()).toBe(true);
    expect(ercTray.classes()).toContain('skin-party');
  });
});
