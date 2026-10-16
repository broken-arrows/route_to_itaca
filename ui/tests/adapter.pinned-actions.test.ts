import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createPinia, setActivePinia } from 'pinia';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

beforeEach(() => {
  setActivePinia(createPinia());
  setAnimationsForTest(false);
});

describe('real role-only card and pinned leadership behavior', () => {
  const currentText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
  // The full pre/post compilation comparison establishes these are the only
  // changed fields. Restore them here to exercise the original mechanics.
  const legacy = JSON.parse(currentText);
  for (const party of ['erc', 'cup']) legacy.scenes[`${party}_shuffle_leadership_pinned`].isCard = true;
  const legacyText = JSON.stringify(legacy);

  afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

  function choose(adapter: DendryAdapter, id: string) {
    const index = adapter.currentFrame().choices.findIndex(choice => choice.id === id && choice.canChoose);
    expect(index, `unavailable ${id} at ${adapter.currentFrame().sceneId}`).toBeGreaterThanOrEqual(0);
    return adapter.choose(index);
  }

  function start(text: string, party: string, mode: string) {
    const adapter = DendryAdapter.fromJSONText(text);
    adapter.beginGame([11, 22, 33, 44, 55]);
    // Setup's quote still uses Math.random directly. Fix only that selection;
    // gameplay and deck draws below use their real, independently seeded RNGs.
    const quoteRandom = vi.spyOn(Math, 'random').mockReturnValue(0.25);
    try { choose(adapter, 'root.start'); } finally { quoteRandom.mockRestore(); }
    choose(adapter, `root.${mode}_mode`);
    choose(adapter, party === 'erc' ? 'root.esquerra' : 'root.cup');
    if (party === 'cup') {
      // CUP remains disabled in production setup. Probe its authored initializer
      // directly without changing the production gate or inventing Q defaults.
      expect(adapter.currentFrame().choices.find(choice => choice.id === 'root.cup_2')?.canChoose).toBe(false);
      adapter.goToScene('root.cup_2');
    } else choose(adapter, 'root.esquerra_2');
    for (let step = 0; !adapter.currentFrame().isHand && step < 40; step++) {
      const frame = adapter.currentFrame();
      const preferred = frame.choices.findIndex(choice => choice.id === 'intros.what_now' && choice.canChoose);
      const index = preferred >= 0 ? preferred : frame.choices.findIndex(choice => choice.canChoose);
      expect(index, `dead end at ${frame.sceneId}`).toBeGreaterThanOrEqual(0);
      adapter.choose(index);
    }
    expect(adapter.currentFrame().isHand).toBe(true);
    return adapter;
  }

  function snapshot(adapter: DendryAdapter) {
    return {
      state: JSON.parse(adapter.exportStateJSON()),
      random: adapter.engine.random.getState(),
      frame: adapter.currentFrame(),
    };
  }

  it.each(['erc', 'cup'].flatMap(party => ['easy', 'normal', 'hard'].map(mode => [party, mode])))(
    '%s / %s preserves draws, pinned eligibility, state, both RNG streams and restoration', (party, mode) => {
      localStorage.clear();
      vi.useFakeTimers({ toFake: ['Date'] });
      vi.setSystemTime(new Date('2026-10-16T12:00:00Z'));
      const originalInit = gameLib.initRandom;
      vi.spyOn(gameLib, 'initRandom').mockImplementation((q, seed) => originalInit(q, seed ?? 12345));
      const errors = vi.spyOn(console, 'error');
      const before = start(legacyText, party, mode);
      const after = start(currentText, party, mode);
      expect(snapshot(after)).toEqual(snapshot(before));

      const pinnedId = `${party}_shuffle_leadership_pinned`;
      expect(after.currentFrame().pinned.some(card => card.id === pinnedId)).toBe(mode !== 'hard');
      const deck = after.currentFrame().decks.find(deck => deck.role === 'deck-party' && deck.canChoose)!;
      expect(deck).toBeDefined();
      const drawnBefore = before.drawCard(deck.id);
      const drawnAfter = after.drawCard(deck.id);
      expect(drawnAfter).toEqual(drawnBefore);
      expect(drawnAfter.result.id).toBeTruthy();
      expect(drawnAfter.result.id).not.toBe(pinnedId);
      expect(snapshot(after)).toEqual(snapshot(before));

      const savedBefore = before.exportStateJSON(), savedAfter = after.exportStateJSON();
      if (mode !== 'hard') {
        before.playPinnedCard(pinnedId);
        after.playPinnedCard(pinnedId);
        expect(after.currentFrame().sceneId).toBe(`${party}_shuffle_leadership`);
        expect(after.currentFrame().effectiveRole).toBe('card-party');
        expect(after.currentFrame().choices.some(choice => choice.title.includes('Remove advisors'))).toBe(true);
        expect(snapshot(after)).toEqual(snapshot(before));
      }
      before.importStateJSON(savedBefore);
      after.importStateJSON(savedAfter);
      expect(after.currentFrame().hand).toHaveLength(1);
      expect(snapshot(after)).toEqual(snapshot(before));
      const playedId = after.currentFrame().hand[0].id;
      before.playCard(playedId);
      after.playCard(playedId);
      expect(snapshot(after)).toEqual(snapshot(before));
      expect(errors).not.toHaveBeenCalled();
    },
  );
});

function pinnedGame(role: 'pinned-advisor' | 'pinned-parliament', canChoose: boolean) {
  return {
    scenes: {
      root: {
        id: 'root', type: 'scene', title: 'Desk', role: 'desk', isHand: true,
        content: [], options: [{ id: '@action' }, ...(canChoose ? [] : [{ id: '@available' }])],
      },
      action: {
        id: 'action', type: 'scene', title: 'Institution', role, isPinnedCard: true,
        chooseIf: { $code: `return ${canChoose};` },
        unavailableSubtitle: ['Institution unavailable for four weeks.'],
        content: [{ type: 'paragraph', content: ['Arbitrary authored document content.'] }],
        options: [],
      },
      available: {
        id: 'available', type: 'scene', title: 'Available advisor', role: 'pinned-advisor',
        isPinnedCard: true, content: [], options: [],
      },
    },
    qualities: {}, qdisplays: {}, tagLookup: {},
  };
}

function redirectingPinnedGame() {
  const game = pinnedGame('pinned-parliament', true);
  const scenes = game.scenes as Record<string, Record<string, unknown>>;
  scenes.action.content = [];
  scenes.action.goTo = [{ id: 'action.options', if: { $code: 'return true;' } }];
  scenes['action.options'] = {
    id: 'action.options', type: 'scene', title: 'Parliament options',
    content: [{ type: 'paragraph', content: ['Choose parliamentary business.'] }], options: [],
  };
  return game;
}

describe('pinned action view contract', () => {
  it('retains pinned eligibility and authored unavailable copy in CardView', () => {
    const game = useGameStore();
    game.initFromText(JSON.stringify(pinnedGame('pinned-parliament', false)));
    game.newGame();
    expect(game.frame?.pinned.find((card) => card.role === 'pinned-parliament')).toEqual(
      expect.objectContaining({
        role: 'pinned-parliament', canChoose: false,
        subtitle: 'Institution unavailable for four weeks.',
      }),
    );
  });

  it.each(['pinned-advisor', 'pinned-parliament'] as const)(
    'routes %s through the unchanged generic dossier host',
    (role) => {
      const game = useGameStore();
      const desk = useDeskStore();
      game.initFromText(JSON.stringify(pinnedGame(role, true)));
      game.newGame();
      expect(desk.phase).toBe('idle');
      desk.playPinned(game.frame!.pinned[0]);
      expect(desk.phase).toBe('dossierOpen');
      expect(game.frame?.html).toContain('Arbitrary authored document content.');
    },
  );

  it('keeps the pinned role when the action root immediately redirects to a role-less child', () => {
    const game = useGameStore();
    const desk = useDeskStore();
    game.initFromText(JSON.stringify(redirectingPinnedGame()));
    game.newGame();
    desk.playPinned(game.frame!.pinned[0]);
    expect(game.frame?.sceneId).toBe('action.options');
    expect(game.effectiveRole).toBe('pinned-parliament');
    expect(desk.phase).toBe('dossierOpen');
  });
});
