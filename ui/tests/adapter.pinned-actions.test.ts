import { beforeEach, describe, expect, it } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';

beforeEach(() => {
  setActivePinia(createPinia());
  setAnimationsForTest(false);
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
