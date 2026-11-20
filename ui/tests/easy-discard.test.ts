import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');

function normalErc() {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  for (const id of ['root.start', 'root.normal_mode', 'root.esquerra', 'root.esquerra_2']) adapter.goToScene(id);
  adapter.goToScene('main');
  return adapter;
}
const choiceIds = (adapter: DendryAdapter) => adapter.currentFrame().choices.map(c => c.id);

describe('Return card to hand', () => {
  it('is not offered by pinned actions, which play no hand card', () => {
    const a = normalErc();
    const errors: string[] = [];
    const original = console.error;
    console.error = (...args: unknown[]) => { errors.push(args.join(' ')); };
    try {
      a.playPinnedCard('erc_shuffle_leadership_pinned');
      expect(a.currentFrame().sceneId).toBe('erc_shuffle_leadership');
      expect(choiceIds(a)).not.toContain('easy_discard');
      // A card played earlier must not be returned by a later pinned action.
      (a.engine as any).state.lastPlayedCard = (a.engine as any).game.scenes['erc_media'];
      a.goToScene('main');
      a.playPinnedCard('erc_shuffle_leadership_pinned');
      expect(choiceIds(a)).not.toContain('easy_discard');
      // Direct entry without a played card changes nothing and does not throw.
      (a.engine as any).state.lastPlayedCard = null;
      const actions = a.qualities.month_actions;
      a.goToScene('easy_discard');
      expect(a.qualities.month_actions).toBe(actions);
    } finally { console.error = original; }
    expect(errors.filter(e => /Scene (action|predicate|expression) error/.test(e))).toEqual([]);
  });

  it('returns the hand card being played and refunds its action', () => {
    const a = normalErc();
    const hand = (a.engine as any).state.currentHands.main;
    const before = a.qualities.month_actions;
    a.playCard('erc_media');
    expect(a.qualities.month_actions).toBe(before + 1);
    expect(choiceIds(a)).toContain('easy_discard');
    const index = a.currentFrame().choices.findIndex(c => c.id === 'easy_discard');
    a.choose(index);
    expect(a.qualities.month_actions).toBe(before);
    expect(hand.map((card: { id: string }) => card.id)).toContain('erc_media');
  });
});
