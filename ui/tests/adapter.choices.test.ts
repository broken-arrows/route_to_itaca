import { describe, it, expect, beforeEach } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import { miniGameText } from './fixtures/mini-game';

describe('DendryAdapter choices', () => {
  let adapter: DendryAdapter;
  beforeEach(() => {
    adapter = DendryAdapter.fromJSONText(miniGameText);
    adapter.beginGame();
  });

  it('choose() transitions to the target scene and returns a hand frame', () => {
    const frame = adapter.choose(0); // -> desk
    expect(frame.sceneId).toBe('desk');
    expect(frame.isHand).toBe(true);
    expect(frame.maxCards).toBe(2);
    expect(frame.html).toContain('Your desk awaits.');
    // hand scenes route through decks/pinned, not plain choices
    expect(frame.choices).toEqual([]);
    expect(frame.decks).toEqual([
      expect.objectContaining({ id: 'gov_deck', title: 'Government', canChoose: true, image: 'img/gov.jpg' }),
    ]);
    expect(frame.pinned).toEqual([
      expect.objectContaining({ id: 'advisor_note', title: 'Advisor' }),
    ]);
    expect(frame.hand).toEqual([]);
  });

  it('choose() with an out-of-range index throws', () => {
    expect(() => adapter.choose(5)).toThrow();
  });

  it('frames do not leak content between scenes with newPage', () => {
    // Confirmed empirically: without `newPage: true` on `desk`, dendry
    // accumulates content across transitions (root's paragraph leaked into
    // desk's frame) — the fixture pins `desk.newPage = true` so this test
    // exercises the actual clearing behaviour it names.
    const frame = adapter.choose(0);
    expect(frame.html).not.toContain('Welcome to the mini game.');
  });

  it('content accumulates across transitions to scenes WITHOUT newPage', () => {
    // dendry only clears prose when the DESTINATION scene declares newPage: true;
    // gov_deck and card_a do not, so prose from every visited scene piles up.
    adapter.goToScene('gov_deck');
    const frame = adapter.goToScene('card_a');
    expect(frame.html).toContain('Welcome to the mini game.');
    expect(frame.html).toContain('Card A: a decision to make.');
  });
});
