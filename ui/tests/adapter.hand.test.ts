import { describe, it, expect, beforeEach } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import { miniGameText } from './fixtures/mini-game';

describe('DendryAdapter hand mechanics', () => {
  let adapter: DendryAdapter;
  beforeEach(() => {
    adapter = DendryAdapter.fromJSONText(miniGameText);
    adapter.beginGame();
    adapter.choose(0); // -> desk (hand scene)
  });

  it('draws cards from a deck until the hand is full', () => {
    const first = adapter.drawCard('gov_deck');
    expect(first.result.id).toMatch(/^card_[ab]$/);
    expect(typeof first.result.title).toBe('string');
    expect(first.frame.hand).toHaveLength(1);

    const second = adapter.drawCard('gov_deck');
    expect(second.result.id).toMatch(/^card_[ab]$/);
    expect(second.result.id).not.toBe(first.result.id);
    expect(second.frame.hand).toHaveLength(2);

    const third = adapter.drawCard('gov_deck');
    expect(third.result).toEqual({ id: null, title: 'no_space_in_hand' });
    expect(third.frame.hand).toHaveLength(2);
  });

  it('playing a card enters its scene and removes it from the hand', () => {
    const { result } = adapter.drawCard('gov_deck');
    const frame = adapter.playCard(result.id as string);
    expect(frame.sceneId).toBe(result.id);
    expect(frame.choices.length).toBeGreaterThan(0);
    // resolving returns to the desk via goTo, hand now empty
    const back = adapter.choose(0); // resolve_cheap
    expect(back.sceneId).toBe('desk');
    expect(back.hand).toHaveLength(0);
    expect(adapter.qualities.gold).toBe(1); // 2 - 1
  });

  it('option gating via viewIf reacts to Q (costly option needs gold >= 2)', () => {
    const { result } = adapter.drawCard('gov_deck');
    const frame = adapter.playCard(result.id as string);
    if (result.id === 'card_a') {
      const titles = frame.choices.map((c) => c.title);
      expect(titles).toContain('Resolve loudly'); // gold is 2
    }
  });

  it('pinned cards play without touching the hand', () => {
    adapter.drawCard('gov_deck');
    const frame = adapter.playPinnedCard('advisor_note');
    expect(frame.sceneId).toBe('advisor_note');
    const back = adapter.choose(0);
    expect(back.sceneId).toBe('desk');
    expect(back.hand).toHaveLength(1); // hand untouched
  });

  it('export/import round-trips hand and qualities', () => {
    adapter.drawCard('gov_deck');
    const saved = adapter.exportStateJSON();

    // mutate after saving
    adapter.drawCard('gov_deck');
    expect(adapter.currentFrame().hand).toHaveLength(2);

    const restored = adapter.importStateJSON(saved);
    expect(restored.sceneId).toBe('desk');
    expect(restored.hand).toHaveLength(1);
    expect(adapter.qualities.gold).toBe(2);

    // saved string is a snapshot, not a live reference
    adapter.drawCard('gov_deck');
    expect(JSON.parse(saved).currentHands.desk).toHaveLength(1);
  });
});
