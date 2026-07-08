import { describe, it, expect, beforeEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useGameStore } from '../src/stores/game';
import { miniGameText } from './fixtures/mini-game';

describe('game store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    localStorage.clear();
  });

  function bootedStore() {
    const store = useGameStore();
    store.initFromText(miniGameText);
    store.newGame();
    return store;
  }

  it('exposes frames reactively', () => {
    const store = bootedStore();
    expect(store.ready).toBe(true);
    expect(store.frame?.sceneId).toBe('root');
    store.choose(0);
    expect(store.frame?.sceneId).toBe('desk');
  });

  it('q snapshot updates after actions (version tick)', () => {
    const store = bootedStore();
    expect(store.q.gold).toBe(2);
    store.choose(0);
    const { result } = { result: store.draw('gov_deck') };
    store.play(result.id as string);
    store.choose(0); // resolve_cheap: gold -= 1
    expect(store.q.gold).toBe(1);
  });

  it('save and load slots with metadata', () => {
    const store = bootedStore();
    store.choose(0);
    store.draw('gov_deck');
    store.saveSlot('test');

    const slots = store.listSlots();
    expect(slots).toHaveLength(1);
    expect(slots[0]).toMatchObject({
      slot: 'test',
      year: 2012,
      month: 8,
      playerParty: 'erc',
      sceneId: 'desk',
    });

    // mutate, then restore
    store.draw('gov_deck');
    expect(store.frame?.hand).toHaveLength(2);
    expect(store.loadSlot('test')).toBe(true);
    expect(store.frame?.hand).toHaveLength(1);
  });

  it('loadSlot returns false for missing slot', () => {
    const store = bootedStore();
    expect(store.loadSlot('nope')).toBe(false);
  });
});
