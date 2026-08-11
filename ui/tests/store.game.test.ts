import { describe, it, expect, beforeEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useGameStore } from '../src/stores/game';
import { miniGame, miniGameText } from './fixtures/mini-game';

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
    expect(store.saveSlot('test')).toMatchObject({ ok: true });

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
    expect(store.loadSlot('test')).toEqual({ status: 'loaded' });
    expect(store.frame?.hand).toHaveLength(1);
  });

  it('loadSlot reports a missing slot', () => {
    const store = bootedStore();
    expect(store.loadSlot('nope')).toMatchObject({ status: 'missing' });
  });

  it('hard-cuts provisional save shelves without migrating or deleting them', () => {
    localStorage.setItem('rti:desk:save:old', JSON.stringify({ meta: {}, state: {} }));
    localStorage.setItem('dnt:save:old', JSON.stringify({ meta: {}, state: {} }));
    const store = bootedStore();

    expect(store.listSlots()).toEqual([]);
    expect(localStorage.getItem('rti:desk:save:old')).not.toBeNull();
    expect(localStorage.getItem('dnt:save:old')).not.toBeNull();
  });

  it('retains corrupt saves in the list and refuses to load them', () => {
    const store = bootedStore();
    localStorage.setItem('test-game:save:manual-1', '{');
    expect(store.listSlots()).toMatchObject([
      { slot: 'manual-1', status: 'corrupt', error: { code: 'invalid-json' } },
    ]);
    expect(store.loadSlot('manual-1')).toMatchObject({ status: 'corrupt' });
  });

  it('requires explicit confirmation before a game-version-incompatible load', () => {
    const store = bootedStore();
    store.choose(0);
    store.saveSlot('manual-1');
    const saved = JSON.parse(localStorage.getItem('test-game:save:manual-1')!);
    saved.gameVersion = '0.2.0';
    localStorage.setItem('test-game:save:manual-1', JSON.stringify(saved));

    expect(store.loadSlot('manual-1')).toEqual({
      status: 'confirmation-required',
      compatibility: 'incompatible',
    });
    expect(store.loadSlot('manual-1', true)).toEqual({ status: 'loaded' });
  });

  it('glossary is empty before boot and empty with no data.glossary registry', () => {
    const store = useGameStore();
    expect(store.glossary).toEqual([]); // no adapter yet
    store.initFromText(miniGameText);
    store.newGame();
    expect(store.glossary).toEqual([]); // miniGame carries no data.glossary
  });

  it('glossary surfaces game.json.data.glossary.terms once compiled in', () => {
    const store = useGameStore();
    const withGlossary = {
      ...miniGame,
      data: { glossary: { terms: [{ id: 'ciu', match: ['CiU'], colour: 'ciu' }] } },
    };
    store.initFromText(JSON.stringify(withGlossary));
    store.newGame();
    expect(store.glossary).toEqual([{ id: 'ciu', match: ['CiU'], colour: 'ciu' }]);
  });
});
