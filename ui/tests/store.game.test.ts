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

  it('migrates pre-rename rti:desk:save:* slots to dnt:save:* at store creation', () => {
    localStorage.clear();
    // A save written by a phases-1–2.5 build, under the old game-named prefix.
    const legacyBlob = JSON.stringify({ meta: { slot: 'old' }, state: {} });
    localStorage.setItem('rti:desk:save:old', legacyBlob);
    // A slot that exists under BOTH prefixes: the new one must never be
    // overwritten by the migration.
    localStorage.setItem('rti:desk:save:both', JSON.stringify({ meta: { slot: 'stale' }, state: {} }));
    const freshBlob = JSON.stringify({ meta: { slot: 'both' }, state: {} });
    localStorage.setItem('dnt:save:both', freshBlob);

    useGameStore();

    expect(localStorage.getItem('dnt:save:old')).toBe(legacyBlob);
    expect(localStorage.getItem('dnt:save:both')).toBe(freshBlob);
    localStorage.clear();
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
