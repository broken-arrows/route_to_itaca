import { afterEach, describe, it, expect, beforeEach, vi } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useGameStore } from '../src/stores/game';
import { miniGame, miniGameText } from './fixtures/mini-game';

describe('game store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
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

  it('allocates the lowest free unbounded manual slot and reuses gaps', () => {
    const store = bootedStore();
    for (let n = 1; n <= 12; n++) {
      expect(store.createManualSave()).toMatchObject({ ok: true, slot: `manual-${n}` });
    }
    store.removeSlot('manual-3');

    expect(store.createManualSave()).toMatchObject({ ok: true, slot: 'manual-3' });
  });

  it('lists autosaves first and manual saves by descending savedAt', () => {
    vi.useFakeTimers();
    const store = bootedStore();

    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'));
    store.saveSlot('manual-8');
    vi.setSystemTime(new Date('2026-01-03T00:00:00.000Z'));
    store.saveSlot('manual-2');
    vi.setSystemTime(new Date('2026-01-04T00:00:00.000Z'));
    store.saveSlot('auto-2');
    vi.setSystemTime(new Date('2026-01-02T00:00:00.000Z'));
    store.saveSlot('manual-1');
    vi.setSystemTime(new Date('2026-01-05T00:00:00.000Z'));
    store.saveSlot('auto-1');

    expect(store.listSlots().map(({ slot }) => slot)).toEqual([
      'auto-1',
      'auto-2',
      'manual-2',
      'manual-1',
      'manual-8',
    ]);
  });

  it('requires confirmation to overwrite an occupied manual record', () => {
    const store = bootedStore();
    store.choose(0);
    store.createManualSave();
    const before = store.exportSlot('manual-1');
    store.draw('gov_deck');

    expect(store.overwriteManualSave('manual-1')).toEqual({
      ok: false,
      status: 'confirmation-required',
      slot: 'manual-1',
    });
    expect(store.exportSlot('manual-1')).toEqual(before);
    expect(store.overwriteManualSave('manual-1', true)).toMatchObject({
      ok: true,
      slot: 'manual-1',
    });
    expect(store.loadSlot('manual-1')).toEqual({ status: 'loaded' });
    expect(store.frame?.hand).toHaveLength(1);
  });

  it('imports into the lowest free manual slot without loading it', () => {
    const store = bootedStore();
    store.choose(0);
    store.createManualSave();
    store.draw('gov_deck');
    store.saveSlot('manual-3');
    const imported = store.exportSlot('manual-3');
    expect(imported.ok).toBe(true);
    store.draw('gov_deck');
    expect(store.frame?.hand).toHaveLength(2);

    const result = store.importManualSave(imported.ok ? imported.data : '');

    expect(result).toMatchObject({ ok: true, status: 'ready', slot: 'manual-2' });
    expect(store.frame?.hand).toHaveLength(2);
    expect(store.listSlots().some(({ slot }) => slot === 'manual-2')).toBe(true);
  });

  it('keeps unsupported and corrupt records exportable and removable', () => {
    const store = bootedStore();
    localStorage.setItem('test-game:save:broken', '{');
    localStorage.setItem('test-game:save:future', JSON.stringify({
      saveFormatVersion: 99,
      gameVersion: '0.1.0',
      meta: { savedAt: '2030-01-01T00:00:00.000Z' },
      state: {},
    }));

    expect(store.listSlots()).toEqual(expect.arrayContaining([
      expect.objectContaining({ slot: 'broken', status: 'corrupt' }),
      expect.objectContaining({ slot: 'future', status: 'unsupported' }),
    ]));
    expect(store.exportSlot('broken')).toEqual({ ok: true, data: '{' });
    expect(store.removeSlot('broken')).toEqual({ ok: true, existed: true });
    expect(store.removeSlot('future')).toEqual({ ok: true, existed: true });
  });

  it('uses one in-place auto-1 and blocks manual operations during an ironman run', () => {
    const ironman = structuredClone(miniGame);
    ironman.scenes.root.onArrival = [{
      $code: "Q.gold = 2; Q.month = 8; Q.year = 2012; Q.player_party = 'erc'; this.state.disableSaves = true;",
    }];
    const store = useGameStore();
    store.initFromText(JSON.stringify(ironman));
    store.newGame();

    expect(store.savesDisabled).toBe(true);
    expect(store.saveAutosave()).toMatchObject({ ok: true });
    const first = store.exportSlot('auto-1');
    store.choose(0);
    const stale = localStorage.getItem('test-game:save:auto-1');
    localStorage.setItem('test-game:save:auto-2', stale!); // prior non-ironman rollback
    expect(store.saveAutosave()).toMatchObject({ ok: true });
    expect(store.listSlots().map(({ slot }) => slot)).toEqual(['auto-1']);
    expect(store.exportSlot('auto-1')).not.toEqual(first);

    expect(store.createManualSave()).toMatchObject({ ok: false, error: { code: 'saves-disabled' } });
    expect(store.overwriteManualSave('manual-1', true)).toMatchObject({
      ok: false,
      error: { code: 'saves-disabled' },
    });
    expect(store.saveSlot('manual-9')).toMatchObject({ ok: false, error: { code: 'saves-disabled' } });
    expect(store.saveSlot('auto-2')).toMatchObject({ ok: false, error: { code: 'saves-disabled' } });
    expect(store.loadSlot('manual-1')).toMatchObject({ status: 'blocked', error: { code: 'saves-disabled' } });
    expect(store.importManualSave('{}')).toMatchObject({ ok: false, error: { code: 'saves-disabled' } });
  });

  it('allows Continue to load an ironman auto-1 from the title state', () => {
    const active = bootedStore();
    active.adapter!.engine.state.disableSaves = true;
    active.saveAutosave();

    setActivePinia(createPinia());
    const title = useGameStore();
    title.initFromText(miniGameText);
    title.newGame();

    expect(title.savesDisabled).toBe(false);
    expect(title.loadSlot('auto-1')).toEqual({ status: 'loaded' });
    expect(title.savesDisabled).toBe(true);
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
