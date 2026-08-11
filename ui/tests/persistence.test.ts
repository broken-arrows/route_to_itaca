import { createRequire } from 'node:module';
import { beforeEach, describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
const { createSaveStore } = require(
  '../../vendor/dendrynexus-ten/lib/persistence.js',
) as {
  createSaveStore(options: {
    storage: Storage;
    storageId: string;
    gameVersion: string;
    now: () => Date;
  }): {
    write(slot: string, state: unknown, meta?: Record<string, unknown>): any;
    read(slot: string): any;
    list(): any[];
    remove(slot: string): any;
    export(slot: string): any;
    import(slot: string, serialized: string): any;
  };
};

describe('shared save persistence', () => {
  beforeEach(() => localStorage.clear());

  it('writes and reads the canonical versioned envelope', () => {
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date('2026-08-11T10:15:00.000Z'),
    });

    expect(store.write('manual-1', { sceneId: 'root' }, { sceneId: 'root' })).toMatchObject({
      ok: true,
    });
    expect(JSON.parse(localStorage.getItem('rti:save:manual-1')!)).toEqual({
      saveFormatVersion: 1,
      gameVersion: '0.1.0',
      meta: { sceneId: 'root', savedAt: '2026-08-11T10:15:00.000Z' },
      state: { sceneId: 'root' },
    });
    expect(store.read('manual-1')).toMatchObject({
      status: 'ready',
      compatibility: 'compatible',
      record: { state: { sceneId: 'root' } },
    });
  });

  it('rejects ambiguous storage namespaces and empty slots', () => {
    for (const storageId of ['RTI', 'two words', '2rti', 'rti:other', '']) {
      expect(() =>
        createSaveStore({
          storage: localStorage,
          storageId,
          gameVersion: '0.1.0',
          now: () => new Date(),
        }),
      ).toThrow(/storageId/);
    }
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'route-to-itaca',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(() => store.read('')).toThrow(/slot/);
  });

  it.each([
    ['0.2.9-release', 'compatible'],
    ['0.3.0', 'incompatible'],
    ['0.2', 'incompatible'],
    ['0.2.1.4', 'unknown'],
    ['v0.2.1', 'unknown'],
    ['', 'unknown'],
  ])('classifies saved game version %j as %s', (savedVersion, expected) => {
    localStorage.setItem(
      'rti:save:manual-1',
      JSON.stringify({
        saveFormatVersion: 1,
        gameVersion: savedVersion,
        meta: {},
        state: {},
      }),
    );
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.2.1-beta',
      now: () => new Date(),
    });
    expect(store.read('manual-1').compatibility).toBe(expected);
  });

  it('treats missing current or saved game versions as unknown', () => {
    localStorage.setItem(
      'rti:save:manual-1',
      JSON.stringify({ saveFormatVersion: 1, meta: {}, state: {} }),
    );
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: undefined as unknown as string,
      now: () => new Date(),
    });
    expect(store.read('manual-1').compatibility).toBe('unknown');
  });

  it.each([
    ['0.2-beta', '0.9-release', 'compatible'],
    ['0.2', '1.2', 'incompatible'],
    ['0.2.1', '0.2.99-any-tag', 'compatible'],
    ['0.2.1', '0.3.0', 'incompatible'],
  ])('compares every numeric component except the last: %s vs %s', (current, saved, expected) => {
    localStorage.setItem(
      'rti:save:manual-1',
      JSON.stringify({ saveFormatVersion: 1, gameVersion: saved, meta: {}, state: {} }),
    );
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: current,
      now: () => new Date(),
    });
    expect(store.read('manual-1').compatibility).toBe(expected);
  });

  it('lists corrupt, invalid-envelope, unsupported, and healthy saves instead of hiding them', () => {
    localStorage.setItem('rti:save:broken-json', '{');
    localStorage.setItem('rti:save:raw-state', JSON.stringify({ sceneId: 'root' }));
    localStorage.setItem(
      'rti:save:future',
      JSON.stringify({ saveFormatVersion: 2, gameVersion: '0.1.0', meta: {}, state: {} }),
    );
    localStorage.setItem(
      'rti:save:manual-1',
      JSON.stringify({ saveFormatVersion: 1, gameVersion: '0.1.0', meta: {}, state: {} }),
    );
    localStorage.setItem('someone-else:save:manual-1', '{}');
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });

    expect(store.list().map((entry) => [entry.slot, entry.status, entry.error?.code])).toEqual([
      ['broken-json', 'corrupt', 'invalid-json'],
      ['future', 'unsupported', 'unsupported-save-format'],
      ['manual-1', 'ready', undefined],
      ['raw-state', 'corrupt', 'invalid-envelope'],
    ]);
  });

  it('reports a discovered record as unreadable when storage denies reading it', () => {
    const storage = {
      length: 1,
      key: () => 'rti:save:manual-1',
      getItem: () => {
        throw new Error('denied');
      },
      setItem: () => undefined,
      removeItem: () => undefined,
      clear: () => undefined,
    } as Storage;
    const store = createSaveStore({
      storage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(store.list()).toMatchObject([
      { status: 'unreadable', slot: 'manual-1', error: { code: 'storage-read-failed' } },
    ]);
  });

  it('keeps the prior record and returns a structured error when a write cannot serialize', () => {
    localStorage.setItem('rti:save:manual-1', 'prior');
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(store.write('manual-1', circular)).toMatchObject({
      ok: false,
      error: { code: 'serialize-failed' },
    });
    expect(localStorage.getItem('rti:save:manual-1')).toBe('prior');
  });

  it('does not write a state value that JSON would silently omit', () => {
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(store.write('manual-1', undefined)).toMatchObject({
      ok: false,
      error: { code: 'serialize-failed' },
    });
    expect(localStorage.getItem('rti:save:manual-1')).toBeNull();
  });

  it('keeps the prior record and returns a structured error when storage rejects a write', () => {
    const values = new Map([['rti:save:manual-1', 'prior']]);
    const storage = {
      get length() {
        return values.size;
      },
      key: (index: number) => [...values.keys()][index] ?? null,
      getItem: (key: string) => values.get(key) ?? null,
      setItem: () => {
        throw new Error('quota');
      },
      removeItem: (key: string) => values.delete(key),
      clear: () => values.clear(),
    } as Storage;
    const store = createSaveStore({
      storage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(store.write('manual-1', {})).toMatchObject({
      ok: false,
      error: { code: 'storage-write-failed' },
    });
    expect(values.get('rti:save:manual-1')).toBe('prior');
  });

  it('imports only save envelopes, strips unrelated fields, and exports only that save', () => {
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(store.import('manual-1', JSON.stringify({ sceneId: 'root' }))).toMatchObject({
      ok: false,
      error: { code: 'invalid-envelope' },
    });
    expect(
      store.import(
        'manual-1',
        JSON.stringify({
          saveFormatVersion: 1,
          gameVersion: '0.1.7',
          meta: { savedAt: 'then' },
          state: { sceneId: 'root' },
          achievements: ['nope'],
          settings: { nope: true },
        }),
      ),
    ).toMatchObject({ ok: true, status: 'ready', compatibility: 'compatible' });

    const exported = store.export('manual-1');
    expect(exported.ok).toBe(true);
    expect(JSON.parse(exported.data)).toEqual({
      saveFormatVersion: 1,
      gameVersion: '0.1.7',
      meta: { savedAt: 'then' },
      state: { sceneId: 'root' },
    });
  });

  it('does not replace an existing save when an import is not a canonical envelope', () => {
    localStorage.setItem('rti:save:manual-1', 'prior');
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(store.import('manual-1', '{')).toMatchObject({
      ok: false,
      error: { code: 'invalid-json' },
    });
    expect(localStorage.getItem('rti:save:manual-1')).toBe('prior');
  });

  it('preserves and exports an unsupported envelope but refuses to report it loadable', () => {
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    const serialized = JSON.stringify({
      saveFormatVersion: 7,
      gameVersion: '4.0',
      meta: {},
      state: {},
    });
    expect(store.import('future', serialized)).toMatchObject({
      ok: true,
      status: 'unsupported',
    });
    expect(store.read('future')).toMatchObject({
      status: 'unsupported',
      error: { code: 'unsupported-save-format', actual: 7, supported: 1 },
    });
    expect(JSON.parse(store.export('future').data)).toEqual(JSON.parse(serialized));
  });

  it('exports corrupt raw data for recovery and removes slots idempotently', () => {
    localStorage.setItem('rti:save:broken', '{');
    const store = createSaveStore({
      storage: localStorage,
      storageId: 'rti',
      gameVersion: '0.1.0',
      now: () => new Date(),
    });
    expect(store.export('broken')).toEqual({ ok: true, data: '{' });
    expect(store.remove('broken')).toEqual({ ok: true, existed: true });
    expect(store.remove('broken')).toEqual({ ok: true, existed: false });
    expect(store.read('broken')).toEqual({ status: 'missing', slot: 'broken' });
  });
});
