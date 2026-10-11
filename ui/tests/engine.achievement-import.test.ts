import { createRequire } from 'node:module';
import { beforeEach, describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
const { DendryEngine, NullUserInterface } = require('../../vendor/dendrynexus-ten/lib/engine.js');
const { createSaveStore } = require('../../vendor/dendrynexus-ten/lib/persistence.js');
const key = 'dnt:achievement-import:achievements';
const date = '2026-08-15T12:00:00.000Z';
const makeEngine = () => new DendryEngine(new NullUserInterface(), {
  info: { ifid: 'achievement-import' }, qualities: {}, scenes: {},
});

describe('achievement isolation when importing an individual save', () => {
  beforeEach(() => localStorage.clear());

  it.each([
    ['missing', null],
    ['empty', {}],
    ['existing', { local_only: { unlockedAt: date }, legacy_local: 1 }],
  ])('uses the recipient ledger when it is %s', (_label, localLedger) => {
    const donor = makeEngine();
    donor.state = {
      qualities: {}, achievements: {}, currentRandomState: [1, 2, 3, 4, 5], gameOver: true,
    };
    donor.achieve('other_run');
    delete donor.state.qualities.game_achievement_other_run;
    donor.achieve('saved_run');
    const saves = createSaveStore({ storage: localStorage, ifid: 'achievement-import' });
    expect(saves.write('manual-1', donor.getExportableState()).ok).toBe(true);
    const exported = saves.export('manual-1');
    expect(exported.ok).toBe(true);

    localStorage.clear();
    if (localLedger !== null) localStorage.setItem(key, JSON.stringify(localLedger));
    expect(saves.import('manual-1', exported.data).ok).toBe(true);
    const recipient = makeEngine();
    recipient.setState(saves.read('manual-1').record.state);

    expect(recipient.state.achievements).toEqual(localLedger ?? {});
    expect(recipient.state.qualities.achievement_other_run).toBeUndefined();
    expect(recipient.state.qualities.achievement_saved_run).toBeUndefined();
    expect(recipient.state.qualities.game_achievement_saved_run).toBe(1);
    expect(recipient.state.qualities.game_achievement_other_run).toBeUndefined();
    for (const id of Object.keys(localLedger ?? {})) {
      expect(recipient.state.qualities['achievement_' + id]).toBe(1);
    }
    expect(localStorage.getItem(key)).toBe(localLedger === null ? null : JSON.stringify(localLedger));

    recipient.achieve('new_unlock');
    const persisted = JSON.parse(localStorage.getItem(key)!);
    expect(persisted).toEqual({
      ...(localLedger ?? {}), new_unlock: { unlockedAt: expect.any(String) },
    });

    // Earning an imported run's achievement again can grant local credit.
    recipient.achieve('saved_run');
    expect(recipient.state.qualities.achievement_saved_run).toBe(1);
    expect(JSON.parse(localStorage.getItem(key)!).saved_run.unlockedAt).toEqual(expect.any(String));

    // An older checkpoint must preserve unlocks earned locally after it was made.
    recipient.setState(saves.read('manual-1').record.state);
    expect(recipient.state.qualities.achievement_new_unlock).toBe(1);
    expect(recipient.state.qualities.achievement_saved_run).toBe(1);
    expect(recipient.state.qualities.achievement_other_run).toBeUndefined();
  });
});
