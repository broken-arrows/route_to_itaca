import { createRequire } from 'node:module';
import { beforeEach, describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
const { createSaveStore } = require('../../vendor/dendrynexus-ten/lib/persistence.js');
const { DendryEngine, NullUserInterface } = require('../../vendor/dendrynexus-ten/lib/engine.js');
const saveKey = 'dnt:integrity:save:manual-1';
const ledgerKey = 'dnt:integrity:achievements';
const store = () => createSaveStore({ storage: localStorage, ifid: 'integrity', gameVersion: '0.1.0' });
const engine = () => new DendryEngine(new NullUserInterface(), {
  info: { ifid: 'integrity' }, qualities: {},
  scenes: { root: { id: 'root', content: [], options: [] } },
});
function save() {
  const state = {
    qualities: { resources: 10, game_achievement_previous: 1 }, achievements: {},
    achievementsEligible: true, currentRandomState: [1, 2, 3, 4, 5], gameOver: true,
  };
  store().write('manual-1', state);
  return JSON.parse(store().export('manual-1').data);
}

describe('save integrity and achievement eligibility', () => {
  beforeEach(() => localStorage.clear());

  it('accepts formatting/key-order changes and awards achievements after import', () => {
    const record = save();
    function reorder(value: any): any {
      if (!value || typeof value !== 'object') return value;
      if (Array.isArray(value)) return value.map(reorder);
      return Object.fromEntries(Object.keys(value).reverse().map(k => [k, reorder(value[k])]));
    }
    store().import('manual-2', JSON.stringify(reorder(record), null, 4));
    const loaded = store().read('manual-2');
    expect(loaded.record.state.achievementsEligible).toBe(true);
    const e = engine();
    e.setState(loaded.record.state);
    e.achieve('new');
    expect(JSON.parse(localStorage.getItem(ledgerKey)!).new).toBeDefined();
  });

  it.each(['edit', 'missing', 'invalid', 'metadata'])('loads %s checksum saves without allowing awards', (kind) => {
    const record = save();
    if (kind === 'edit') record.state.qualities.resources = 100;
    if (kind === 'missing') delete record.checksum;
    if (kind === 'invalid') record.checksum = 'invalid';
    if (kind === 'metadata') record.meta.savedAt = 'edited';
    // Check local-storage loads as well as external imports.
    localStorage.setItem(saveKey, JSON.stringify(record));
    expect(store().read('manual-1').record.state.achievementsEligible).toBe(false);
    expect(store().import('manual-2', JSON.stringify(record)).ok).toBe(true);
    const loaded = store().read('manual-2');
    expect(loaded.status).toBe('ready');
    expect(loaded.record.state.achievementsEligible).toBe(false);
    localStorage.setItem(ledgerKey, JSON.stringify({ previous_local: 1 }));
    const e = engine();
    e.setState(loaded.record.state);
    e.achieve('blocked');
    expect(e.state.qualities.game_achievement_previous).toBe(1);
    expect(e.state.qualities.game_achievement_blocked).toBeUndefined();
    expect(e.state.qualities.achievement_blocked).toBeUndefined();
    expect(JSON.parse(localStorage.getItem(ledgerKey)!)).toEqual({ previous_local: 1 });
    if (kind === 'edit') expect(e.state.qualities.resources).toBe(100);

    // Fresh checksums, slot rotation and export/import cannot restore eligibility.
    store().write('auto-1', e.getExportableState());
    store().import('auto-2', store().export('auto-1').data);
    const next = engine();
    next.setState(store().read('auto-2').record.state);
    expect(next.state.achievementsEligible).toBe(false);
    next.achieve('still_blocked');
    expect(JSON.parse(localStorage.getItem(ledgerKey)!)).toEqual({ previous_local: 1 });

    next.setGameLib({ fixture: true });
    next.beginGame([1]);
    expect(next.state.achievementsEligible).toBe(true);
    next.achieve('fresh_game');
    expect(JSON.parse(localStorage.getItem(ledgerKey)!).fresh_game).toBeDefined();
  });

  it('detects editing the eligibility flag of a previously disabled save', () => {
    const record = save();
    record.state.achievementsEligible = false;
    store().write('manual-1', record.state);
    const disabled = JSON.parse(store().export('manual-1').data);
    disabled.state.achievementsEligible = true;
    store().import('manual-2', JSON.stringify(disabled));
    expect(store().read('manual-2').record.state.achievementsEligible).toBe(false);
  });
});
