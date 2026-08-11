import { createRequire } from 'node:module';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const require = createRequire(import.meta.url);
(globalThis as any).jQuery = vi.fn();

const BrowserUserInterface = require(
  '../../vendor/dendrynexus-ten/lib/ui/browser.js',
) as {
  prototype: Record<string, any>;
  canonicalSlot(slot: string | number): string;
};
const { createSaveStore } = require(
  '../../vendor/dendrynexus-ten/lib/persistence.js',
) as {
  createSaveStore(options: Record<string, unknown>): any;
};

function makeUi(gameVersion = '0.1.0') {
  const ui = Object.create(BrowserUserInterface.prototype);
  ui.max_slots = 8;
  ui.DateOptions = {};
  ui.saveStore = createSaveStore({
    storage: localStorage,
    storageId: 'rti',
    gameVersion,
    now: () => new Date('2026-08-11T12:00:00.000Z'),
  });
  ui.dendryEngine = {
    state: { sceneId: 'parlament_digital.vote_result' },
    getExportableState: vi.fn(() => ({ sceneId: 'parlament_digital.vote_result', qualities: {} })),
    setState: vi.fn(),
  };
  ui.hideSaveSlots = vi.fn();
  ui.populateSaveSlots = vi.fn();
  return ui;
}

describe('old-shell persistence adapter', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    vi.spyOn(window, 'alert').mockImplementation(() => undefined);
    vi.spyOn(window, 'confirm').mockReturnValue(true);
  });

  it('keeps zero-based DOM ids behind one-based canonical slots', () => {
    expect(BrowserUserInterface.canonicalSlot('a0')).toBe('auto-1');
    expect(BrowserUserInterface.canonicalSlot('a1')).toBe('auto-2');
    expect(BrowserUserInterface.canonicalSlot(0)).toBe('manual-1');
    expect(BrowserUserInterface.canonicalSlot(7)).toBe('manual-8');
  });

  it('writes the shared envelope and unwraps state before loading', () => {
    const ui = makeUi();
    ui.saveSlot(0);

    const envelope = JSON.parse(localStorage.getItem('rti:save:manual-1')!);
    expect(envelope).toMatchObject({
      saveFormatVersion: 1,
      gameVersion: '0.1.0',
      meta: {
        savedAt: '2026-08-11T12:00:00.000Z',
        sceneId: 'parlament_digital.vote_result',
      },
    });

    ui.loadSlot(0);
    expect(ui.dendryEngine.setState).toHaveBeenCalledWith(envelope.state);
  });

  it('rotates the newest autosave to auto-2 without changing its envelope', () => {
    const ui = makeUi();
    ui.dendryEngine.getExportableState
      .mockReturnValueOnce({ sceneId: 'first' })
      .mockReturnValueOnce({ sceneId: 'second' });
    ui.dendryEngine.state.sceneId = 'first';
    ui.autosave();
    const firstEnvelope = localStorage.getItem('rti:save:auto-1');

    ui.dendryEngine.state.sceneId = 'second';
    ui.autosave();

    expect(localStorage.getItem('rti:save:auto-2')).toBe(firstEnvelope);
    expect(JSON.parse(localStorage.getItem('rti:save:auto-1')!).state.sceneId).toBe('second');
  });

  it('requires confirmation for an incompatible game version', () => {
    const ui = makeUi('0.2.0');
    localStorage.setItem('rti:save:manual-1', JSON.stringify({
      saveFormatVersion: 1,
      gameVersion: '0.3.0',
      meta: { savedAt: '2026-08-11T12:00:00.000Z', sceneId: 'older' },
      state: { sceneId: 'older' },
    }));
    vi.mocked(window.confirm).mockReturnValue(false);

    ui.loadSlot(0);
    expect(ui.dendryEngine.setState).not.toHaveBeenCalled();

    vi.mocked(window.confirm).mockReturnValue(true);
    ui.loadSlot(0);
    expect(ui.dendryEngine.setState).toHaveBeenCalledWith({ sceneId: 'older' });
  });

  it('renders corrupt saves as occupied, deletable, and exportable', () => {
    const ui = makeUi();
    ui.populateSaveSlots = BrowserUserInterface.prototype.populateSaveSlots;
    localStorage.setItem('rti:save:auto-1', '{bad json');
    document.body.innerHTML = `
      <span id="save_info_a0"></span>
      <button id="save_button_a0"></button>
      <button id="delete_button_a0"></button>
      <button id="export_button_a0"></button>
    `;

    ui.populateSaveSlots(0, 1);

    expect(document.getElementById('save_info_a0')!.textContent).toBe('Corrupt save');
    expect((document.getElementById('save_button_a0') as HTMLButtonElement).disabled).toBe(true);
    expect((document.getElementById('delete_button_a0') as HTMLButtonElement).disabled).toBe(false);
    expect((document.getElementById('export_button_a0') as HTMLButtonElement).disabled).toBe(false);
  });

  it('stores all old-shell settings in the canonical settings-old record', () => {
    const ui = makeUi();
    Object.assign(ui, {
      settings_key: 'rti:settings-old',
      animate: true,
      disable_bg: false,
      animate_bg: true,
      show_portraits: false,
      disable_audio: true,
      dark_mode: true,
    });

    ui.saveSettings();
    expect(JSON.parse(localStorage.getItem('rti:settings-old')!)).toEqual({
      animate: true,
      disableBg: false,
      animateBg: true,
      showPortraits: false,
      disableAudio: true,
      darkMode: true,
    });

    Object.assign(ui, { animate: false, show_portraits: true, disable_audio: false });
    ui.loadSettings();
    expect(ui.animate).toBe(true);
    expect(ui.show_portraits).toBe(false);
    expect(ui.disable_audio).toBe(true);
  });
});
