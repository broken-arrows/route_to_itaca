import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const gameJs = readFileSync(resolve(import.meta.dirname, '../../out/html/game.js'), 'utf8');
const indexHtml = readFileSync(resolve(import.meta.dirname, '../../out/html/index.html'), 'utf8');

describe('old-shell alert modal', () => {
  const originalAlert = window.alert;

  afterEach(() => {
    window.alert = originalAlert;
    Reflect.deleteProperty(window, 'showClassicMessage');
    document.body.replaceChildren();
    vi.restoreAllMocks();
  });

  it('uses a compact accessible dialog without a visible heading', () => {
    expect(indexHtml).toContain('id="message_dialog"');
    expect(indexHtml).toContain('aria-label="Message"');
    expect(indexHtml).not.toContain('message_dialog_title');
    expect(indexHtml).not.toContain('>Notice<');
  });

  it('renders alert text in the reusable dialog and opens it modally', () => {
    document.body.innerHTML = `
      <dialog id="message_dialog">
        <p id="message_dialog_text"></p>
      </dialog>
    `;
    const dialog = document.getElementById('message_dialog') as HTMLDialogElement;
    const showModal = vi.fn();
    dialog.showModal = showModal;

    new Function(gameJs)();
    window.alert('Import failed.\nUnreadable save.');

    expect(document.getElementById('message_dialog_text')?.textContent).toBe(
      'Import failed.\nUnreadable save.',
    );
    expect(showModal).toHaveBeenCalledOnce();
  });

  it('only shows the download action on messages that supply it', () => {
    document.body.innerHTML = `
      <dialog id="message_dialog">
        <p id="message_dialog_text"></p>
        <button type="button" id="message_dialog_download" hidden>Download</button>
      </dialog>
    `;
    const dialog = document.getElementById('message_dialog') as HTMLDialogElement;
    dialog.showModal = vi.fn();
    const download = document.getElementById('message_dialog_download') as HTMLButtonElement;
    const action = vi.fn();

    new Function(gameJs + '\nwindow.showClassicMessage("Save failed.", arguments[0]);')(action);
    expect(download.hidden).toBe(false);
    download.click();
    expect(action).toHaveBeenCalledOnce();

    window.alert('Loaded.');
    expect(download.hidden).toBe(true);
    expect(download.onclick).toBeNull();
  });
});
