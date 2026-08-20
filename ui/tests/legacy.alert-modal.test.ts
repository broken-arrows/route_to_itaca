import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const gameJs = readFileSync(resolve(import.meta.dirname, '../../out/html/game.js'), 'utf8');
const indexHtml = readFileSync(resolve(import.meta.dirname, '../../out/html/index.html'), 'utf8');

describe('old-shell alert modal', () => {
  const originalAlert = window.alert;

  afterEach(() => {
    window.alert = originalAlert;
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
});
