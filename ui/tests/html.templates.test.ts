// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { runInNewContext } from 'node:vm';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
Object.assign(globalThis, { jQuery: () => {} });
const BrowserUI = require('../../vendor/dendrynexus-ten/lib/ui/browser.js');
delete (globalThis as { jQuery?: unknown }).jQuery;

describe('Classic formatter installation', () => {
  it.each(['default', 'default-tabbed-sidebar', 'no-sidebar', 'dendry-basic', 'custom'])(
    '%s installs on its UI without a global hook', (template) => {
      const ui = Object.create(BrowserUI.prototype);
      ui.game = { data: { glossary: { terms: [{ id: 'erc', match: ['ERC'], colour: 'erc' }] } } };
      ui.dendryEngine = { touchedSettings: true, setGameLib() {} };
      const window: Record<string, any> = { dendryUI: ui, alert() {}, RTI_GAME_LIB: {} };
      const file = template === 'custom' ? '../../out/html/game.js'
        : `../../vendor/dendrynexus-ten/lib/templates/html/${template}/+game.js`;
      runInNewContext(readFileSync(resolve(import.meta.dirname, file), 'utf8'), {
        window, console: { log() {} },
      });
      window.dendryModifyUI(ui);
      expect(window.displayText).toBeUndefined();
      const html = ui.contentToHTML.convertLine('ERC');
      if (template === 'custom') expect(html).toContain("data-term='erc'");
      else expect(html).toBe('ERC');
      const other = Object.create(BrowserUI.prototype);
      other.setTextFormatter((text: string) => `Other:${text}`);
      expect(other.contentToHTML.convertLine('ERC')).toBe('Other:ERC');
      expect(ui.contentToHTML.convertLine('ERC')).toBe(html);
    },
  );
});
