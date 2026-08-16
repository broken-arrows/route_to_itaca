import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const repo = resolve(import.meta.dirname, '..', '..');
const scene = (relative: string) =>
  readFileSync(resolve(repo, 'source', 'scenes', relative), 'utf8');

describe('Parliament pinned-action migration', () => {
  it('lists the one direct action in every difficulty hub and has no Parliament deck', () => {
    const main = scene('main.scene.dry');
    expect(main.match(/- @parlament_card/g)).toHaveLength(3);
    expect(main).not.toContain('@parlament_deck');
    expect(main).not.toContain('role: deck-parliament');
    expect(main).not.toContain('#parlament_card');
    const hubBodies = [
      main.slice(0, main.indexOf('@main_easy')),
      main.slice(main.indexOf('@main_easy'), main.indexOf('@main_hard')),
      main.slice(main.indexOf('@main_hard'), main.indexOf('\n@party_erc\n', main.indexOf('@main_hard'))),
    ];
    for (const hub of hubBodies) {
      const options = hub.split('\n').filter((line) => line.startsWith('- '));
      expect(options.at(-1)).toBe('- @parlament_card');
    }
  });

  it('keeps Party Affairs pools free of Parliament and preserves its intended player messaging', () => {
    const parliament = scene('parlament/parlament_card.scene.dry');
    expect(parliament).toContain('role: pinned-parliament');
    expect(parliament).toContain('view-if: year > 2012');
    expect(parliament).toContain('choose-if: parlament_timer <= 0 and not cat_caretaker_gov');
    expect(parliament).toContain('unavailable-subtitle: [? if cat_caretaker_gov : The Parlament is currently dissolved.?]');
    expect(parliament).toContain('The Parlament is currently dissolved.');
    expect(
      parliament.match(
        /Our lawmakers are still unready\. Next action possible in \[\+ parlament_timer \+\].*?months\..*?parlament_timer = 1: month\./g,
      ),
    ).toHaveLength(5);
    expect(parliament).toContain('Q.parlament_action_open = Q.parlament_timer <= 0');
    expect(parliament.match(/choose-if: parlament_action_open/g)).toHaveLength(5);
    expect(parliament).not.toContain('tags: erc_party cup_party parlament_card');
    expect(parliament).not.toContain('is-card: true');
    expect(parliament).not.toContain('@easy_discard');
  });

  it('uses the native old-shell pinned renderer without custom script or styles', () => {
    const index = readFileSync(resolve(repo, 'out', 'html', 'index.html'), 'utf8');
    const css = readFileSync(resolve(repo, 'out', 'html', 'game.css'), 'utf8');
    expect(index).not.toContain('pinned-actions.js');
    expect(existsSync(resolve(repo, 'out', 'html', 'pinned-actions.js'))).toBe(false);
    expect(css).not.toContain('pinned-card--parliament');
    expect(css).not.toContain('pinned-cards--actions');
  });

  it('returns an abandoned bank-tax draft to the open institution, not its entry action', () => {
    const bankTax = scene('parlament/parlament_bank_taxes.scene.dry');
    expect(bankTax).toContain('go-to: parlament_card.parlament_options');
    expect(bankTax).not.toMatch(/go-to: parlament_card\s/);
  });
});
