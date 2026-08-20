import { beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const WIDGETS_JS = readFileSync(
  path.join(__dirname, '..', '..', 'out', 'html', 'widgets.js'),
  'utf8',
);
const GAME_CSS = readFileSync(
  path.join(__dirname, '..', '..', 'out', 'html', 'game.css'),
  'utf8',
);

describe('the old shell chamber-vote handler', () => {
  beforeEach(() => {
    (window as any).dendryUI = { game: { data: {} } };
    (window as any).initCataloniaPolls = vi.fn();
    (window as any).initCatLocalMap = vi.fn();
    (window as any).initCongresoMap = vi.fn();
    (window as any).initCatCoalitions = vi.fn();
    (window as any).initCongresoPartyTour = vi.fn();
    (window as any).applyWholesome = vi.fn(
      (name: string) => `<span data-wholesome="${name}">${name}</span>`,
    );
    window.eval(WIDGETS_JS);
  });

  it('resolves configFrom and keeps party breakdowns optional', () => {
    document.body.innerHTML =
      '<main><div data-widget="chamber-vote" ' +
      'data-props=\'{"configFrom":"result"}\'></div></main>';
    const q = {
      result: {
        outcomes: [
          {
            kind: 'yes',
            label: 'Yes',
            votes: 84,
            parties: [{ label: 'CiU' }, { label: 'PSC', count: 3 }],
          },
          { kind: 'abstain', label: 'Abstention', votes: 0 },
          { kind: 'no', label: 'No', votes: 51 },
        ],
      },
    };

    (window as any).mountWidgets(document.querySelector('main'), q);

    const outcomes = document.querySelectorAll('.chamber-vote__outcome');
    expect(outcomes).toHaveLength(2);
    expect((outcomes[0] as HTMLElement).style.flexGrow).toBe('84');
    expect(document.querySelector('.chamber-vote__breakdown--no .chamber-vote__parties')).toBeNull();
    expect(document.body.textContent).toContain('PSC (3)');
  });

  it('passes every generated party name through applyWholesome', () => {
    document.body.innerHTML =
      '<main><div data-widget="chamber-vote" ' +
      'data-props=\'{"outcomes":[{"kind":"yes","label":"Yes","votes":1,' +
      '"parties":[{"label":"CiU"},{"label":"ERC"}]}]}\'></div></main>';

    (window as any).mountWidgets(document.querySelector('main'), {});

    expect((window as any).applyWholesome).toHaveBeenCalledTimes(2);
    expect((window as any).applyWholesome).toHaveBeenNthCalledWith(1, 'CiU');
    expect((window as any).applyWholesome).toHaveBeenNthCalledWith(2, 'ERC');
    expect(document.querySelector('[data-wholesome="CiU"]')).not.toBeNull();
  });

  it('aligns the readable abstention column with its proportional bar segment', () => {
    document.body.innerHTML =
      '<main><div data-widget="chamber-vote" ' +
      'data-props=\'{"outcomes":[' +
      '{"kind":"yes","label":"Yes","votes":84},' +
      '{"kind":"abstain","label":"Abstention","votes":15,"parties":[{"label":"PSC"}]},' +
      '{"kind":"no","label":"No","votes":51}]}\'></div></main>';

    (window as any).mountWidgets(document.querySelector('main'), {});

    const shift = parseFloat(
      (document.querySelector('.chamber-vote') as HTMLElement).style.getPropertyValue(
        '--chamber-vote-abstain-shift',
      ),
    );
    expect(shift).toBeCloseTo(33);
  });

  it('marks the absolute majority of all votes, including abstentions', () => {
    document.body.innerHTML =
      '<main><div data-widget="chamber-vote" ' +
      'data-props=\'{"outcomes":[' +
      '{"kind":"yes","label":"Yes","votes":60},' +
      '{"kind":"abstain","label":"Abstention","votes":20},' +
      '{"kind":"no","label":"No","votes":55}]}\'></div></main>';

    (window as any).mountWidgets(document.querySelector('main'), {});

    const bar = document.querySelector('.chamber-vote__bar') as HTMLElement;
    // The full 135-seat chamber requires 68 yes votes.
    expect(
      parseFloat(bar.style.getPropertyValue('--chamber-vote-majority-left')),
    ).toBeCloseTo((68 / 135) * 100);
    expect(bar.title).toBe('Majority: 68 yes votes');
    expect(bar.dataset.majority).toBe('68');
  });

  it('lets hover pass through overlapping breakdown boxes to actual tooltip labels', () => {
    expect(GAME_CSS).toMatch(
      /\.chamber-vote__breakdown\s*{[^}]*pointer-events:\s*none;/s,
    );
    expect(GAME_CSS).toMatch(
      /\.chamber-vote__breakdown \.mytooltip\s*{[^}]*pointer-events:\s*auto;/s,
    );
  });
});
