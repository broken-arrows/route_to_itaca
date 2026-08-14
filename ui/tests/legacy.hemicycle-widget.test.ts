import { beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const WIDGETS_JS = readFileSync(
  path.join(__dirname, '..', '..', 'out', 'html', 'widgets.js'),
  'utf8',
);

describe('the old shell hemicycle handler', () => {
  let fromCenter: ReturnType<typeof vi.fn>;
  let smallToBig: ReturnType<typeof vi.fn>;
  let renderedData: unknown;

  beforeEach(() => {
    const enter = {
      fromCenter: vi.fn(),
      smallToBig: vi.fn(),
    };
    enter.fromCenter.mockReturnValue(enter);
    const exit = {
      toCenter: vi.fn(),
      bigToSmall: vi.fn(),
    };
    exit.toCenter.mockReturnValue(exit);
    fromCenter = enter.fromCenter;
    smallToBig = vi.fn();
    enter.smallToBig = smallToBig;
    const parliament = Object.assign(vi.fn(), {
      width: vi.fn().mockReturnThis(),
      height: vi.fn().mockReturnThis(),
      innerRadiusCoef: vi.fn().mockReturnThis(),
      enter,
      exit,
      highlightedParty: vi.fn(),
    });
    (window as any).d3 = {
      parliament: vi.fn(() => parliament),
      select: vi.fn(() => ({
        datum(data: unknown) {
          renderedData = data;
          return { call: vi.fn() };
        },
      })),
    };
    (window as any).dendryUI = { game: { data: {} } };
    (window as any).applyWholesome = vi.fn((value: string) => value);
    (window as any).initCataloniaPolls = vi.fn();
    (window as any).initCatLocalMap = vi.fn();
    (window as any).initCongresoMap = vi.fn();
    (window as any).initCongresoPartyTour = vi.fn();
    window.eval(WIDGETS_JS);
  });

  it.each([
    { animate: false, expected: false },
    { animate: true, expected: true },
  ])('reconstructs the SVG and preserves animate=$animate', ({ animate, expected }) => {
    document.body.innerHTML =
      '<main><div id="chamber" data-widget="hemicycle" ' +
      `data-props='{"configFrom":"result","animate":${animate}}'></div></main>`;
    const q = {
      player_party: 'erc',
      result: { seats: [{ party: 'erc', seats: 12, colour: 'erc' }], majority: 7 },
    };

    (window as any).mountWidgets(document.querySelector('main'), q);

    expect(document.querySelector('#chamber > svg')).not.toBeNull();
    expect(renderedData).toEqual([{ id: 'erc', seats: 12 }]);
    expect(fromCenter).toHaveBeenCalledWith(expected);
    expect(smallToBig).toHaveBeenCalledWith(expected);
  });
});
