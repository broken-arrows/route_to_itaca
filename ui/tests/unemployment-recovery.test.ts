import { it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const lib = gameLib as any;
const game = readFileSync('../out/game.json', 'utf8');
function fixture(unemployment: number, growth = 3) {
  const adapter = DendryAdapter.fromJSONText(game);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  Object.assign(q, { unemployment, gdp_growth: growth, active_mods: {}, game_random_state: 123 });
  return q;
}

it('absorbs unemployment faster at high slack but continues below 10%', () => {
  const rates = [10, 14, 20, 25, 30];
  const states = rates.map(rate => fixture(rate));
  states.forEach(q => lib.engineTick(q));
  states.forEach(q => expect(q.gdp_growth).toBeCloseTo(states[0].gdp_growth, 10));
  const gains = states.map((q, i) => rates[i] - q.unemployment);
  expect(gains[0]).toBeGreaterThan(0);
  expect(gains[1]).toBeGreaterThan(gains[0]);
  expect(gains[2]).toBeGreaterThan(gains[1]);
  expect(gains[3]).toBeGreaterThan(gains[2]);
  expect(gains[4]).toBeCloseTo(gains[3], 10);
});

it('retains the same recession losses at different unemployment levels', () => {
  const low = fixture(5, -3), high = fixture(24, -3);
  lib.engineTick(low); lib.engineTick(high);
  expect(low.gdp_growth).toBeLessThan(0);
  expect(low.unemployment).toBeGreaterThan(5);
  expect(low.unemployment - 5).toBeCloseTo(high.unemployment - 24, 10);
  expect(low.unemployment - 5).toBeCloseTo(-low.gdp_growth / 12 * .3, 10);
});

it.each([10, 14, 24])('preserves direct employment policy effects at %s%% unemployment', rate => {
  const control = fixture(rate), treated = fixture(rate);
  lib.registerLaw(treated, { id: 'employment_test', targets: { unemployment_recovery: .08 } });
  lib.engineTick(control); lib.engineTick(treated);
  expect(control.unemployment - treated.unemployment).toBeCloseTo(.08, 10);
  expect(treated.game_random_state).toBe(control.game_random_state);
});

it.each([[0, .2], [1, .28], [5, .6], [9, .92]])(
  'tapers net policy gains at %s%% unemployment to %s of their strength', (rate, scale) => {
    // A recession lets us observe the policy effect even starting at zero,
    // without the lower bound clipping it.
    const control = fixture(rate, -3), treated = fixture(rate, -3);
    lib.registerLaw(treated, { id: 'employment_test', targets: { unemployment_recovery: .08 } });
    lib.engineTick(control); lib.engineTick(treated);
    expect(control.unemployment - treated.unemployment).toBeCloseTo(.08 * scale, 10);
  },
);

it('preserves policy job losses below 10% unemployment', () => {
  const control = fixture(5), treated = fixture(5);
  lib.registerLaw(treated, { id: 'job_losses', targets: { unemployment_recovery: -.08 } });
  lib.engineTick(control); lib.engineTick(treated);
  expect(treated.unemployment - control.unemployment).toBeCloseTo(.08, 10);
});

it('continues growth-driven recovery below 10% at one fifth of reference strength', () => {
  const low = fixture(5), reference = fixture(20);
  lib.engineTick(low); lib.engineTick(reference);
  expect(5 - low.unemployment).toBeGreaterThan(0);
  expect(5 - low.unemployment).toBeCloseTo((20 - reference.unemployment) * .2, 10);
});

it('allows full employment but never negative unemployment', () => {
  const q = fixture(.01);
  lib.registerLaw(q, { id: 'employment_test', targets: { unemployment_recovery: 1 } });
  lib.engineTick(q);
  expect(q.unemployment).toBe(0);
  lib.engineTick(q);
  expect(q.unemployment).toBe(0);
});
