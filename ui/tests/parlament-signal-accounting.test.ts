import { beforeAll, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { gameLib } from '../src/game-bindings';
import { DendryAdapter } from '../src/engine/adapter';

let fixture: Record<string, any>;
beforeAll(() => {
  const adapter = DendryAdapter.fromJSONText(
    readFileSync(path.resolve(__dirname, '../../out/game.json'), 'utf8'),
  );
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start');
  adapter.goToScene('root.normal_mode');
  adapter.goToScene('root.esquerra');
  let frame = adapter.goToScene('root.esquerra_2');
  for (let guard = 0; frame.effectiveRole !== 'desk' && guard < 20; guard++) {
    const choice = frame.choices.findIndex(choice => choice.canChoose);
    if (choice < 0) throw new Error(`Introduction dead-ended at ${frame.sceneId}`);
    frame = adapter.choose(choice);
  }
  expect(frame.effectiveRole).toBe('desk');
  adapter.goToScene('election_simulation.2012_11');
  fixture = structuredClone(adapter.qualities);
  Object.assign(fixture, {
    year: 2012, month: 11, week: 1, time: 4,
    next_election_year: 2015, next_election_month: 9,
    next_election_week: 1, next_election_time: 38,
  });
});
function tick(q: Record<string, any>) {
  q.parlament_vote_trace_enabled = true;
  q.parlament_vote_trace = { ticks: 0, mechanisms: {} };
  const rng = vi.spyOn(Math, 'random').mockReturnValue(0.6);
  try { gameLib.engineTick(q); } finally { rng.mockRestore(); }
}
function ready() {
  const q = JSON.parse(JSON.stringify(fixture));
  // A single real cell keeps its matrix coefficient independent of changing
  // national population weights. All macro inputs still come from the game.
  q.parlament_constituencies = ['girona'];
  q.parlament_demographics = ['rural'];
  tick(q);
  return q;
}
function coefficient(q: Record<string, any>, signal: string) {
  const c = JSON.parse(JSON.stringify(q));
  const previous = c[signal];
  tick(c);
  return c.parlament_vote_trace.mechanisms[`matrix.${signal}`].abs / (c[signal] - previous);
}
describe('Parlament political signal observation', () => {
  it.each([40, 10])('reinforces active-movement CUP disappointment while preserving recovery (previous trust %s)', previousTrust => {
    const q = ready();
    q.independence_movement = 95;
    q.independence_trust = 20;
    q.parlament_signal_baseline.independence_trust = previousTrust;
    tick(q);
    const change = q.independence_trust - previousTrust;
    const depth = (30 - q.independence_trust) / 30;
    const oldResponse = -.025 * change * depth * .6 * 1.2;
    const route = q.parlament_vote_trace.mechanisms['requested.nonlinear.cup_trust'];
    expect(Math.abs(route.cup)).toBeGreaterThan(Math.abs(oldResponse) * 2);
    expect(Math.sign(route.cup)).toBe(-Math.sign(change));
    expect(route.cup + route.icr + route.il).toBeCloseTo(0, 10);
  });

  it('separates accumulated FNC disappointment from socioeconomic feeding without bypassing formation', () => {
    const q = ready();
    q.independence_movement = 95;
    q.independence_trust = 20;
    q.fnc_formed = true;
    q.pxc_dissolved = true;
    const dormant = structuredClone(q);
    dormant.fnc_formed = false;
    tick(q);
    tick(dormant);
    const dissent = Math.min(1, Math.max(0, (q.social_dissent - 40) / 32));
    const route = q.parlament_vote_trace.mechanisms['requested.nonlinear.fnc_feeding'];
    expect(route.fnc).toBeCloseTo(.018 * .35 * dissent * .8 * 1.1, 10);
    expect(q.parlament_vote_trace.mechanisms['requested.nonlinear.fnc_accumulated_disappointment'].fnc).toBeGreaterThan(0);
    expect(Object.values(route).reduce((sum: number, value) => sum + Number(value), 0)).toBeCloseTo(0, 10);
    expect(dormant.parlament_vote_trace.mechanisms['requested.nonlinear.fnc_feeding']).toBeUndefined();
    expect(dormant.parlament_vote_trace.mechanisms['requested.nonlinear.fnc_accumulated_disappointment']).toBeUndefined();
  });

  it.each(['independence_movement', 'independence_trust'])('counts between-tick %s after serialization, exactly once', signal => {
    let q = ready();
    const factor = coefficient(q, signal);
    const previous = q[signal];
    q[signal] += 5;
    q = JSON.parse(JSON.stringify(q));
    tick(q);
    expect(q.parlament_vote_trace.mechanisms[`matrix.${signal}`].abs).toBeCloseTo(factor * (q[signal] - previous), 9);
    const nextPrevious = q[signal];
    tick(q);
    expect(q.parlament_vote_trace.mechanisms[`matrix.${signal}`].abs).toBeCloseTo(factor * (q[signal] - nextPrevious), 9);
  });

  it('preserves direct vote transfers and still counts the same event\'s momentum change', () => {
    const q = ready();
    const previous = q.independence_movement;
    const factor = coefficient(q, 'independence_movement');
    const ercBefore = q.erc_parlament_girona_rural_support;
    gameLib.applyParlamentTransfers(q, 'girona', 'rural', {}, [
      { mechanism: 'event.rally', from: 'abs', to: 'il', amount: 1 },
    ]);
    expect(q.erc_parlament_girona_rural_support).toBeCloseTo(ercBefore + 1);
    q.independence_movement += 5;
    tick(q);
    expect(q.parlament_vote_trace.mechanisms['matrix.independence_movement'].abs)
      .toBeCloseTo(factor * (q.independence_movement - previous), 9);
  });

  it('starts old saves and replacement electorate fixtures from their current signals', () => {
    const q = ready();
    delete q.parlament_signal_baseline;
    q.independence_movement += 8;
    const previous = q.independence_movement;
    const factor = coefficient(q, 'independence_movement');
    tick(q);
    expect(q.parlament_vote_trace.mechanisms['matrix.independence_movement'].abs)
      .toBeCloseTo(factor * (q.independence_movement - previous), 9);
    q.independence_movement -= 12;
    gameLib.resetParlamentSignalBaseline(q);
    const resetPrevious = q.independence_movement;
    tick(q);
    expect(q.parlament_vote_trace.mechanisms['matrix.independence_movement'].abs)
      .toBeCloseTo(factor * (q.independence_movement - resetPrevious), 9);
  });
});
