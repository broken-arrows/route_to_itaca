import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { runNoActionSimulation } from './support/no-action-monte-carlo';

const GAME = path.join(__dirname, '..', '..', 'out', 'game.json');

describe('no-action Monte Carlo runner', () => {
  it.skipIf(!existsSync(GAME))('advances only through pass-time and records one post-2012 election', () => {
    const result = runNoActionSimulation({
      gamePath: GAME,
      seed: 12092012,
      difficulty: 'normal',
    });
    expect(result.baseline.year).toBe(2012);
    expect(result.final.year).toBeGreaterThan(2012);
    expect(result.monthsAdvanced).toBeGreaterThanOrEqual(12);
    expect(result.targetElectionKeys).toEqual([`${result.final.year}-${result.final.month}`]);
    expect(result.historicalStructuralEvents).toContain('plebiscite_election');
    expect(result.historicalStructuralEvents).toContain('jxsi_formation_offered');
    expect(Number.isFinite(result.seatDelta)).toBe(true);
    expect(Number.isFinite(result.final.validVoteShare.cs)).toBe(true);
    expect(result.baseline.families.abstention).toBeGreaterThan(0);
    expect(result.final.families).toHaveProperty('sovereignty_joint_lists');
    expect(result.final.families).toHaveProperty('ciu_successors_outside_joint_lists');
    expect(result.final.families).not.toHaveProperty('icr');
    expect(result.final.families).not.toHaveProperty('erc');
    expect(result.final.support.ppc).toBeGreaterThanOrEqual(0);
  }, 30_000);

  it.skipIf(!existsSync(GAME))('can execute the ERC JxSi formation path headlessly', () => {
    const results = [12092012, 12092013, 12092014, 12092015].map((seed) =>
      runNoActionSimulation({ gamePath: GAME, seed, difficulty: 'normal' }));
    const formed = results.find((result) => result.structuralState.jxsiFormed);
    expect(formed).toBeDefined();
    expect(formed!.final.support.jxsi).toBeGreaterThan(0);
    expect(formed!.final.validVoteShare.jxsi).toBeGreaterThan(0);
  }, 30_000);

  it.skipIf(!existsSync(GAME))('does not retain stale DL seats when ERC refuses JxSi', () => {
    const result = runNoActionSimulation({
      gamePath: GAME,
      seed: 12092025,
      difficulty: 'normal',
    });
    expect(result.structuralState.dlFormed).toBe(true);
    expect(result.structuralState.jxsiFormed).toBe(false);
    expect(result.final.seats.dl).toBeGreaterThan(0);
    expect(result.final.totalSeats).toBe(135);
  }, 30_000);
});
