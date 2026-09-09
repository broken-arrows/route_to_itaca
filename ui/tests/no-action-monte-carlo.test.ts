import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import {
  aggregateNoActionRuns,
  formatNoActionAggregate,
  runNoActionSimulation,
  type Difficulty,
} from './support/no-action-monte-carlo';

const GAME = path.join(__dirname, '..', '..', 'out', 'game.json');
const RUNS = Number(process.env.MC_RUNS ?? 0);
const SEED = Number(process.env.MC_SEED ?? 12092012);
const DIFFICULTY = (process.env.MC_DIFFICULTY ?? 'normal') as Difficulty;
const DETAILS = process.env.MC_DETAILS === '1';
const ENABLED = existsSync(GAME) && Number.isInteger(RUNS) && RUNS > 0;

describe('opt-in no-action Dendry Monte Carlo', () => {
  it.skipIf(!ENABLED)('measures the 2012 to first post-2012 Parlament election interval', async () => {
    expect(['easy', 'normal', 'hard']).toContain(DIFFICULTY);
    const root = path.join(__dirname, '../..');
    const files = ['out/game.json', 'source/lib/cat_engine.js', 'ui/tests/support/no-action-monte-carlo.ts'];
    const identity = Object.fromEntries(files.filter((file) => existsSync(path.join(root, file))).map((file) =>
      [file, createHash('sha256').update(readFileSync(path.join(root, file))).digest('hex')],
    ));
    const results: ReturnType<typeof runNoActionSimulation>[] = [];
    for (let index = 0; index < RUNS; index++) {
      const seed = SEED + index;
      try {
        results.push(runNoActionSimulation({ gamePath: GAME, seed, difficulty: DIFFICULTY }));
      } catch (error) {
        throw new Error(`No-action seed ${seed} failed`, { cause: error });
      }
      // Keep Vitest's worker reporting alive during large synchronous batches.
      if (index % 25 === 24) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (DETAILS) {
      for (const result of results.filter((run) => run.final.totalSeats !== 135)) {
        const nonZeroSeats = Object.entries(result.final.seats)
          .filter(([, seats]) => seats !== 0)
          .map(([party, seats]) => `${party}=${seats}`)
          .join(' ');
        console.log(
          `INVALID seed=${result.seed} date=${result.final.year}-${result.final.month} `
          + `total=${result.final.totalSeats} flags=${JSON.stringify(result.structuralState)} `
          + `events=${result.historicalStructuralEvents.join(',')} seats=${nonZeroSeats}`,
        );
      }
    }
    const aggregate = aggregateNoActionRuns(results);
    expect(aggregate.invalidSeatTotalRuns).toBe(0);
    expect(aggregate.invalidValidVoteTotalRuns).toBe(0);
    if (process.env.MC_OUTPUT) {
      writeFileSync(path.resolve(process.env.MC_OUTPUT), JSON.stringify({ identity, aggregate, results }, null, 2));
    }
    for (const run of results) {
      for (const [cell, total] of Object.entries(run.final.cellSupportTotals)) {
        expect(total, `Electorate cell ${cell} for seed ${run.seed}`).toBeCloseTo(run.baseline.cellSupportTotals[cell], 6);
      }
      expect(Object.values(run.final.support).every((value) => Number.isFinite(value) && value >= 0)).toBe(true);
    }
    console.log(formatNoActionAggregate(aggregate));
  }, Math.max(120_000, RUNS * 500));
});
