import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
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
  it.skipIf(!ENABLED)('measures the 2012 to first post-2012 Parlament election interval', () => {
    expect(['easy', 'normal', 'hard']).toContain(DIFFICULTY);
    const results = Array.from({ length: RUNS }, (_, index) => {
      const seed = SEED + index;
      try {
        return runNoActionSimulation({ gamePath: GAME, seed, difficulty: DIFFICULTY });
      } catch (error) {
        throw new Error(`No-action seed ${seed} failed`, { cause: error });
      }
    });
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
    console.log(formatNoActionAggregate(aggregateNoActionRuns(results)));
  }, 120_000);
});
