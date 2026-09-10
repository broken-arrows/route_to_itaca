import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  aggregateNoActionLocalRuns,
  formatNoActionLocalAggregate,
  runNoActionLocalSimulation,
  type Difficulty,
} from './support/no-action-monte-carlo';

const GAME = path.join(__dirname, '..', '..', 'out', 'game.json');
const RUNS = Number(process.env.LOCAL_MC_RUNS ?? 0);
const SEED = Number(process.env.LOCAL_MC_SEED ?? 12092012);
const DIFFICULTY = (process.env.LOCAL_MC_DIFFICULTY ?? 'normal') as Difficulty;
const CUP_START = process.env.LOCAL_MC_CUP_START === undefined
  ? undefined
  : Number(process.env.LOCAL_MC_CUP_START);
const CUP_SENSITIVITY = process.env.LOCAL_MC_CUP_SENSITIVITY?.split(',').map(Number);
const ENABLED = existsSync(GAME) && Number.isInteger(RUNS) && RUNS > 0;

describe('opt-in no-action 2015 local-election Monte Carlo', () => {
  it.skipIf(!ENABLED)('records Barcelona and red-belt results across every timeline', async () => {
    expect(['easy', 'normal', 'hard']).toContain(DIFFICULTY);
    const root = path.join(__dirname, '../..');
    const files = [
      'out/game.json',
      'source/lib/cat_engine.js',
      'source/scenes/root.scene.dry',
      'source/scenes/election_algorithm.scene.dry',
      'ui/tests/support/no-action-monte-carlo.ts',
    ];
    const identity = Object.fromEntries(files.map((file) => [file,
      createHash('sha256').update(readFileSync(path.join(root, file))).digest('hex'),
    ]));
    const results: ReturnType<typeof runNoActionLocalSimulation>[] = [];
    for (let index = 0; index < RUNS; index++) {
      const seed = SEED + index;
      try {
        results.push(runNoActionLocalSimulation({
          gamePath: GAME,
          seed,
          difficulty: DIFFICULTY,
          localBarcelonaSupportOverrides: CUP_START === undefined ? undefined : { cup: CUP_START },
          localBarcelonaSensitivityOverrides: CUP_SENSITIVITY === undefined
            ? undefined
            : { cup: CUP_SENSITIVITY },
        }));
      } catch (error) {
        throw new Error(`Local-election seed ${seed} failed`, { cause: error });
      }
      if (index % 25 === 24) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const aggregate = aggregateNoActionLocalRuns(results);
    expect(results.every((run) => run.result.year === 2015 && run.result.month === 5)).toBe(true);
    expect(results.every((run) => run.result.totalBarcelonaSeats === 41)).toBe(true);
    expect(results.every((run) => run.result.currentCiu === 'ciu'
      || run.result.barcelonaValidVoteShare.ciu === 0)).toBe(true);
    const cupEntryRate = 1 - aggregate.cupNoSeatRuns / aggregate.runs;
    expect(cupEntryRate).toBeGreaterThanOrEqual(0.60);
    expect(cupEntryRate).toBeLessThanOrEqual(0.70);
    if (process.env.LOCAL_MC_OUTPUT) {
      writeFileSync(path.resolve(process.env.LOCAL_MC_OUTPUT), JSON.stringify({ identity, aggregate, results }, null, 2));
    }
    console.log(formatNoActionLocalAggregate(aggregate));
  }, Math.max(120_000, RUNS * 500));
});
