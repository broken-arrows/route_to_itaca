import { expect, it } from 'vitest';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { HISTORICAL_BALANCE_ASSUMPTIONS, runHistoricalBalance, type Strategy } from './support/historical-balance';

const root = resolve(__dirname, '../..');
const game = readFileSync(resolve(root, 'out/game.json'), 'utf8');
const runs = Number(process.env.BALANCE_RUNS ?? (process.env.npm_lifecycle_event === 'balance:historical' ? 20 : 0));
const firstSeed = Number(process.env.BALANCE_SEED ?? 12092012);
type Result = ReturnType<typeof runHistoricalBalance>;

function validate(result: Result) {
  expect(result.pathFailures, `${result.strategy}/${result.seed}: reference path diverged`).toEqual([]);
  expect(result.elections.map(e => e.date)).toEqual(['2012-11', '2015-09']);
  for (const election of result.elections) {
    const seats = Object.values(election.seats) as number[];
    const votes = Object.values(election.votes) as number[];
    expect(seats.every(n => Number.isInteger(n) && n >= 0)).toBe(true);
    expect(seats.reduce((a,b) => a+b, 0)).toBe(135);
    expect(votes.every(n => Number.isFinite(n) && n >= 0)).toBe(true);
    expect(Math.abs(votes.reduce((a,b) => a+b, 0) - 100)).toBeLessThan(0.15);
  }
  expect(result.consultation?.date).toBe('2014-11');
  const c = result.consultation!;
  expect(c.consultation_question).toBe(2);
  expect(Number(c.yesVotes) + Number(c.yesNoVotes) + Number(c.noVotes)).toBe(c.consultation_total_votes);
  expect(Number(c.consultation_yes_pc) + Number(c.consultation_yesno_pc) + Number(c.consultation_no_pc)).toBeCloseTo(100, 0);
  for (const row of result.monthly) for (const [key,value] of Object.entries(row))
    if (key !== 'date') expect(Number.isFinite(value), `${row.date}: ${key}`).toBe(true);
  expect(result.finalEconomy.public_salary_level).toBe(0);
  expect(result.finalEconomy.education_spending).toBe(result.strategy === 'services' ? 0 : -1);
  expect(result.finalEconomy.healthcare_spending).toBe(result.strategy === 'services' ? 0 : -1);
}

function stats(values: number[]) {
  const mean = values.reduce((a,b) => a+b, 0) / values.length;
  return {mean, min: Math.min(...values), max: Math.max(...values),
    sd: Math.sqrt(values.reduce((sum,n) => sum + (n-mean)**2, 0) / values.length)};
}
function aggregate(results: Result[]) {
  return Object.fromEntries((['historical', 'services', 'campaign'] as Strategy[]).map(strategy => {
    const rows = results.filter(r => r.strategy === strategy);
    const measure = (fn: (r: Result) => number) => stats(rows.map(fn));
    return [strategy, {runs: rows.length,
      erc2012Seats: measure(r => r.elections[0].seats.erc), erc2012Vote: measure(r => r.elections[0].votes.erc),
      jxsi2015Seats: measure(r => r.elections[1].seats.jxsi), jxsi2015Vote: measure(r => r.elections[1].votes.jxsi),
      independence2015Seats: measure(r => r.elections[1].seats.jxsi + r.elections[1].seats.cup),
      independence2015Vote: measure(r => r.elections[1].votes.jxsi + r.elections[1].votes.cup),
      consultationVotes: measure(r => Number(r.consultation!.consultation_total_votes)),
      consultationYesVotes: measure(r => Number(r.consultation!.yesVotes)),
      consultationYesShare: measure(r => Number(r.consultation!.consultation_yes_pc)),
      surplus: measure(r => Number(r.finalEconomy.generalitat_surplus)),
      unemployment: measure(r => Number(r.finalEconomy.unemployment)), welfare: measure(r => Number(r.finalEconomy.welfare_index)),
      debt: measure(r => Number(r.finalEconomy.public_debt)), campaigns: measure(r => r.campaignCount),
      positiveBalanceRuns: rows.filter(r => Number(r.finalEconomy.generalitat_surplus) > 0).length,
      popularMajorityRuns: rows.filter(r => r.elections[1].votes.jxsi + r.elections[1].votes.cup > 50).length,
      jxsiMajorityRuns: rows.filter(r => r.elections[1].seats.jxsi >= 68).length,
    }];
  }));
}

it('replays the historical path deterministically and completes real votes and policies', () => {
  const first = runHistoricalBalance(game, firstSeed, 'historical');
  validate(first);
  expect(runHistoricalBalance(game, firstSeed, 'historical')).toEqual(first);
}, 20_000);

it.skipIf(!runs)('measures matched historical, services and campaign scenarios without prescribing results', async () => {
  expect(Number.isInteger(runs) && runs > 0).toBe(true);
  expect(Number.isInteger(firstSeed) && firstSeed >= 0).toBe(true);
  const output = resolve(process.env.BALANCE_OUTPUT || resolve(root, '.scratch/historical-balance-latest.json'));
  mkdirSync(dirname(output), {recursive: true});
  const identity = Object.fromEntries(['out/game.json', 'source/lib/cat_engine.js', 'source/lib/government.js',
    'ui/tests/support/historical-balance.ts', 'ui/tests/historical-balance.test.ts'].map(file =>
    [file, createHash('sha256').update(readFileSync(resolve(root, file))).digest('hex')]));
  const results: Result[] = [];
  const save = (complete: boolean) => writeFileSync(output, JSON.stringify({complete, identity,
    assumptions: HISTORICAL_BALANCE_ASSUMPTIONS, results, ...(complete ? {aggregate: aggregate(results)} : {})}, null, 2));
  for (let i = 0; i < runs; i++) {
    for (const strategy of ['historical', 'services', 'campaign'] as const) {
      const result = runHistoricalBalance(game, firstSeed + i, strategy);
      results.push(result);
      save(false);
      validate(result);
    }
    if (i % 5 === 4) await new Promise(resolve => setTimeout(resolve, 0));
  }
  save(true);
  for (const [strategy, a] of Object.entries(aggregate(results))) console.log(
    `${strategy} n=${a.runs}: ERC2012=${a.erc2012Seats.mean.toFixed(1)} seats/${a.erc2012Vote.mean.toFixed(2)}%; ` +
    `JxSi2015=${a.jxsi2015Seats.mean.toFixed(1)} seats; JxSi+CUP=${a.independence2015Vote.mean.toFixed(2)}%; ` +
    `9N=${Math.round(a.consultationVotes.mean)} ballots/${a.consultationYesShare.mean.toFixed(2)}% yes-yes; ` +
    `balance=${a.surplus.mean.toFixed(3)}, positive=${a.positiveBalanceRuns}/${a.runs}`);
  console.log(`Full trajectories, choices, assumptions, hashes and distributions: ${output}`);
}, Math.max(120_000, runs * 3000));
