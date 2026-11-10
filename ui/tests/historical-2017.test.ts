import { it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { runHistoricalBalance } from './support/historical-balance';

const game = readFileSync(resolve(__dirname, '../../out/game.json'), 'utf8');

it('extends the historical path through August 2017 with real government and budget outcomes', () => {
  // Fixed successful fixture under the calibrated macro model. Calibration
  // batches retain every seed, including genuine negotiation failures.
  const short = runHistoricalBalance(game, 12092014, 'historical');
  const result = runHistoricalBalance(game, 12092014, 'historical', {}, undefined, '2017-08');
  expect(result.pathFailures).toEqual([]);
  expect(result.elections).toEqual(short.elections);
  expect(result.consultation).toEqual(short.consultation);
  expect(result.monthly.filter(r => String(r.date) < '2015-09'))
    .toEqual(short.monthly.filter(r => String(r.date) < '2015-09'));
  expect(result.checkpoints!.map(r => r.date)).toEqual(Array.from({length:61}, (_, i) => {
    const month = 7 + i;
    return `${2012 + Math.floor(month / 12)}-${String(month % 12 + 1).padStart(2, '0')}`;
  }));
  const at = (date: string) => result.checkpoints!.find(r => r.date === date)!;
  expect(at('2016-01').president).toBe('Carles Puigdemont');
  expect(at('2016-06').budget_year).toBe(2015);
  expect(at('2016-06').railway).toBe(1);
  expect(at('2016-07').railway).toBe(-1);
  expect(at('2017-03').budget_year).toBe(2017);
  const budgets = result.interventions.filter(r => r.kind === 'budget-result');
  expect(budgets.find(r => r.date === '2016-06')?.positions).toMatchObject({jxsi:'yes',cup:'no'});
  expect(budgets.find(r => r.date === '2017-03')?.positions).toMatchObject({jxsi:'yes',cup:'yes'});
  expect(result.finalGovernment).toEqual({president:'Carles Puigdemont',coalition:['jxsi'],support:['cup'],
    spanish:['pp'],caretaker:false,rubicon:0});
  expect(result.finalEconomy.date).toBe('2017-08');
  for (const row of result.checkpoints!) for (const key of ['gdp_growth','unemployment','generalitat_surplus','public_debt'])
    expect(Number.isFinite(row[key]), `${row.date}: ${key}`).toBe(true);
  expect(runHistoricalBalance(game, 12092014, 'historical', {}, undefined, '2017-08')).toEqual(result);
}, 30_000);

it('reports a stochastic negotiation collapse instead of forcing the historical government', () => {
  // Fixed failure from the Mediterranean batch; changed active ballots consume
  // a different number of shared random draws. Keep all seeds in batch reports.
  const result = runHistoricalBalance(game, 12092015, 'historical', {}, undefined, '2017-08');
  expect(result.finalEconomy.date).toBe('2017-08');
  expect(result.pathFailures).toContain('Puigdemont not invested in January 2016');
  expect(result.elections.length).toBeGreaterThan(2);
  expect(result.log.some(r => r.choice === 'parlament_coalition_sovereignty_dispute.collapse')).toBe(true);
}, 30_000);
