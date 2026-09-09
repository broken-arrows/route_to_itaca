import { describe, expect, it } from 'vitest';
import { createRequire } from 'node:module';
const { buildParlamentCompetitionTransfers: build, getFederalLeftLeadershipProfile: profile } = createRequire(import.meta.url)('../../source/lib/cat_engine.js');
const context = { activeFederalLeft: 'csqp', dCatSpa: -1, scaleCatSpa: 1, support: { ppc: 10, cs: 6, psc: 10, fl: 10 } };
const q = { psc_leader: 'Miquel Iceta', psc_recovery_mult: 1, csqp_leader: 'Lluís Rabell', cat_spa_relations: 25, independence_movement: 75, spa_cs_active: true, time: 31, next_election_time: 38 };
const amount = (state: object, mechanism: string, ctx = context) => build(state, ctx).find((t: any) => t.mechanism === mechanism)?.amount ?? 0;
describe('Parlament party competition', () => {
  it('reads the active federal-left leader with a neutral unknown fallback', () => {
    expect(profile({ icv_leader: 'Joan Herrera', csqp_leader: 'Arcadi Oliveres' }, 'icv').conflict).toBe(1);
    expect(profile({ csqp_leader: 'Arcadi Oliveres' }, 'csqp').conflict).toBeGreaterThan(1);
    expect(profile({ csqp_leader: 'Unknown' }, 'csqp')).toEqual({ channeling: 1, conflict: 1, retention: 1 });
  });
  it('lets both leaderships alter competition and treats recovery as resistance to losses', () => {
    const route = 'nonlinear.psc_federal_conflict';
    expect(amount({ ...q, psc_leader: 'Montserrat Tura', psc_recovery_mult: 1.15 }, route)).toBeLessThan(amount(q, route));
    expect(amount({ ...q, csqp_leader: 'Arcadi Oliveres' }, route)).toBeGreaterThan(amount(q, route));
    const reversed = build(q, { ...context, dCatSpa: 1 }).find((t: any) => t.mechanism === route);
    expect(reversed.from).toBe('fl'); expect(reversed.to).toBe('psc');
    expect(reversed.amount).toBeCloseTo(amount(q, route));
  });
  it('funds Cs exclusively from PPC, works under sustained conflict and strengthens after launch', () => {
    const route = 'nonlinear.ppc_cs_competition';
    const steady = { ...context, dCatSpa: 0 };
    const transfer = build(q, steady).find((t: any) => t.mechanism === route);
    expect(transfer.from).toBe('ppc'); expect(transfer.to).toBe('cs'); expect(transfer.amount).toBeGreaterThan(0);
    expect(amount(q, route, steady)).toBeGreaterThan(amount({ ...q, spa_cs_active: false }, route, steady));
    expect(amount(q, route, { ...steady, support: { ...steady.support, ppc: 2, cs: 14 } })).toBe(0);
  });
  it('saturates under repeated unchanged conditions without exhausting PPC', () => {
    const ctx = structuredClone(context); ctx.dCatSpa = 0;
    for (let month = 0; month < 500; month++) {
      const a = amount(q, 'nonlinear.ppc_cs_competition', ctx);
      ctx.support.ppc -= a; ctx.support.cs += a;
    }
    expect(ctx.support.ppc).toBeGreaterThan(5);
    expect(ctx.support.ppc + ctx.support.cs).toBeCloseTo(16);
    expect(amount(q, 'nonlinear.ppc_cs_competition', ctx)).toBeLessThan(1e-8);
  });
  it('lets persistent corruption strengthen competition even without rising territorial tension', () => {
    const route = 'nonlinear.ppc_cs_competition';
    const state = { ...q, cat_spa_relations: 80, independence_movement: 20, spa_cs_active: false };
    const ctx = { ...context, dCatSpa: 0 };
    expect(amount({ ...state, corruption_pp: 0 }, route, ctx)).toBe(0);
    expect(amount({ ...state, corruption_pp: 100 }, route, ctx)).toBeGreaterThan(0);
    expect(amount({ ...q, corruption_pp: 50 }, route)).toBeGreaterThan(amount({ ...q, corruption_pp: 35 }, route));
    expect(amount({ ...q, corruption_pp: -10 }, route)).toBe(amount(q, route));
    expect(amount({ ...q, corruption_pp: NaN }, route)).toBe(amount(q, route));
    expect(amount({ ...q, corruption_pp: 200 }, route)).toBe(amount({ ...q, corruption_pp: 100 }, route));
  });
  it('preserves viability and a finite PPC pool even under maximum corruption and conflict', () => {
    const route = 'nonlinear.ppc_cs_competition';
    const state = { ...q, corruption_pp: 100, cat_spa_relations: 0, independence_movement: 100, time: 38 };
    const ctx = { ...structuredClone(context), scaleCatSpa: 4 };
    expect(amount(state, route, { ...ctx, support: { ...ctx.support, cs: 0 } })).toBe(0);
    expect(amount({ ...state, cs_parlament_s: 1 }, route, { ...ctx, support: { ...ctx.support, cs: 0 } })).toBeGreaterThan(0);
    for (let month = 0; month < 500; month++) {
      const transferred = amount(state, route, ctx);
      expect(transferred).toBeLessThanOrEqual(ctx.support.ppc);
      ctx.support.ppc -= transferred; ctx.support.cs += transferred;
    }
    expect(ctx.support.ppc).toBeGreaterThan(3);
    expect(ctx.support.ppc + ctx.support.cs).toBeCloseTo(16);
    expect(amount(state, route, ctx)).toBeLessThan(1e-8);
  });
});
