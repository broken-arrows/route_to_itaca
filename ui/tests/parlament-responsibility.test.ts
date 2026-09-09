import { describe, expect, it } from 'vitest';
import { gameLib } from '../src/game-bindings';

const context = { province: 'barcelona', demographic: 'middle', dWelfare: -2, dUnemployment: 0, dDissent: 0 };
function state(cabinet = ['erc']) {
  const parties = ['ciu', 'erc', 'cup', 'psc', 'icv', 'cs', 'ppc'];
  return {
    parties, cat_coalition: cabinet, cat_coalition_support: [], cat_coalition_abstain: [],
    ...Object.fromEntries([...parties, 'abstain'].map(p => [`${p}_parlament_barcelona_middle_support`, 12.5])),
  } as Record<string, any>;
}
describe('Generalitat electoral responsibility', () => {
  it('uses formal roles, shares collective weights and excludes duplicate weaker roles', () => {
    const q = { cat_coalition: ['erc', 'ciu', 'erc'], cat_coalition_support: ['erc', 'ppc'], cat_coalition_abstain: ['ppc', 'psc'] };
    expect(gameLib.getParlamentResponsibility(q)).toEqual({ il: .5, icr: .5, ppc: .38, psc: .16 });
    expect(gameLib.getParlamentResponsibility({ cat_coalition: ['jxsi'], erc_in_jxsi: true })).toEqual({ icr: .5, il: .5 });
  });
  it('charges the governing family and routes ERC and PPC losses to different alternatives', () => {
    const erc = gameLib.buildParlamentResponsibilityTransfers(state(), context);
    const ppc = gameLib.buildParlamentResponsibilityTransfers(state(['ppc']), context);
    expect(erc.every(t => t.from === 'il')).toBe(true);
    expect(erc.find(t => t.to === 'cup')!.amount).toBeGreaterThan(erc.find(t => t.to === 'psc')!.amount);
    expect(ppc.every(t => t.from === 'ppc')).toBe(true);
    expect(ppc.find(t => t.to === 'cs')!.amount).toBeGreaterThan(ppc.find(t => t.to === 'psc')!.amount);
    expect(erc.reduce((n, t) => n + t.amount, 0)).toBeCloseTo(.09);
  });
  it('gives credit for improvement and never rewards a cabinet partner as an opposition alternative', () => {
    const q = state(['erc', 'ciu']);
    const losses = gameLib.buildParlamentResponsibilityTransfers(q, context);
    expect(losses.every(t => !['il', 'icr'].includes(t.to))).toBe(true);
    expect(losses.reduce((n, t) => n + t.amount, 0)).toBeCloseTo(.09);
    const gains = gameLib.buildParlamentResponsibilityTransfers(q, { ...context, dWelfare: 2 });
    expect(gains).toEqual(losses.map(t => ({ ...t, from: t.to, to: t.from })));
    expect(gameLib.buildParlamentResponsibilityTransfers(q, { ...context, dWelfare: 0 })).toEqual([]);
  });
  it('limits support/abstention blame and preserves the electorate after shared settlement', () => {
    const q = state(['ciu']); q.cat_coalition_support = ['ppc']; q.cat_coalition_abstain = ['psc'];
    const requests = gameLib.buildParlamentResponsibilityTransfers(q, context);
    const out = (f: string) => requests.filter(t => t.from === f).reduce((n, t) => n + t.amount, 0);
    expect(out('ppc') / out('icr')).toBeCloseTo(.38);
    expect(out('psc') / out('icr')).toBeCloseTo(.16);
    gameLib.applyParlamentTransfers(q, 'barcelona', 'middle', {}, requests);
    expect([...q.parties, 'abstain'].reduce((n, p) => n + q[`${p}_parlament_barcelona_middle_support`], 0)).toBeCloseTo(100);
  });
});
