import { describe, expect, it } from 'vitest';
import { gameLib } from '../src/game-bindings';

const supportKey = (p: string) => `${p}_parlament_girona_rural_support`;
function state(support: Record<string, number>, flags = {}) {
  return {
    parties: Object.keys(support).filter((p) => p !== 'abstain'),
    ...Object.fromEntries(Object.entries(support).map(([p, value]) => [supportKey(p), value])),
    ...flags,
  } as Record<string, any>;
}
const settle = (q: Record<string, any>, matrix: Record<string, number>, transfers: Array<{ mechanism: string; from: string; to: string; amount: number }>) =>
  gameLib.applyParlamentTransfers(q, 'girona', 'rural', matrix, transfers, 1);
const total = (q: Record<string, any>) => [...q.parties, 'abstain'].reduce((n, p) => n + (q[supportKey(p)] || 0), 0);

describe('monthly Parlament transfer settlement', () => {
  it('shares opening donor support proportionally and cannot spend incoming voters', () => {
    const requests = [
      { mechanism: 'first', from: 'ppc', to: 'cs', amount: 8 },
      { mechanism: 'second', from: 'ppc', to: 'psc', amount: 4 },
      { mechanism: 'incoming', from: 'abs', to: 'ppc', amount: 10 },
    ];
    const q = state({ ppc: 6, cs: 10, psc: 10, abstain: 74 });
    const reverse = structuredClone(q);
    settle(q, {}, requests);
    settle(reverse, {}, [...requests].reverse());
    expect(q[supportKey('ppc')]).toBeCloseTo(10);
    expect(q[supportKey('cs')]).toBeCloseTo(14);
    expect(q[supportKey('psc')]).toBeCloseTo(12);
    expect(total(q)).toBeCloseTo(100);
    for (const p of [...q.parties, 'abstain']) expect(reverse[supportKey(p)]).toBeCloseTo(q[supportKey(p)]);
  });

  it('shares the donor budget with the net matrix and preserves unrelated support', () => {
    const q = state({ ppc: 6, cs: 10, psc: 10, cup: 10, abstain: 64 });
    settle(q, { ppc: -8, psc: 16 }, [{ mechanism: 'switching', from: 'ppc', to: 'cs', amount: 4 }]);
    expect(q[supportKey('ppc')]).toBeCloseTo(0);
    expect(q[supportKey('psc')]).toBeCloseTo(14);
    expect(q[supportKey('cs')]).toBeCloseTo(12);
    expect(q[supportKey('cup')]).toBe(10);
    expect(q[supportKey('abstain')]).toBe(64);
    expect(total(q)).toBeCloseTo(100);
  });

  it('nets coalition and pre-split matrix rows before capping and cancels internal transfers', () => {
    const q = state({ ciu: 0, erc: 0, jxsi: 50, cup: 10, abstain: 40 }, { erc_in_jxsi: true });
    settle(q, { icr: -100, il: 95, unio: 2, pdcat: 3 }, [
      { mechanism: 'internal', from: 'icr', to: 'il', amount: 100 },
      { mechanism: 'external', from: 'icr', to: 'cup', amount: 10 },
    ]);
    expect(q[supportKey('jxsi')]).toBeCloseTo(40);
    expect(q[supportKey('cup')]).toBeCloseTo(20);
    expect(q[supportKey('ciu')]).toBe(0);
    expect(q[supportKey('erc')]).toBe(0);
    expect(total(q)).toBeCloseTo(100);
  });

  it('does not revive dormant parties or charge donors for unavailable recipients', () => {
    const q = state({ ciu: 30, erc: 10, vox: 0, abstain: 60 });
    settle(q, { icr: -5, vox: 5 }, [{ mechanism: 'inactive', from: 'il', to: 'vox', amount: 5 }]);
    expect(q[supportKey('vox')]).toBe(0);
    expect(q[supportKey('ciu')]).toBe(30);
    expect(q[supportKey('erc')]).toBe(10);
    expect(total(q)).toBe(100);
  });

  it('reports requested and realized named transfers without changing results', () => {
    const q = state({ ppc: 2, cs: 8, abstain: 90 }, { parlament_vote_trace_enabled: true });
    settle(q, {}, [{ mechanism: 'switching', from: 'ppc', to: 'cs', amount: 10 }]);
    expect(q.parlament_vote_trace.mechanisms['requested.switching']).toEqual({ ppc: -10, cs: 10 });
    expect(q.parlament_vote_trace.mechanisms.switching).toEqual({ ppc: -2, cs: 2 });
    expect(total(q)).toBeCloseTo(100);
  });

  it('allows a nationally live party and a newly formed FNC to gain in empty cells', () => {
    const q = state({ ppc: 30, cs: 0, fnc: 0, abstain: 70 }, {
      parlament_constituencies: ['girona', 'barcelona'], parlament_demographics: ['rural'],
      cs_parlament_barcelona_rural_support: 10, fnc_formed: true, pxc_dissolved: true,
    });
    settle(q, {}, [
      { mechanism: 'competition', from: 'ppc', to: 'cs', amount: 3 },
      { mechanism: 'formation', from: 'ppc', to: 'fnc', amount: 1 },
    ]);
    expect(q[supportKey('cs')]).toBe(3);
    expect(q[supportKey('fnc')]).toBe(1);
    expect(total(q)).toBeCloseTo(100);
  });

  it('routes to the canonical successor rather than stale positive predecessor support', () => {
    const q = state({ ciu: 10, cdc: 30, cup: 10, abstain: 50 }, { parlament_current_ciu: 'cdc' });
    settle(q, { icr: -2, cup: 2 }, []);
    expect(q[supportKey('ciu')]).toBe(10);
    expect(q[supportKey('cdc')]).toBe(28);
    expect(total(q)).toBeCloseTo(100);
  });
});
