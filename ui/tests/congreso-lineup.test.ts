import { beforeEach, describe, expect, it } from 'vitest';
import { gameLib } from '../src/game-bindings';

// ---------------------------------------------------------------------------
// The congreso "lineup gates" (spa_*_active / upn_in_pp / iu_in_up) declare WHO
// IS ON THE BALLOT. The engine's only notion of that is `support > 0`
// (cat_engine.js `activeParties`), so a flag that never moves support is inert.
// `reconcileCongresoLineup` is what makes the vote match the flags.
//
// Invariant under test throughout: support is only ever MOVED. A constituency's
// total must be unchanged by any reconcile — nothing appears, nothing vanishes.
// ---------------------------------------------------------------------------

type Q = Record<string, any>;

const PARTIES: Record<string, string[]> = {
  navarra: ['pp', 'psoe', 'podemos', 'cs', 'iu', 'vox', 'up', 'amaiur', 'nsuma', 'ehbildu', 'upn', 'gbai'],
  galicia: ['pp', 'psoe', 'podemos', 'cs', 'iu', 'vox', 'up', 'bng'],
  valencia: ['pp', 'psoe', 'podemos', 'cs', 'iu', 'compromis', 'vox', 'up'],
};

function makeQ(overrides: Q = {}): Q {
  const q: Q = {
    congreso_constituencies: Object.keys(PARTIES),
    corruption_pp: 0,
    corruption_psoe: 0,
    podemos_channeling: 0,
    social_dissent: 0,
    // gates at their root.scene.dry defaults
    upn_in_pp: true,
    iu_in_up: false,
    spa_bng_active: true,
    spa_compromis_active: false,
    spa_foro_active: false,
    spa_nsuma_formed: false,
    spa_ehbildu_active: true,
  };
  for (const [c, ps] of Object.entries(PARTIES)) {
    q['congreso_parties_' + c] = ps;
    for (const p of ps) q[`${p}_congreso_${c}_support`] = 0;
    q[`abstain_congreso_${c}_support`] = 0;
  }
  // Real starting supports from root.scene.dry
  Object.assign(q, {
    pp_congreso_navarra_support: 21.0,
    psoe_congreso_navarra_support: 18.0,
    amaiur_congreso_navarra_support: 13.0,
    gbai_congreso_navarra_support: 9.0,
    iu_congreso_navarra_support: 3.0,
    abstain_congreso_navarra_support: 36.0,

    pp_congreso_galicia_support: 31.0,
    psoe_congreso_galicia_support: 20.0,
    bng_congreso_galicia_support: 5.0,
    iu_congreso_galicia_support: 3.0,
    abstain_congreso_galicia_support: 41.0,

    pp_congreso_valencia_support: 35.0,
    psoe_congreso_valencia_support: 20.0,
    iu_congreso_valencia_support: 5.0,
    abstain_congreso_valencia_support: 40.0,
  });
  return Object.assign(q, overrides);
}

const sup = (q: Q, p: string, c: string) => q[`${p}_congreso_${c}_support`] || 0;
const total = (q: Q, c: string) =>
  PARTIES[c].concat(['abstain']).reduce((a, p) => a + sup(q, p, c), 0);

const reconcile = (q: Q) => (gameLib as any).reconcileCongresoLineup(q);

describe('reconcileCongresoLineup', () => {
  it('is exported for content to call as G.*', () => {
    expect(typeof (gameLib as any).reconcileCongresoLineup).toBe('function');
  });

  describe('adopting the starting state', () => {
    let q: Q;
    beforeEach(() => {
      q = makeQ();
      reconcile(q);
    });

    it('leaves a party that already matches its gate alone (BNG runs, and is live)', () => {
      expect(sup(q, 'bng', 'galicia')).toBeCloseTo(5.0, 6);
    });

    it('folds Amaiur into EH Bildu — the flag says EH Bildu carries the abertzale left', () => {
      expect(sup(q, 'amaiur', 'navarra')).toBe(0);
      expect(sup(q, 'ehbildu', 'navarra')).toBeCloseTo(13.0, 6);
    });

    it('conserves every constituency total', () => {
      const fresh = makeQ();
      for (const c of Object.keys(PARTIES)) {
        expect(total(q, c)).toBeCloseTo(total(fresh, c), 6);
      }
    });
  });

  describe('splitting a party out when its gate turns on', () => {
    it('gives UPN a real vote drained from PP when it stops running inside the PP', () => {
      const q = makeQ();
      reconcile(q);
      const before = total(q, 'navarra');

      q.upn_in_pp = false;
      reconcile(q);

      expect(sup(q, 'upn', 'navarra')).toBeGreaterThan(0);
      expect(sup(q, 'pp', 'navarra')).toBeLessThan(21.0);
      expect(total(q, 'navarra')).toBeCloseTo(before, 6);
    });

    it('drains only live donors — Cs and Vox are absent, so PP funds the whole split', () => {
      const q = makeQ();
      reconcile(q);
      const ppBefore = sup(q, 'pp', 'navarra');

      q.upn_in_pp = false;
      reconcile(q);

      // every point UPN gained came out of PP, since cs/vox sit at 0
      expect(ppBefore - sup(q, 'pp', 'navarra')).toBeCloseTo(sup(q, 'upn', 'navarra'), 6);
      expect(sup(q, 'cs', 'navarra')).toBe(0);
      expect(sup(q, 'vox', 'navarra')).toBe(0);
    });

    it('spreads the drain across Cs and Vox once they exist', () => {
      const q = makeQ({
        cs_congreso_navarra_support: 8.0,
        vox_congreso_navarra_support: 4.0,
        abstain_congreso_navarra_support: 24.0,
      });
      reconcile(q);
      const before = { pp: sup(q, 'pp', 'navarra'), cs: 8.0, vox: 4.0 };

      q.upn_in_pp = false;
      reconcile(q);

      expect(sup(q, 'cs', 'navarra')).toBeLessThan(before.cs);
      expect(sup(q, 'vox', 'navarra')).toBeLessThan(before.vox);
      expect(sup(q, 'pp', 'navarra')).toBeLessThan(before.pp);
    });

    it('seeds Compromís from the left pool when it starts running', () => {
      const q = makeQ();
      reconcile(q);
      const before = total(q, 'valencia');
      expect(sup(q, 'compromis', 'valencia')).toBe(0);

      q.spa_compromis_active = true;
      reconcile(q);

      expect(sup(q, 'compromis', 'valencia')).toBeGreaterThan(0);
      expect(sup(q, 'psoe', 'valencia')).toBeLessThan(20.0);
      expect(total(q, 'valencia')).toBeCloseTo(before, 6);
    });
  });

  describe('folding a party back in when its gate turns off', () => {
    it('empties BNG into the left carriers', () => {
      const q = makeQ();
      reconcile(q);
      const before = total(q, 'galicia');
      const psoeBefore = sup(q, 'psoe', 'galicia');

      q.spa_bng_active = false;
      reconcile(q);

      expect(sup(q, 'bng', 'galicia')).toBe(0);
      expect(sup(q, 'psoe', 'galicia')).toBeGreaterThan(psoeBefore);
      expect(total(q, 'galicia')).toBeCloseTo(before, 6);
    });

    it('restores the folded magnitude on the way back out, not a fresh share', () => {
      const q = makeQ();
      reconcile(q);

      q.spa_bng_active = false;
      reconcile(q);
      expect(sup(q, 'bng', 'galicia')).toBe(0);

      q.spa_bng_active = true;
      reconcile(q);
      // BNG folded at 5.0, so it comes back at 5.0 — the gates flip back and
      // forth (post_event.scene.dry re-resolves them on failed elections), and
      // a round trip must not quietly resize the party.
      expect(sup(q, 'bng', 'galicia')).toBeCloseTo(5.0, 6);
    });
  });

  describe('idempotency', () => {
    it('does nothing when no gate has changed, however many times it runs', () => {
      const q = makeQ();
      reconcile(q);
      const snapshot = JSON.stringify(q);
      for (let i = 0; i < 12; i++) reconcile(q);
      expect(JSON.stringify(q)).toBe(snapshot);
    });

    it('survives a save/load round trip without re-applying a fold', () => {
      const q = makeQ();
      reconcile(q);
      q.spa_bng_active = false;
      reconcile(q);
      const psoeAfterFold = sup(q, 'psoe', 'galicia');

      // what setState does: JSON round trip, and on-arrival never re-runs
      const loaded = JSON.parse(JSON.stringify(q));
      reconcile(loaded);

      expect(sup(loaded, 'psoe', 'galicia')).toBeCloseTo(psoeAfterFold, 6);
      expect(sup(loaded, 'bng', 'galicia')).toBe(0);
    });
  });

  describe('guards', () => {
    it('never drives a donor negative, even asking for more than the pool holds', () => {
      const q = makeQ({
        pp_congreso_navarra_support: 0.4,
        abstain_congreso_navarra_support: 56.6,
      });
      reconcile(q);
      q.upn_in_pp = false;
      reconcile(q);

      for (const p of PARTIES.navarra) expect(sup(q, p, 'navarra')).toBeGreaterThanOrEqual(0);
      expect(total(q, 'navarra')).toBeCloseTo(100, 6);
    });

    it('does nothing when the donor pool is entirely absent', () => {
      const q = makeQ({
        pp_congreso_navarra_support: 0,
        cs_congreso_navarra_support: 0,
        vox_congreso_navarra_support: 0,
        abstain_congreso_navarra_support: 57.0,
      });
      reconcile(q);
      q.upn_in_pp = false;
      expect(() => reconcile(q)).not.toThrow();
      expect(sup(q, 'upn', 'navarra')).toBe(0);
    });

    it('tolerates a Q with no congreso state at all', () => {
      expect(() => reconcile({})).not.toThrow();
    });
  });
});
