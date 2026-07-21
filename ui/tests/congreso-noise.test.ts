import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gameLib } from '../src/game-bindings';

// ---------------------------------------------------------------------------
// Step 4j's gaussian shock used to be added straight into support, which made it
// a RANDOM WALK: variance grew with time, so a party's decade-long fate was
// decided by noise rather than by anything the player or the world did (BNG
// measured at [1.38, 9.37] from a 5.0 start), and small parties walked into
// support 0 — an ABSORBING state, since `activeParties` filters `> 0`.
//
// The shock is still additive in percentage points. What changed: its width
// scales by sqrt(p(1-p)) (the sampling-error law), and it is carried in a
// mean-reverting deviation so it cannot accumulate. Structure is permanent,
// noise is transient.
// ---------------------------------------------------------------------------

const game = JSON.parse(readFileSync(resolve(__dirname, '../../out/game.json'), 'utf8'));

type Q = Record<string, any>;

/** Boot a real game state, then hold every macro driver constant so the only
 *  force acting on support is step 4j. Any drift observed is noise, by
 *  construction. */
function quietGame(): Q {
  const prevWindow = (globalThis as any).window;
  (globalThis as any).window = { location: { protocol: 'https:', hostname: 'test', href: '' } };
  const prevLog = console.log;
  const prevInfo = console.info;
  console.log = () => {};
  console.info = () => {};
  try {
    const q: Q = {};
    for (const id of ['root.start_menu', 'root.start']) {
      for (const blk of game.scenes[id].onArrival) {
        new Function('state', 'Q', 'G', blk.$code).call(
          { achieve() {}, game, _compileChoices: () => null },
          { qualities: q },
          q,
          gameLib,
        );
      }
    }
    q.year = 2012;
    q.month = 8;
    q.time = 1;
    q.gdp_growth = 1;
    q.unemployment = 18;
    q.welfare_index = 50;
    q.social_dissent = 40;
    q.cat_spa_relations = 45;
    q.podemos_channeling = 0;
    q.corruption_pp = 0;
    q.corruption_psoe = 0;
    return q;
  } finally {
    console.log = prevLog;
    console.info = prevInfo;
    (globalThis as any).window = prevWindow;
  }
}

function tick(q: Q, months: number) {
  const prevLog = console.log;
  const prevInfo = console.info;
  console.log = () => {};
  console.info = () => {};
  try {
    for (let m = 0; m < months; m++) {
      q.time++;
      q.month++;
      if (q.month > 12) {
        q.month = 1;
        q.year++;
      }
      (gameLib as any).engineTick(q);
    }
  } finally {
    console.log = prevLog;
    console.info = prevInfo;
  }
}

const sup = (q: Q, p: string, c: string) => q[`${p}_congreso_${c}_support`] || 0;

const TRIALS = 40;
const MONTHS = 85; // Aug 2012 -> 2019, the game's full span

describe('congreso noise', () => {
  // Bands are RELATIVE to each party's own starting support. Absolutes would
  // re-break on every calibration change (they did, when turnout was rescaled
  // and BNG's start moved 5.0 -> 5.50); the property under test is that noise
  // does not accumulate, not what the current numbers happen to be.
  const finalsFor = (party: string, c: string) => {
    const start = sup(quietGame(), party, c);
    const finals: number[] = [];
    for (let t = 0; t < TRIALS; t++) {
      const q = quietGame();
      tick(q, MONTHS);
      finals.push(sup(q, party, c));
    }
    return { start, finals };
  };

  it('does not accumulate — a mid-size party stays near its structural level', () => {
    const { start, finals } = finalsFor('bng', 'galicia');
    // No macro movement, so any drift is noise. As a random walk this measured
    // [1.38, 9.37] from a 5.0 start — i.e. -72%/+87%, far outside this band.
    expect(Math.min(...finals)).toBeGreaterThan(start * 0.75);
    expect(Math.max(...finals)).toBeLessThan(start * 1.25);
  });

  it('is unbiased — it must not push a party systematically up or down', () => {
    const { start, finals } = finalsFor('pnv', 'euskadi');
    const mean = finals.reduce((a, b) => a + b, 0) / finals.length;
    expect(mean).toBeGreaterThan(start * 0.95);
    expect(mean).toBeLessThan(start * 1.05);
  });

  it('never walks a small party into the absorbing zero', () => {
    // `te` sits at 0.05 and died in 32-39% of runs under the walk, permanently:
    // support 0 drops it out of `activeParties`, so it can never recover.
    for (let t = 0; t < TRIALS; t++) {
      const q = quietGame();
      tick(q, MONTHS);
      expect(sup(q, 'te', 'rest')).toBeGreaterThan(0);
      expect(sup(q, 'prc', 'rest')).toBeGreaterThan(0);
    }
  });

  it('leaves a deliberately folded party at zero — noise must not zombie it back', () => {
    const q = quietGame();
    q.spa_bng_active = false;
    (gameLib as any).reconcileCongresoLineup(q);
    expect(sup(q, 'bng', 'galicia')).toBe(0);
    tick(q, MONTHS);
    expect(sup(q, 'bng', 'galicia')).toBe(0);
  });

  it('keeps structural change permanent — an injection is not reverted away', () => {
    const q = quietGame();
    tick(q, 1); // settle the macro deltas to zero
    const before = sup(q, 'bng', 'galicia');
    (gameLib as any).spaSupportInject(q, 'bng', 'galicia', 4.0, 'psoe');
    const injected = sup(q, 'bng', 'galicia');
    expect(injected).toBeGreaterThan(before + 3.5);

    tick(q, 40);
    // The wobble is transient; the injection is not. BNG must still be sitting
    // at its new structural level, not have decayed back toward 5.
    expect(sup(q, 'bng', 'galicia')).toBeGreaterThan(injected - 1.5);
  });
});
