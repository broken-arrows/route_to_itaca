import { describe, it, expect } from 'vitest';
import { computeScale, STAGE_W, STAGE_H } from '../src/engine/stage';

describe('computeScale', () => {
  it('exports the design stage size', () => {
    expect(STAGE_W).toBe(1512);
    expect(STAGE_H).toBe(860);
  });
  it('fits width-constrained viewports', () => {
    expect(computeScale(756, 2000)).toBeCloseTo(0.5);
  });
  it('fits height-constrained viewports', () => {
    expect(computeScale(3000, 430)).toBeCloseTo(0.5);
  });
  it('scales up on large screens', () => {
    expect(computeScale(3024, 1720)).toBeCloseTo(2);
  });
  it('accepts custom stage dimensions', () => {
    expect(computeScale(500, 500, 1000, 250)).toBeCloseTo(0.5);
  });
});
