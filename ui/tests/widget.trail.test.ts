import { beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import Trail from '../src/components/viz/Trail.vue';

// Design §1 shape 1 anchors (also BriefSheet's band ramp): good green / bad red.
const GOOD = '#3f8f3f';
const BAD = '#b03030';

// jsdom serialises an inline `color` hex into `rgb(...)`; normalise the expected
// hex the same way so the two compare like-for-like without hardcoding rgb().
function asRendered(hex: string): string {
  const el = document.createElement('span');
  el.style.color = hex;
  return el.style.color;
}

type Dir = 'up' | 'down' | 'flat';
interface TrailRow {
  id: string;
  label: string;
  value: number;
  unit: string;
  dir: Dir;
  good: boolean;
  series: number[];
}

// The verified `trails()` row shape (source/lib/brief.js). Defaults are a plain
// all-positive GDP row; each test overrides only what it exercises.
function makeRow(over: Partial<TrailRow> = {}): TrailRow {
  return {
    id: 'gdp',
    label: 'GDP growth',
    value: 2.1,
    unit: '%',
    dir: 'up',
    good: true,
    series: [1, 2, 3],
    ...over,
  };
}

function mountTrail(rows: TrailRow[]) {
  setActivePinia(createPinia());
  return mount(Trail, { props: { rows } });
}

beforeEach(() => {
  setActivePinia(createPinia());
});

describe('Trail widget', () => {
  it('renders one block per row (the three ECONOMY trails)', () => {
    const w = mountTrail([
      makeRow({ id: 'gdp' }),
      makeRow({ id: 'unemployment' }),
      makeRow({ id: 'surplus' }),
    ]);
    expect(w.findAll('.trail-block')).toHaveLength(3);
  });

  it('degrades on an empty series: no crash, no line, header value still shown', () => {
    const w = mountTrail([makeRow({ series: [] })]);
    // The block still exists — never a crash, never a broken axis.
    expect(w.find('[data-test="trail-gdp"]').exists()).toBe(true);
    // No plot, no line path — the run has not started.
    expect(w.find('.trail-plot').exists()).toBe(false);
    expect(w.find('.trail-line').exists()).toBe(false);
    // The scalar summary is still legible.
    expect(w.find('.trail-value').text()).toContain('2.1%');
    // And the neutral start-of-run caption stands in for the trail.
    expect(w.find('.trail-empty').exists()).toBe(true);
  });

  it('draws a lone dot (no line) for a single reading', () => {
    const w = mountTrail([makeRow({ series: [1.5] })]);
    expect(w.find('.trail-plot').exists()).toBe(true);
    expect(w.find('.trail-line').exists()).toBe(false);
    expect(w.find('.trail-dot').exists()).toBe(true);
  });

  it('an all-positive series sits the zero ground line at the bottom of the box', () => {
    const w = mountTrail([makeRow({ series: [1, 2, 3], value: 3 })]);
    const zero = w.find('.trail-zero');
    expect(zero.exists()).toBe(true);
    // min(0,…)=0, max(0,…)=3 -> zero is the domain floor -> at the bottom.
    expect(Number(zero.attributes('data-zero-frac'))).toBeCloseTo(1, 5);
    const zeroY = Number(zero.attributes('y1'));
    const dotY = Number(w.find('.trail-dot').attributes('cy'));
    // Current value 3 (>0) sits ABOVE the zero line (smaller y in SVG space).
    expect(dotY).toBeLessThan(zeroY);
  });

  it('a series crossing zero places the zero line inside the box and maps negatives below it', () => {
    // hi = max(0,3) = 3, lo = min(0,-2) = -2, current = -2 (negative).
    const w = mountTrail([
      makeRow({ series: [1, 2, -1, 3, -2], value: -2, dir: 'down', good: false }),
    ]);
    const zero = w.find('.trail-zero');
    const frac = Number(zero.attributes('data-zero-frac'));
    // Strictly inside the box, and at the exact domain fraction 3/(3+2)=0.6.
    expect(frac).toBeGreaterThan(0);
    expect(frac).toBeLessThan(1);
    expect(frac).toBeCloseTo(0.6, 5);
    const zeroY = Number(zero.attributes('y1'));
    // Zero line lies within the plotted band (T=8 … B=102).
    expect(zeroY).toBeGreaterThan(8);
    expect(zeroY).toBeLessThan(102);
    // The current (negative) reading's dot is BELOW the zero line (larger y).
    const dotY = Number(w.find('.trail-dot').attributes('cy'));
    expect(dotY).toBeGreaterThan(zeroY);
  });

  it('colours the value and the dot by `good`, never by `dir`', () => {
    // A RISE that is BAD for the metric (unemployment up): dir=up but good=false.
    // A dir-driven implementation would wrongly ink this green.
    const risingBad = makeRow({
      id: 'u', label: 'Unemployment', value: 1, dir: 'up', good: false, series: [1],
    });
    // A FALL that is GOOD (unemployment easing): dir=down but good=true.
    const fallingGood = makeRow({
      id: 's', label: 'Surplus', value: -1, dir: 'down', good: true, series: [1],
    });
    const w = mountTrail([risingBad, fallingGood]);
    const values = w.findAll('.trail-value');
    // good=false -> RED, even though dir=up.
    expect((values[0].element as HTMLElement).style.color).toBe(asRendered(BAD));
    // good=true -> GREEN, even though dir=down.
    expect((values[1].element as HTMLElement).style.color).toBe(asRendered(GOOD));
    // The inked current-month dot follows the SAME good/bad colour.
    const dots = w.findAll('.trail-dot');
    expect(dots[0].attributes('fill')).toBe(BAD);
    expect(dots[1].attributes('fill')).toBe(GOOD);
    // The direction ARROW still reflects `dir` (▲ up / ▼ down) — orthogonal to colour.
    expect(values[0].text()).toContain('▲');
    expect(values[1].text()).toContain('▼');
  });
});
