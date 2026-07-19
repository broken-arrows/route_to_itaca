import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { mount } from '@vue/test-utils';
import Hemicycle from '../src/components/viz/Hemicycle.vue';

const SEATS = [
  { party: 'ciu', seats: 62, colour: 'ciu' },
  { party: 'erc', seats: 10, colour: 'erc' },
  { party: 'psc', seats: 28, colour: 'psc' },
];

describe('Hemicycle', () => {
  it('renders one arc per seat', () => {
    const w = mount(Hemicycle, { props: { seats: SEATS, majority: 68 } });
    expect(w.findAll('path.seat')).toHaveLength(100);
  });

  it('colours each seat from its party token', () => {
    const w = mount(Hemicycle, { props: { seats: SEATS, majority: 68 } });
    expect(w.get('path.seat').attributes('fill')).toBe('var(--ciu)');
  });

  it('marks the majority line', () => {
    const w = mount(Hemicycle, { props: { seats: SEATS, majority: 68 } });
    expect(w.find('[data-majority="68"]').exists()).toBe(true);
  });

  it('renders nothing but stays mounted when given no seats', () => {
    const w = mount(Hemicycle, { props: { seats: [], majority: 0 } });
    expect(w.findAll('path.seat')).toHaveLength(0);
    expect(w.find('svg').exists()).toBe(true);
  });

  // CUP is a central Catalan party this card surfaces. The glossary stores its
  // colour as a raw hex (no --var), so tokens.css originally lacked --cup and
  // CUP seats rendered black. Guard the token so it cannot silently regress.
  it('resolves CUP to a defined palette token, not black', () => {
    const w = mount(Hemicycle, { props: { seats: [{ party: 'cup', seats: 10, colour: 'cup' }], majority: 68 } });
    expect(w.get('path.seat').attributes('fill')).toBe('var(--cup)');
    const tokens = readFileSync(resolve(__dirname, '../src/styles/tokens.css'), 'utf8');
    expect(tokens, '--cup must be defined in tokens.css or CUP seats render black').toMatch(/--cup:\s*#/);
  });

  it('renders a raw hex colour literally (token-or-hex tolerant, like the glossary)', () => {
    const w = mount(Hemicycle, { props: { seats: [{ party: 'x', seats: 1, colour: '#b8a12b' }], majority: 68 } });
    expect(w.get('path.seat').attributes('fill')).toBe('#b8a12b');
  });

  it('consumes the q prop instead of leaking it onto the svg', () => {
    const w = mount(Hemicycle, { props: { seats: SEATS, majority: 68, q: { anything: 1 } } });
    expect(w.get('svg').attributes('q')).toBeUndefined();
  });
});
