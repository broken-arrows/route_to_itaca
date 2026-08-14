import { afterEach, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import Hemicycle from '../src/components/viz/Hemicycle.vue';
import { useGameStore } from '../src/stores/game';

const SEATS = [
  { party: 'ciu', seats: 62, colour: 'ciu' },
  { party: 'erc', seats: 10, colour: 'erc' },
  { party: 'psc', seats: 28, colour: 'psc' },
];

const GAME = {
  scenes: { root: { id: 'root', type: 'scene', content: [], options: [] } },
  qualities: {},
  qdisplays: {},
  data: {
    glossary: {
      terms: [{
        id: 'ciu', match: ['CiU'], display: 'CiU', colour: 'ciu',
        tooltip: { title: 'Convergència i Unió', img: 'img/ciu.png' },
      }],
    },
  },
};

function mountHemicycle(props: Record<string, unknown> = {}) {
  const pinia = createPinia();
  setActivePinia(pinia);
  useGameStore().initFromText(JSON.stringify(GAME));
  return mount(Hemicycle, {
    props: { seats: SEATS, majority: 68, ...props },
    global: { plugins: [pinia] },
  });
}

describe('Hemicycle', () => {
  afterEach(() => { document.body.innerHTML = ''; });

  it('renders one arc per seat', () => {
    const w = mountHemicycle();
    expect(w.findAll('path.seat')).toHaveLength(100);
  });

  it('colours each seat from its party token', () => {
    const w = mountHemicycle();
    expect(w.get('path.seat').attributes('fill')).toBe('var(--ciu)');
  });

  it('does not draw a majority line through the seats', () => {
    const w = mountHemicycle();
    expect(w.find('.majority-line').exists()).toBe(false);
  });

  it('animates only when the authored surface opts in', () => {
    const staticView = mountHemicycle();
    const electionView = mountHemicycle({ animate: true });
    expect(staticView.find('.hemicycle').classes()).not.toContain('is-animated');
    expect(electionView.find('.hemicycle').classes()).toContain('is-animated');
    const style = electionView.find('.seat-position').attributes('style');
    expect(style).toContain('--seat-duration:');
    expect(style).not.toContain('--seat-delay:');
  });

  it('matches the old interaction: generous hit targets highlight a whole party and show its glossary tooltip', async () => {
    const w = mountHemicycle();
    const hit = w.get('.seat-hit.ciu');
    expect(Number(hit.attributes('r'))).toBeGreaterThan(0);

    await hit.trigger('mouseenter');
    expect(w.findAll('path.seat.ciu').every((seat) => seat.classes().includes('party-hovered'))).toBe(true);
    expect(w.findAll('path.seat.erc').every((seat) => seat.classes().includes('party-nothovered'))).toBe(true);
    const tooltip = document.querySelector('[data-test="hemicycle-tooltip"]');
    expect(tooltip?.textContent).toContain('CiU');
    expect(tooltip?.textContent).toContain('62 seats');
    expect(tooltip?.querySelector('img')?.getAttribute('src')).toContain('img/ciu.png');

    await hit.trigger('mouseleave');
    expect(document.querySelector('[data-test="hemicycle-tooltip"]')).toBeNull();
  });

  it('renders nothing but stays mounted when given no seats', () => {
    const w = mountHemicycle({ seats: [], majority: 0 });
    expect(w.findAll('path.seat')).toHaveLength(0);
    expect(w.find('svg').exists()).toBe(true);
  });

  // CUP is a central Catalan party this card surfaces. The glossary stores its
  // colour as a raw hex (no --var), so tokens.css originally lacked --cup and
  // CUP seats rendered black. Guard the token so it cannot silently regress.
  it('resolves CUP to a defined palette token, not black', () => {
    const w = mountHemicycle({ seats: [{ party: 'cup', seats: 10, colour: 'cup' }] });
    expect(w.get('path.seat').attributes('fill')).toBe('var(--cup)');
    const tokens = readFileSync(resolve(__dirname, '../src/styles/tokens.css'), 'utf8');
    expect(tokens, '--cup must be defined in tokens.css or CUP seats render black').toMatch(/--cup:\s*#/);
  });

  it('renders a raw hex colour literally (token-or-hex tolerant, like the glossary)', () => {
    const w = mountHemicycle({ seats: [{ party: 'x', seats: 1, colour: '#b8a12b' }] });
    expect(w.get('path.seat').attributes('fill')).toBe('#b8a12b');
  });

  it('consumes the q prop instead of leaking it onto the svg', () => {
    const w = mountHemicycle({ q: { anything: 1 } });
    expect(w.get('svg').attributes('q')).toBeUndefined();
  });
});
