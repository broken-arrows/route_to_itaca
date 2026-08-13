import { beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { useGameStore } from '../src/stores/game';
import LevelBars from '../src/components/viz/LevelBars.vue';
import TensionRows from '../src/components/viz/TensionRows.vue';
import SeatBars from '../src/components/viz/SeatBars.vue';
import RosterRows from '../src/components/viz/RosterRows.vue';
import LeaderRows from '../src/components/viz/LeaderRows.vue';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');

beforeEach(() => {
  setActivePinia(createPinia());
  const game = useGameStore();
  game.initFromText(readFileSync(GAME, 'utf8'));
  game.newGame();
});

describe('Brief row widgets', () => {
  it('level-bars clamps geometry and classifies the displayed word through the qdisplay', () => {
    const wrapper = mount(LevelBars, {
      props: {
        rows: [
          {
            id: 'social',
            label: 'Social dissent',
            value: 45,
            valueDisplay: 'social_dissent',
            share: 1.4,
          },
        ],
      },
    });
    expect(wrapper.find('.level-fill').attributes('style')).toContain('width: 100%');
    expect(wrapper.find('.level-word').text()).toBe('medium');
    expect(wrapper.find('.level-word').attributes('data-band')).toBe('medium');
  });

  it('tension-rows keeps strength and dissent as two independently banded words', () => {
    const wrapper = mount(TensionRows, {
      props: {
        rows: [
          {
            id: 'left',
            label: 'Left',
            strength: 10,
            dissent: 90,
            strengthDisplay: 'dissent',
            dissentDisplay: 'dissent',
          },
        ],
      },
    });
    const bands = wrapper.findAll('[data-band]');
    expect(bands.map((node) => node.attributes('data-band'))).toEqual(['low', 'very_high']);
  });

  it('seat-bars sizes segments by the whole chamber and places the majority tick against it', () => {
    const wrapper = mount(SeatBars, {
      props: {
        rows: [
          { id: 'ciu', label: 'CiU', value: 62, majority: 68 },
          { id: 'psc', label: 'PSC', value: 28, majority: 68 },
          { id: 'erc', label: 'ERC', value: 45, majority: 68 },
        ],
      },
    });
    const widths = wrapper.findAll('.seat-segment').map((node) => node.attributes('style'));
    expect(widths[0]).toContain(`${(62 / 135) * 100}%`);
    expect(wrapper.find('.majority-tick').attributes('style')).toContain(`${(68 / 135) * 100}%`);
    expect(wrapper.find('.seat-strip').attributes('aria-label')).toBe('135 seats; majority 68');
  });

  it('roster-rows distinguishes the player stamp and international flag/stance rows', () => {
    const benches = mount(RosterRows, {
      props: {
        rows: [
          {
            id: 'erc',
            label: 'ERC',
            value: 10,
            stamp: null,
            stampDisplay: 'relationships',
            subtitle: 'Oriol Junqueras — left',
            isPlayer: true,
          },
          {
            id: 'si',
            label: 'SI',
            value: 4,
            stamp: 75,
            stampDisplay: 'relationships',
            subtitle: 'Alfons López Tena — unilateralism',
          },
        ],
      },
    });
    expect(benches.find('.roster-stamp').text()).toBe('you');
    expect(benches.find('.roster-stamp').attributes('data-band')).toBe('you');
    expect(benches.findAll('.roster-logo')[0].attributes('src')).toBe(
      'http://localhost:3000/img/parties/logo_erc.png',
    );
    expect(benches.findAll('.roster-logo')[1].attributes('src')).toBe(
      'http://localhost:3000/img/parties/logo_si.jpg',
    );
    expect(benches.find('.roster-roundel').exists()).toBe(false);

    const world = mount(RosterRows, {
      props: {
        rows: [
          {
            id: 'eu',
            label: 'European Union',
            value: 1,
            valueDisplay: 'international_opinion',
            flag: 'img/flags/eu.svg',
          },
        ],
      },
    });
    expect(world.find('.roster-flag').attributes('src')).toBe('img/flags/eu.svg');
    expect(world.find('.roster-stamp').text()).toBe('watching');
    expect(world.find('.roster-copy > i').text()).toContain("internal matter");
  });

  it('leader-rows renders party markers for people and label-only state control', () => {
    const cabinet = mount(LeaderRows, {
      props: {
        rows: [
          { id: 'economy', label: 'Economy', value: 'Andreu Mas-Colell', party: 'ciu' },
        ],
      },
    });
    expect(cabinet.find('.party-square').exists()).toBe(true);
    expect(cabinet.find('.person-value').text()).toContain('Andreu Mas-Colell');

    const control = mount(LeaderRows, {
      props: {
        rows: [
          { id: 'security', label: 'Security', value: 4, valueDisplay: 'control', party: null },
        ],
      },
    });
    expect(control.find('.control-value').attributes('data-band')).toBe('disputed');
    expect(control.find('.control-value b').text()).toBe('Disputed');
    expect(control.find('.pips').exists()).toBe(false);
    expect(control.find('.control-caption').exists()).toBe(false);
  });
});
