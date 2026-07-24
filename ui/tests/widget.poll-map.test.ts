import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { useGameStore } from '../src/stores/game';
import PollMap from '../src/components/viz/PollMap.vue';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');
const MAP = join(__dirname, '..', '..', 'source', 'img', 'maps', 'catalonia-provinces.svg');

beforeEach(() => {
  setActivePinia(createPinia());
  const game = useGameStore();
  game.initFromText(readFileSync(GAME, 'utf8'));
  game.newGame();
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    text: async () => readFileSync(MAP, 'utf8'),
  })));
});

describe('PollMap widget', () => {
  it('full mode renders the map, province controls, crosstab, and projected columns', async () => {
    const q = useGameStore().q;
    const wrapper = mount(PollMap, { props: { q } });
    await flushPromises();
    expect(wrapper.attributes('data-variant')).toBe('full');
    expect(wrapper.findAll('.map-shell .province')).toHaveLength(4);
    expect(wrapper.findAll('.province-tabs button')).toHaveLength(4);
    expect(wrapper.find('.crosstab').exists()).toBe(true);
    expect(wrapper.find('.projection').exists()).toBe(true);

    await wrapper.findAll('.province-tabs button')[1].trigger('click');
    expect(wrapper.findAll('.province-tabs button')[1].attributes('aria-selected')).toBe('true');
    expect(wrapper.find('.province-caption').text().toLowerCase()).toContain('tarragona');
  });

  it('compact is a painted map only; blank is a neutral map only', async () => {
    const q = useGameStore().q;
    const compact = mount(PollMap, { props: { q, variant: 'compact' } });
    await flushPromises();
    expect(compact.find('.map-shell').exists()).toBe(true);
    expect(compact.find('.province-tabs').exists()).toBe(false);
    expect(compact.attributes('style')).toContain('--province-barcelona');

    const blank = mount(PollMap, { props: { q, variant: 'blank' } });
    expect(blank.classes()).toContain('poll-map-blank');
    expect(blank.attributes('style')).toContain('--province-barcelona: #a6a6a6');
    expect(blank.find('.crosstab').exists()).toBe(false);
  });
});
