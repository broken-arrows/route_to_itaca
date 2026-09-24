import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { useGameStore } from '../src/stores/game';
import { gameLib } from '../src/game-bindings';
import ParlamentResultsMap from '../src/components/viz/ParlamentResultsMap.vue';
import CongresoResultsMap from '../src/components/viz/CongresoResultsMap.vue';
import LocalResultsMap from '../src/components/viz/LocalResultsMap.vue';
import CongresoPartyTour from '../src/components/viz/CongresoPartyTour.vue';

const here = resolve(__dirname, '../..');
const asset = (name: string) => readFileSync(resolve(here, 'source/img/maps', name), 'utf8');
beforeEach(() => {
  setActivePinia(createPinia());
  const game = useGameStore();
  game.initFromText(readFileSync(resolve(here, 'out/game.json'), 'utf8'));
  game.newGame();
  vi.stubGlobal('fetch', vi.fn(async (url: string) => ({
    ok: true,
    text: async () => asset(url.includes('local-results') ? 'local-results.svg'
      : url.includes('congreso-results') ? 'congreso-results.svg' : 'catalonia-provinces.svg'),
  })));
});

describe('generic election result widgets', () => {
  it('the Parlament result uses the same province data and seat projection as the poll model', async () => {
    const q = { ...useGameStore().q,
      parties: ['erc', 'ciu'], parlament_demographics: ['buss'],
      parlament_barcelona_buss_pop: 100,
      erc_parlament_barcelona_buss_support: 60,
      ciu_parlament_barcelona_buss_support: 40,
      parlament_seats: { barcelona: 85 },
    };
    const wrapper = mount(ParlamentResultsMap, { props: { q } });
    await flushPromises();
    expect(wrapper.find('[data-layout="wide"]').exists()).toBe(true);
    expect(wrapper.findAll('.province-tabs button')).toHaveLength(4);
    expect(wrapper.find('.projection').exists()).toBe(true);
    expect(wrapper.find('.crosstab [data-term="erc"]').exists()).toBe(true);
    expect(wrapper.find('.projection [data-term="erc"]').exists()).toBe(true);
    expect(wrapper.find('#barcelona').attributes('data-term')).toBe('erc');
  });

  it('paints Congreso winners and opens the selected constituency seat bars', async () => {
    const q = { ...useGameStore().q,
      congreso_catalunya_wp: 'erc', congreso_parties_catalunya: ['erc', 'pp'],
      erc_congreso_s_catalunya: 9, pp_congreso_s_catalunya: 2,
      erc_congreso_catalunya_support: 40, pp_congreso_catalunya_support: 20,
      congreso_seats: { catalunya: 11 },
    };
    const wrapper = mount(CongresoResultsMap, { props: { q } });
    await flushPromises();
    expect(wrapper.find('#catalonia').classes()).toContain('election-region');
    expect(wrapper.find('#catalonia').attributes('data-term')).toBe('erc');
    await wrapper.find('#catalonia').trigger('click');
    expect(wrapper.find('.panel-title').text()).toContain('Catalonia');
    expect(wrapper.findAll('.bar-column')).toHaveLength(2);
    expect(wrapper.findAll('.bar-column')[0].text()).toContain('9');
    expect(wrapper.find('.bar-column [data-term="erc"]').exists()).toBe(true);
  });

  it('uses stored local winners for comarques and exact city names in the click legend', async () => {
    const q = { ...useGameStore().q, 'local_barcelones_wp': 'erc', 'local_barcelona_wp': 'cup' };
    const model = gameLib.localResultsMap(q);
    expect(model.cities.find(city => city.id === 'barcelona')?.name).toBe('Barcelona');
    const wrapper = mount(LocalResultsMap, { props: { q } });
    await flushPromises();
    expect(wrapper.find('#barcelones').exists()).toBe(true);
    expect(wrapper.find('#barcelona').attributes('aria-label')).toContain('Barcelona');
    expect(wrapper.find('#barcelona').attributes('data-term')).toBe('cup');
    await wrapper.find('#barcelona').trigger('click');
    expect(wrapper.find('.legend').text()).toContain('Barcelona');
    expect(wrapper.find('.legend').text()).toContain('CUP');
    await wrapper.setProps({ q: { ...q, local_barcelona_wp: null, local_barcelones_wp: null } });
    await flushPromises();
    expect(wrapper.find('.legend').text()).toContain('Click a city');
    expect(wrapper.find('#barcelona').attributes('aria-label')).toBeUndefined();
  });

  it('reveals party-tour logos only for viewed regions and respects active party successors', async () => {
    const q = { ...useGameStore().q,
      congreso_party_tour_viewed_catalonia: true, junts_formed: true,
      congreso_party_tour_highlight: ['catalonia'],
    };
    const wrapper = mount(CongresoPartyTour, { props: { q } });
    await flushPromises();
    const cat = wrapper.find('[data-interest-region="catalonia"]');
    expect(cat.exists()).toBe(true);
    expect(cat.find('image[href*="logo_junts"]').exists()).toBe(true);
    expect(wrapper.find('[data-interest-region="catalonia"] [data-term="junts"]').exists()).toBe(true);
    expect(wrapper.find('[data-interest-region="rest"]').exists()).toBe(false);
    const firstBox = wrapper.find('svg').attributes('viewBox');
    const firstGroups = wrapper.findAll('.party-tour-logos').length;
    await wrapper.setProps({ q: { ...q, spa_te_active: true } });
    await flushPromises();
    expect(wrapper.find('svg').attributes('viewBox')).toBe(firstBox);
    expect(wrapper.findAll('.party-tour-logos')).toHaveLength(firstGroups);
  });
});
