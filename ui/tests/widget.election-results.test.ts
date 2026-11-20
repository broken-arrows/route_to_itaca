import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { i18n } from '../src/i18n';
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
    const wrapper = mount(ParlamentResultsMap, { global: { plugins: [i18n] }, props: { q } });
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
    const wrapper = mount(CongresoResultsMap, { global: { plugins: [i18n] }, props: { q } });
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
    const wrapper = mount(LocalResultsMap, { global: { plugins: [i18n] }, props: { q } });
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
    const wrapper = mount(CongresoPartyTour, { global: { plugins: [i18n] }, props: { q } });
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

// Rest-of-Spain map communities. Andalusia carries a distinctive result; the
// others share a one-seat placeholder so every region has its own record.
const REST_IDS = ['madrid', 'andalusia', 'extremadura', 'castile_la_mancha', 'castile_and_leon', 'murcia',
  'la_rioja', 'aragon', 'asturias', 'cantabria', 'canary_islands', 'ceuta', 'melilla'];
function communityQ() {
  const communities: Record<string, unknown> = Object.fromEntries(REST_IDS.map(id =>
    [id, { total: 1, seats: { pp: 1, psoe: 0 }, votes: { pp: 2, psoe: 1 } }]));
  communities.andalusia = { total: 61, seats: { pp: 22, psoe: 25, podemos: 14, cs: 0 },
    votes: { pp: 280, psoe: 300, podemos: 200, cs: 20 } };
  return { ...pooledRestQ(), congreso_rest_communities: communities, congreso_prov_andalusia_wp: 'psoe' };
}
// A save from before community results: only the pooled electorate and winners.
function pooledRestQ() {
  return {
    ...Object.fromEntries(REST_IDS.map(id => [`congreso_prov_${id}_wp`, 'pp'])),
    congreso_parties_rest: ['pp', 'psoe', 'cs'],
    pp_congreso_s_rest: 120, psoe_congreso_s_rest: 96, cs_congreso_s_rest: 0,
    pp_congreso_rest_support: 30, psoe_congreso_rest_support: 25, cs_congreso_rest_support: 2,
    congreso_seats: { rest: 216 },
  };
}

describe('Congreso rest-of-Spain community panels', () => {
  it('opens each community on its own and localizes its name', async () => {
    const model = gameLib.congresoResultsMap(communityQ());
    expect(model.panels.rest).toBeUndefined();
    expect(model.panels.andalusia.rows.map(row => [row.party, row.seats])).toEqual([['psoe', 25], ['pp', 22], ['podemos', 14]]);
    expect(model.panels.andalusia.rows[0].support).toBeCloseTo(37.5);
    const wrapper = mount(CongresoResultsMap, { global: { plugins: [i18n] }, props: { q: { ...useGameStore().q, ...communityQ() } } });
    await flushPromises();
    await wrapper.find('#andalusia').trigger('mousemove');
    expect(wrapper.find('#madrid').classes()).not.toContain('rest-hover');
    await wrapper.find('#andalusia').trigger('click');
    expect(wrapper.find('.panel-title').text()).toContain('Andalusia');
    expect(wrapper.find('.panel-title').text()).toContain('61 seats');
    expect(wrapper.findAll('.bar-column').map(column => column.find('strong').text())).toEqual(['25', '22', '14']);
    expect(wrapper.find('#andalusia').classes()).toContain('selected');
    expect(wrapper.find('#madrid').classes()).not.toContain('selected');
    i18n.global.locale.value = 'ca';
    try {
      await wrapper.find('#castile_la_mancha').trigger('click');
      expect(wrapper.find('.panel-title').text()).toContain('Castella-la Manxa');
      await wrapper.find('#catalonia').trigger('click');
      expect(wrapper.find('.panel-title b').text()).toBe('Catalunya');
    } finally { i18n.global.locale.value = 'en'; }
  });

  it('falls back to the pooled rest panel for saves without community results', async () => {
    const model = gameLib.congresoResultsMap(pooledRestQ());
    expect(model.regions.filter(region => region.constituency === 'rest').every(region => region.panel === 'rest')).toBe(true);
    expect(Object.keys(model.panels)).not.toContain('madrid');
    const wrapper = mount(CongresoResultsMap, { global: { plugins: [i18n] }, props: { q: { ...useGameStore().q, ...pooledRestQ() } } });
    await flushPromises();
    await wrapper.find('#madrid').trigger('mousemove');
    expect(wrapper.find('#andalusia').classes()).toContain('rest-hover');
    await wrapper.find('#madrid').trigger('click');
    expect(wrapper.find('.panel-title').text()).toContain('Rest of Spain');
    expect(wrapper.find('.panel-title').text()).toContain('216 seats');
    expect(wrapper.findAll('.bar-column')).toHaveLength(2);
    expect(wrapper.find('#andalusia').classes()).toContain('selected');
  });

  it('gives the classic shell the same panels from the shared model', () => {
    const log = vi.spyOn(console, 'log').mockImplementation(() => {});
    (window as any).RTI_GAME_LIB = gameLib;
    try {
      new Function(readFileSync(resolve(here, 'out/html/cat_maps.js'), 'utf8'))();
      const render = (q: Record<string, unknown>) => {
        document.body.innerHTML = '<div id="congreso-map-widget"></div>';
        (window as any).initCongresoMap('congreso-map-widget', q);
        return document.getElementById('congreso-map-widget')!;
      };
      let map = render(communityQ());
      (map.querySelector('#andalusia') as SVGElement).dispatchEvent(new Event('mouseenter'));
      expect(map.querySelector('#madrid')!.classList.contains('rest-hover')).toBe(false);
      (map.querySelector('#andalusia') as SVGElement).dispatchEvent(new MouseEvent('click'));
      expect(map.querySelector('.cong-title')!.textContent).toBe('Andalusia61 seats');
      expect([...map.querySelectorAll('.cong-seats-n')].map(node => node.textContent)).toEqual(['25', '22', '14']);
      expect(map.querySelector('#madrid')!.classList.contains('selected')).toBe(false);
      (map.querySelector('#ceuta') as SVGElement).dispatchEvent(new MouseEvent('click'));
      expect(map.querySelector('.cong-title')!.textContent).toBe('Ceuta1 seat');

      map = render(pooledRestQ());
      (map.querySelector('#madrid') as SVGElement).dispatchEvent(new MouseEvent('click'));
      expect(map.querySelector('.cong-title')!.textContent).toBe('Rest of Spain216 seats');
      expect(map.querySelectorAll('.cong-col')).toHaveLength(2);
      expect(map.querySelector('#andalusia')!.classList.contains('selected')).toBe(true);
    } finally {
      log.mockRestore();
      delete (window as any).RTI_GAME_LIB;
      for (const name of ['initCatLocalMap', 'initCongresoMap', 'initCongresoPartyTour']) delete (window as any)[name];
      document.body.innerHTML = '';
    }
  });
});
