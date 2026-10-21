import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { beforeEach, expect, it } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { mount } from '@vue/test-utils';
import { extractFiles } from 'dendrynexus-ten/lib/localization-extract.js';
import { useGameStore } from '../src/stores/game';
import Coalitions from '../src/components/viz/Coalitions.vue';
import { DendryAdapter } from '../src/engine/adapter';

const root = resolve(import.meta.dirname, '../..');
const read = (file: string) => readFileSync(join(root, file), 'utf8');
const catalog = {
  'New Tripartite': 'Nou tripartit',
  'PSOE-led Grand Coalition': 'Gran coalició liderada pel PSOE',
  '{party} minority Government': 'Govern minoritari de {party}',
};

beforeEach(() => setActivePinia(createPinia()));

it('captures the dynamic party in the real Parlament producer and preserves it after rebranding and save/load', () => {
  const adapter = DendryAdapter.fromJSONText(read('out/game.json'));
  adapter.beginGame([1, 2, 3, 4]);
  Object.assign(adapter.qualities, {
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    ciu_parlament_s: 50, erc_parlament_s: 21, psc_parlament_s: 20,
    icv_parlament_s: 13, parlament_current_ciu_s: 50, parlament_s_majority: 68,
  });
  adapter.goToScene('parlament_coalition');
  const config = adapter.qualities.parlament_coalition_config as any;
  const label = config.coalitions.find((row: any) => row.labelKey === '{party} minority Government');
  expect(label).toMatchObject({ label: 'ciu minority Government', labelParams: { party: 'ciu' } });
  adapter.qualities.parlament_current_ciu = 'junts';
  adapter.importStateJSON(adapter.exportStateJSON());
  const restored = (adapter.qualities.parlament_coalition_config as any).coalitions
    .find((row: any) => row.labelKey === label.labelKey);
  adapter.setLocale('ca', catalog);
  expect(adapter.translate(restored.labelKey, restored.labelParams)).toBe('Govern minoritari de ciu');
});

it('extracts every authored coalition label, including conditional branches and the dynamic template', async () => {
  const names = [
    'source/scenes/events/elections/parlament_coalition.scene.dry',
    'source/scenes/events/congreso_elections/congreso_coalition.scene.dry',
  ];
  const files = names.map(name => ({ name: join(...name.split('/')), contents: read(name) }));
  const result = await extractFiles(files, { catalogs: { ca: catalog } });
  expect(result.errors).toEqual([]);
  const keys = new Set(result.units.map((unit: { key: string }) => unit.key));
  for (const key of Object.keys(catalog)) expect(keys.has(key), key).toBe(true);
  for (const file of files) {
    const labels = file.contents.split(/\r?\n/).filter(line => line.includes('label:') && !line.includes('label: pb.label'));
    for (const line of labels) {
      expect(line).toContain('G.text(');
      for (const match of line.matchAll(/G\.text\("([^"]+)"/g)) expect(keys.has(match[1]!)).toBe(true);
    }
  }
  for (const locale of ['en', 'ca']) {
    const published = JSON.parse(read(`source/locales/${locale}/content.json`));
    for (const key of Object.keys(catalog)) expect(Object.hasOwn(published, key)).toBe(true);
  }
});

it('translates static and captured dynamic labels in both shells across locale refresh and save/load', async () => {
  const game = useGameStore();
  game.initFromText(read('out/game.json'));
  game.newGame();
  const adapter = game.adapter!;
  const config = {
    majoritySeats: 68,
    coalitions: [
      { label: 'New Tripartite', seatsVar: 'first' },
      { label: 'PSOE-led Grand Coalition', seatsVar: 'second' },
      { label: 'cdc minority Government', labelKey: '{party} minority Government', labelParams: { party: 'cdc' }, seatsVar: 'third' },
    ],
  };
  adapter.qualities.test_coalitions = config;
  const saved = adapter.exportStateJSON();
  adapter.importStateJSON(saved);
  expect(adapter.qualities.test_coalitions).toEqual(config);
  const before = adapter.exportStateJSON();
  const rng = adapter.engine.random.getState();
  const q = { first: 70, second: 70, third: 50 };
  const wrapper = mount(Coalitions, { props: { ...config, q } });
  Object.assign(window, {
    ClassicI18n: { content: (key: string, params?: Record<string, unknown>) => adapter.translate(key, params) },
    applyWholesome: (value: string) => value,
  });
  new Function(read('out/html/cat_coalitions.js'))();
  const classic = () => (window as any).CatCoalitionViz.render(config, q).textContent;
  for (const locale of ['ca', 'en', 'ca']) {
    adapter.refreshLocale(locale, locale === 'ca' ? catalog : {});
    // The real shell publishes a new frame when locale refresh completes.
    game.frame = { ...game.frame! };
    await wrapper.vm.$nextTick();
    const expected = locale === 'ca'
      ? ['Nou tripartit', 'Gran coalició liderada pel PSOE', 'Govern minoritari de cdc']
      : ['New Tripartite', 'PSOE-led Grand Coalition', 'cdc minority Government'];
    for (const label of expected) {
      expect(wrapper.text().toLowerCase()).toContain(label.toLowerCase());
      expect(classic().toLowerCase()).toContain(label.toLowerCase());
    }
  }
  expect(adapter.exportStateJSON()).toBe(before);
  expect(adapter.engine.random.getState()).toEqual(rng);
  wrapper.unmount();
});
