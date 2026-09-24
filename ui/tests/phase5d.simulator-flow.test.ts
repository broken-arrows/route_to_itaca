import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import { DendryAdapter } from '../src/engine/adapter';
import GameView from '../src/views/GameView.vue';
import { useGameStore } from '../src/stores/game';
import { useShellStore } from '../src/stores/shell';
import { setAnimationsForTest } from '../src/stores/desk';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');
const presets = [
  { id: '2012_11', family: 'parlament', result: 'post_election_sim', seats: 135 },
  { id: '2015_5', family: 'local', result: 'post_election_sim_local', seats: 41 },
  { id: '2015_9', family: 'parlament', result: 'post_election_sim', seats: 135 },
  { id: '2015_12', family: 'congreso', result: 'post_election_sim_congreso', seats: 350 },
  { id: '2016_6', family: 'congreso', result: 'post_election_sim_congreso', seats: 350 },
  { id: '2017_12', family: 'parlament', result: 'post_election_sim', seats: 135 },
  { id: '2019_4', family: 'congreso', result: 'post_election_sim_congreso', seats: 350 },
  { id: '2019_5', family: 'local', result: 'post_election_sim_local', seats: 41 },
  { id: '2019_11', family: 'congreso', result: 'post_election_sim_congreso', seats: 350 },
] as const;

describe('Phase 5D election simulator', () => {
  it.each(presets)('runs the $id $family preset to its result and publishes its chamber', ({ id, family, result, seats }) => {
    const game = DendryAdapter.fromJSONText(gameText);
    game.beginGame([1, 2, 3, 4]);
    const menu = game.goToScene('election_simulation');
    const choice = menu.choices.findIndex((entry) => entry.id === `election_simulation.${id}`);
    expect(choice).toBeGreaterThanOrEqual(0);

    const frame = game.choose(choice);
    expect(frame.sceneId).toBe(`election_simulation.${result}`);
    const key = `election_simulation_${family === 'local' ? 'local' : family}_hemicycle`;
    const model = game.qualities[key] as { seats: Array<{ seats: number }> };
    expect(model.seats.reduce((sum, party) => sum + party.seats, 0)).toBe(seats);
    expect(frame.html).toContain('data-widget="hemicycle"');
    if (family === 'local') {
      const localChoice = frame.choices.findIndex((entry) => entry.id === 'election_simulation.local_cat_results');
      expect(localChoice).toBeGreaterThanOrEqual(0);
      expect(game.choose(localChoice).html).toContain('data-widget="local-results-map"');
    } else {
      expect(frame.html).toContain(`data-widget="${family}-results-map"`);
    }
  });
});

const mounted: Array<{ unmount(): void }> = [];
beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  setAnimationsForTest(false);
});
afterEach(() => {
  for (const wrapper of mounted.splice(0)) wrapper.unmount();
});

describe('Phase 5D title shell navigation', () => {
  it('keeps ribbons available at a nested result and fully resets before New Game', async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const game = useGameStore();
    const shell = useShellStore();
    game.initFromText(gameText);
    const wrapper = mount(GameView, { attachTo: document.body, global: { plugins: [pinia, i18n] } });
    mounted.push(wrapper);
    await flushPromises();
    const ribbon = (label: string) => {
      const button = wrapper.findAll('[data-test="ribbon-stack"] button').find((node) => node.text().includes(label));
      if (!button) throw new Error(`Missing ${label} ribbon: mode=${shell.mode} scene=${game.frame?.sceneId} choices=${JSON.stringify(wrapper.findAll('[data-test="ribbon-stack"] button').map((node) => node.text()))}`);
      return button;
    };

    await ribbon('Election Simulation').trigger('click');
    expect(shell.mode).toBe('title');
    const preset = game.frame?.choices.findIndex((entry) => entry.id === 'election_simulation.2012_11') ?? -1;
    expect(preset).toBeGreaterThanOrEqual(0);
    game.chooseFromShell(preset);
    await flushPromises();
    expect(game.frame?.sceneId).toBe('election_simulation.post_election_sim');
    expect(wrapper.find('[data-test="ribbon-stack"]').exists()).toBe(true);

    await ribbon('About').trigger('click');
    expect(game.frame?.sceneId).toContain('about');
    await ribbon('New Game').trigger('click');
    expect(shell.mode).toBe('playing');
    expect(game.frame?.sceneId).toBe('root.start');
    expect(game.q.year).toBe(2012);
    expect(game.q.election_simulation_parlament_hemicycle).toBeUndefined();
    expect(game.q.started).toBe(1);
  });
});
