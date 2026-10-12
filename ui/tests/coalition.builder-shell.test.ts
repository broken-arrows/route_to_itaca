import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import GameView from '../src/views/GameView.vue';
import { useGameStore } from '../src/stores/game';
import { useShellStore, type ApplicationMode, type ShellOverlay, type ShellPane } from '../src/stores/shell';
import { setAnimationsForTest } from '../src/stores/desk';
import { miniGameText } from './fixtures/mini-game';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
let pinia: ReturnType<typeof createPinia>;
const mounted: Array<{ unmount(): void }> = [];

beforeEach(() => {
  localStorage.clear();
  pinia = createPinia();
  setActivePinia(pinia);
  setAnimationsForTest(false);
  i18n.global.locale.value = 'en';
});
afterEach(() => {
  for (const wrapper of mounted.splice(0)) wrapper.unmount();
});

describe('Coalition Builder from the title shell', () => {
  it('edits the draft from the real Playground pane and publishes only on Run', async () => {
    const game = useGameStore();
    const shell = useShellStore();
    game.initFromText(gameText);
    const wrapper = mount(GameView, { attachTo: document.body, global: { plugins: [pinia, i18n] } });
    mounted.push(wrapper);
    await flushPromises();

    const click = async (selector: string, text: string) => {
      const button = wrapper.findAll(selector).find(node => node.text() === text);
      expect(button, `Missing ${text}`).toBeDefined();
      await button!.trigger('click');
      await flushPromises();
    };
    await click('[data-test="ribbon-stack"] button', 'Playground');
    await click('.authored-choices button', 'Coalition Builder');
    await click('.authored-choices button', 'Parlament de Catalunya');
    expect(shell.mode).toBe('title');
    expect(shell.overlay).toBe('title-pane');
    expect(shell.pane).toBe('authored');
    expect(shell.blocksEngineChoices).toBe(true);

    const published = () => Object.fromEntries(Object.entries(game.adapter!.qualities)
      .filter(([key]) => key.endsWith('_parlament_s')));
    const initialPublished = published();
    const initialRandom = game.adapter!.engine.random.getState();
    const initialTotal = Number(game.q.playground_coalitions_set_seats);
    const initialERC = Number(game.q.erc_parlament_s);
    const erc = () => wrapper.get('.builder-playground [data-party="erc"]');
    const run = () => game.frame!.choices.find(choice => choice.id.endsWith('run_parlament'))!;

    await erc().get('.builder-controls button').trigger('click');
    expect((erc().get('input[type="text"]').element as HTMLInputElement).value).toBe(String(initialERC - 1));
    expect(game.q.playground_coalitions_set_seats).toBe(initialTotal - 1);
    expect(run().canChoose).toBe(false);
    expect(published()).toEqual(initialPublished);

    await erc().get('input[type="text"]').setValue(String(initialERC));
    expect(run().canChoose).toBe(true);
    await wrapper.get('[data-party="icv"] .builder-next').trigger('click');
    expect(wrapper.find('[data-party="csqp"]').exists()).toBe(true);
    await erc().get('input[type="checkbox"]').setValue(true);
    expect((erc().get('input[type="checkbox"]').element as HTMLInputElement).checked).toBe(true);
    await erc().get('input[type="checkbox"]').setValue(false);
    expect(game.q.playground_coalitions_set_seats).toBe(initialTotal);
    expect(published()).toEqual(initialPublished);
    expect(game.adapter!.engine.random.getState()).toEqual(initialRandom);

    const draft = game.adapter!.qualities.playground_coalitions as { rows: Array<{ id: string; seats: number }> };
    const draftCSQP = draft.rows.find(row => row.id === 'csqp')!.seats;
    await click('.authored-choices button', 'Set this Parlament');
    expect(game.q.csqp_parlament_s).toBe(draftCSQP);
    expect(game.q.icv_parlament_s).toBe(0);
    expect(shell.mode).toBe('title');
  });
});

describe('Inline edit shell boundaries', () => {
  const blocked: Array<{ name: string; mode: ApplicationMode; overlay: ShellOverlay; pane: ShellPane; transitioning?: boolean }> = [
    { name: 'closed title menu', mode: 'title', overlay: 'closed', pane: null },
    { name: 'title Options', mode: 'title', overlay: 'title-pane', pane: 'options' },
    { name: 'title Load', mode: 'title', overlay: 'title-pane', pane: 'load' },
    { name: 'transitioning authored title pane', mode: 'title', overlay: 'title-pane', pane: 'authored', transitioning: true },
    { name: 'pause menu', mode: 'playing', overlay: 'pause-menu', pane: null },
    { name: 'pause pane', mode: 'playing', overlay: 'pause-pane', pane: 'authored' },
    { name: 'exit confirmation', mode: 'playing', overlay: 'exit-confirm', pane: null },
    { name: 'playing transition', mode: 'playing', overlay: 'closed', pane: null, transitioning: true },
  ];
  it.each(blocked)('keeps edits blocked in $name', ({ mode, overlay, pane, transitioning }) => {
    const game = useGameStore();
    game.initFromText(miniGameText);
    game.newGame();
    const shell = useShellStore();
    Object.assign(shell, { mode, overlay, pane, transitioning: transitioning ?? false });
    const initial = game.captureState();
    const initialRandom = game.adapter!.engine.random.getState();
    let called = false;
    game.editQualities(q => { called = true; q.gold = 999; return true; });
    expect(called).toBe(false);
    expect(game.captureState()).toBe(initial);
    expect(game.adapter!.engine.random.getState()).toEqual(initialRandom);
  });
});
