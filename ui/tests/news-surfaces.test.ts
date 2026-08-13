import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { i18n } from '../src/i18n';
import Newspaper from '../src/components/desk/Newspaper.vue';
import FrontPage from '../src/components/desk/FrontPage.vue';
import { useGameStore } from '../src/stores/game';
import { useDeskStore, setAnimationsForTest } from '../src/stores/desk';
import uiEn from '../../source/locales/en/ui.json';

i18n.global.mergeLocaleMessage('en', uiEn as never);

let pinia: ReturnType<typeof createPinia>;
beforeEach(() => {
  pinia = createPinia();
  setActivePinia(pinia);
  setAnimationsForTest(false);
});

function plugins() {
  return { global: { plugins: [pinia, i18n] } };
}

function sceneGame(role: 'newspaper' | 'event', count = 3) {
  const options = Array.from({ length: count }, (_, i) => ({ id: `@choice_${i}` }));
  const choices = Object.fromEntries(
    options.map((_, i) => [
      `choice_${i}`,
      {
        id: `choice_${i}`,
        type: 'scene',
        title: `<strong>Choice ${i + 1}</strong>`,
        subtitle: `Authored subtitle ${i + 1}`,
        chooseIf: i === 1 ? { $code: 'return false;' } : undefined,
        content: [],
        goTo: [{ id: 'root' }],
      },
    ]),
  );
  return {
    scenes: {
      root: {
        id: 'root',
        type: 'scene',
        title: 'El Matí',
        role,
        faceImage: role === 'event' ? 'img/events/authored.jpg' : undefined,
        newPage: true,
        content: [{ type: 'paragraph', content: ['Engine-authored current prose.'] }],
        options,
      },
      ...choices,
    },
    qualities: {},
    qdisplays: {},
    tagLookup: {},
  };
}

function start(role: 'newspaper' | 'event', count = 3) {
  const desk = useDeskStore();
  const game = useGameStore();
  game.initFromText(JSON.stringify(sceneGame(role, count)));
  game.newGame();
  return { game, desk };
}

describe('Newspaper', () => {
  it('uses the authored scene title as its masthead', () => {
    start('newspaper');
    const wrapper = mount(Newspaper, plugins());
    expect(wrapper.find('.masthead > h1').text()).toBe('El Matí');
  });

  it('renders every authored story in order without a presentation cap', () => {
    start('newspaper', 12);
    const wrapper = mount(Newspaper, plugins());
    const stories = wrapper.findAll('[data-test^="newspaper-story-"]');
    expect(stories).toHaveLength(12);
    expect(stories[0].text()).toContain('Choice 1');
    expect(stories[11].text()).toContain('Choice 12');
    expect(wrapper.find('.news-region').exists()).toBe(true);
  });

  it('uses the ordinary indexed choice and ignores unavailable stories', async () => {
    const { desk } = start('newspaper');
    const choose = vi.spyOn(desk, 'chooseNewspaperStory');
    const wrapper = mount(Newspaper, plugins());

    await wrapper.find('[data-test="newspaper-story-0"]').trigger('click');
    expect(choose).toHaveBeenCalledWith(0);

    choose.mockClear();
    await wrapper.find('[data-test="newspaper-story-1"]').trigger('click');
    expect(choose).not.toHaveBeenCalled();
    expect(wrapper.find('[data-test="newspaper-story-1"]').attributes('aria-disabled')).toBe('true');
  });

  it('keeps the live Brief mounted beside the scrollable paper', () => {
    start('newspaper');
    const wrapper = mount(Newspaper, plugins());
    expect(wrapper.find('.clipboard-frame').exists()).toBe(true);
    expect(wrapper.find('.news-region').exists()).toBe(true);
  });
});

describe('FrontPage', () => {
  it('renders the current event face image from the generic frame contract', () => {
    start('event');
    const wrapper = mount(FrontPage, plugins());
    expect(wrapper.find('.event-image').attributes('src')).toContain('img/events/authored.jpg');
  });

  it('renders only the current authored prose and ordinary choices', () => {
    start('event');
    const wrapper = mount(FrontPage, plugins());
    expect(wrapper.text()).toContain('Engine-authored current prose.');
    expect(wrapper.findAll('[data-test^="event-choice-"]')).toHaveLength(3);
    expect(wrapper.text()).not.toContain("Tomorrow's headlines");
    expect(wrapper.text()).not.toContain('Your answer');
  });

  it('dispatches enabled answers by their unchanged frame index', async () => {
    const { desk } = start('event');
    const choose = vi.spyOn(desk, 'chooseEventChoice');
    const wrapper = mount(FrontPage, plugins());

    await wrapper.find('[data-test="event-choice-2"]').trigger('click');
    expect(choose).toHaveBeenCalledWith(2);

    choose.mockClear();
    await wrapper.find('[data-test="event-choice-1"]').trigger('click');
    expect(choose).not.toHaveBeenCalled();
  });

  it('keeps arbitrary widget markers in the generic prose pipeline', () => {
    const { game } = start('event');
    game.frame!.html = '<p>Before.</p><div data-widget="unknown-event-widget"></div><p>After.</p>';
    const wrapper = mount(FrontPage, plugins());
    expect(wrapper.text()).toContain('Before.');
    expect(wrapper.text()).toContain('After.');
    expect(wrapper.find('[data-widget-missing="unknown-event-widget"]').exists()).toBe(true);
  });
});
