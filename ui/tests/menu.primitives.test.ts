import { describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import RibbonStack, { type RibbonItem } from '../src/components/menu/RibbonStack.vue';
import StepAsidePane from '../src/components/menu/StepAsidePane.vue';
import { splitAuthoredPane } from '../src/components/menu/authoredPane';

const items: RibbonItem[] = [
  { id: 'continue', title: 'Continue', subtitle: 'autosave', tone: 'gold' },
  { id: 'load', title: 'Load Game', subtitle: 'No usable saves', disabled: true, tone: 'red' },
  { id: 'options', title: 'Options', tone: 'dark' },
];

describe('RibbonStack', () => {
  it('keeps logical item order in semantic buttons and emits only enabled selections', async () => {
    const wrapper = mount(RibbonStack, { props: { items, ariaLabel: 'Main menu' } });
    const buttons = wrapper.findAll('button');

    expect(wrapper.get('nav').attributes('aria-label')).toBe('Main menu');
    expect(buttons.map(button => button.text())).toEqual([
      'Continueautosave',
      'Load GameNo usable saves',
      'Options',
    ]);
    expect(buttons[1].attributes('disabled')).toBeDefined();

    await buttons[0].trigger('click');
    await buttons[1].trigger('click');
    expect(wrapper.emitted('select')).toEqual([['continue']]);
  });

  it('marks the active item and exposes focus entry without owning selection state', async () => {
    const wrapper = mount(RibbonStack, {
      attachTo: document.body,
      props: { items, activeId: 'options' },
    });
    const options = wrapper.get('[data-test="ribbon-options"]');

    expect(options.attributes('aria-current')).toBe('page');
    await (wrapper.vm as unknown as { focusActive: () => Promise<boolean> }).focusActive();
    expect(document.activeElement).toBe(options.element);
    wrapper.unmount();
  });

  it('removes its entrance-animation class when animations are disabled', () => {
    const wrapper = mount(RibbonStack, { props: { items, animations: false } });
    expect(wrapper.get('[data-test="ribbon-stack"]').classes()).not.toContain('is-animated');
  });
});

describe('StepAsidePane', () => {
  it('provides a labelled paper region, header metadata, and scrollable content slot', () => {
    const wrapper = mount(StepAsidePane, {
      props: { title: 'Achievements', meta: '3 / 24 filed' },
      slots: { default: '<p>Ledger contents</p>' },
    });
    const pane = wrapper.get('[data-test="step-aside-pane"]');
    const heading = wrapper.get('[data-test="pane-heading"]');

    expect(pane.attributes('aria-labelledby')).toBe(heading.attributes('id'));
    expect(heading.text()).toBe('Achievements');
    expect(wrapper.text()).toContain('3 / 24 filed');
    expect(wrapper.text()).toContain('Ledger contents');
  });

  it('emits close and exposes both focus-entry targets', async () => {
    const wrapper = mount(StepAsidePane, {
      attachTo: document.body,
      props: { title: 'Options', closeLabel: 'Back to menu' },
    });
    const exposed = wrapper.vm as unknown as {
      focusHeading: () => Promise<boolean>;
      focusClose: () => Promise<boolean>;
    };

    await exposed.focusHeading();
    expect(document.activeElement).toBe(wrapper.get('[data-test="pane-heading"]').element);
    await exposed.focusClose();
    expect(document.activeElement).toBe(wrapper.get('[data-test="pane-close"]').element);

    await wrapper.get('[data-test="pane-close"]').trigger('click');
    expect(wrapper.emitted('close')).toEqual([[]]);
    wrapper.unmount();
  });

  it('can be rendered as a stable, motionless pane without close chrome', () => {
    const wrapper = mount(StepAsidePane, {
      props: { title: 'Load Game', animations: false, showClose: false },
    });
    expect(wrapper.get('[data-test="step-aside-pane"]').classes()).not.toContain('is-animated');
    expect(wrapper.find('[data-test="pane-close"]').exists()).toBe(false);
  });
});

describe('authored pane headings', () => {
  it('promotes only a leading authored h1 and removes it from the body', () => {
    expect(splitAuthoredPane('<p> </p>\n<h1>Authored <em>heading</em></h1><p>Body</p>')).toEqual({
      titleHtml: 'Authored <em>heading</em>',
      bodyHtml: '<p>Body</p>',
    });
  });

  it('does not invent a visual heading from scene metadata when prose has none', () => {
    expect(splitAuthoredPane('<p>Body</p>')).toEqual({
      titleHtml: null,
      bodyHtml: '<p>Body</p>',
    });
  });
});
