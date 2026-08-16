import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { SaveSlotEntry } from '../src/engine/types';
import { i18n, setLocale } from '../src/i18n';
import SaveLoadPane from '../src/components/menu/SaveLoadPane.vue';

const mocked = vi.hoisted(() => ({ store: {} as Record<string, unknown> }));

vi.mock('../src/stores/game', () => ({
  useGameStore: () => mocked.store,
}));

function ready(slot: string, savedAt: string, compatibility: SaveSlotEntry['compatibility'] = 'compatible'): SaveSlotEntry {
  return {
    slot,
    status: 'ready',
    compatibility,
    savedAt,
    year: 2016,
    month: 8,
    playerParty: 'ERC',
    resources: 12,
    sceneId: 'desk',
  };
}

function setup(initial: SaveSlotEntry[], savesDisabled = false) {
  let slots = initial;
  const store = {
    savesDisabled,
    listSlots: vi.fn(() => slots),
    loadSlot: vi.fn(() => ({ status: 'loaded' as const })),
    exportSlot: vi.fn(() => ({ ok: true as const, data: '{"save":true}' })),
    removeSlot: vi.fn((slot: string) => {
      slots = slots.filter((entry) => entry.slot !== slot);
      return { ok: true as const };
    }),
    createManualSave: vi.fn(() => {
      const slot = 'manual-2';
      slots = [ready(slot, '2026-08-15T15:00:00.000Z'), ...slots];
      return { ok: true, slot };
    }),
    overwriteManualSave: vi.fn((_slot: string, confirmed = false) => confirmed
      ? { ok: true, slot: _slot }
      : { ok: false, status: 'confirmation-required', slot: _slot, error: { code: 'confirmation-required' } }),
    importManualSave: vi.fn(() => {
      const slot = 'manual-3';
      slots = [ready(slot, '2026-08-15T16:00:00.000Z'), ...slots];
      return { ok: true, slot, status: 'ready' };
    }),
  };
  mocked.store = store;
  return store;
}

function mountPane(mode: 'title' | 'pause' = 'title') {
  return mount(SaveLoadPane, {
    attachTo: document.body,
    props: { mode, animations: false },
    global: { plugins: [i18n] },
  });
}

describe('SaveLoadPane', () => {
  beforeEach(() => {
    setLocale('en');
    document.body.innerHTML = '';
  });

  it('promotes autosaves, preserves supplied manual order, and keeps save commands out of title mode', async () => {
    const store = setup([
      ready('manual-9', '2026-08-15T14:00:00.000Z'),
      ready('auto-2', '2026-08-15T12:00:00.000Z'),
      ready('manual-1', '2026-08-15T13:00:00.000Z'),
      ready('auto-1', '2026-08-15T15:00:00.000Z'),
    ]);
    const wrapper = mountPane('title');

    expect(wrapper.findAll('[role="option"]').map((row) => row.attributes('data-test'))).toEqual([
      'save-row-auto-1',
      'save-row-auto-2',
      'save-row-manual-9',
      'save-row-manual-1',
    ]);
    expect(wrapper.find('[data-test="save-new"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="save-overwrite"]').exists()).toBe(false);
    expect(wrapper.get('[data-test="save-delete"]').attributes('disabled')).toBeDefined();

    await wrapper.get('[data-test="save-row-manual-9"]').trigger('click');
    await wrapper.get('[data-test="save-load"]').trigger('click');
    expect(store.loadSlot).toHaveBeenCalledWith('manual-9', false);
    expect(wrapper.emitted('loaded')).toEqual([['manual-9']]);
    wrapper.unmount();
  });

  it('keeps damaged records manageable while refusing to load them', async () => {
    const store = setup([{ slot: 'manual-4', status: 'corrupt', error: { code: 'invalid-json' } }]);
    const wrapper = mountPane();

    expect(wrapper.get('[data-test="save-load"]').attributes('disabled')).toBeDefined();
    expect(wrapper.get('[data-test="save-export"]').attributes('disabled')).toBeUndefined();
    expect(wrapper.get('[data-test="save-delete"]').attributes('disabled')).toBeUndefined();
    expect(wrapper.text()).toContain('cannot be loaded');

    await wrapper.get('[data-test="save-export"]').trigger('click');
    expect(store.exportSlot).toHaveBeenCalledWith('manual-4');
    expect(wrapper.emitted('exported')).toEqual([['manual-4', '{"save":true}']]);
    await wrapper.get('[data-test="save-delete"]').trigger('click');
    expect(store.removeSlot).toHaveBeenCalledWith('manual-4');
    expect(wrapper.find('[role="option"]').exists()).toBe(false);
    wrapper.unmount();
  });

  it('requires explicit confirmation for risky loads and occupied overwrites', async () => {
    const store = setup([ready('manual-1', '2026-08-15T13:00:00.000Z', 'incompatible')]);
    store.loadSlot
      .mockReturnValueOnce({ status: 'confirmation-required', compatibility: 'incompatible' } as never)
      .mockReturnValueOnce({ status: 'loaded' } as never);
    const wrapper = mountPane('pause');

    await wrapper.get('[data-test="save-load"]').trigger('click');
    expect(wrapper.get('[role="alertdialog"]').text()).toContain('different or unknown game version');
    await wrapper.get('[data-test="confirm-load"]').trigger('click');
    expect(store.loadSlot).toHaveBeenLastCalledWith('manual-1', true);

    await wrapper.get('[data-test="save-overwrite"]').trigger('click');
    expect(store.overwriteManualSave).toHaveBeenCalledWith('manual-1', false);
    expect(wrapper.get('[role="alertdialog"]').text()).toContain('replaced by the current game state');
    await wrapper.get('[data-test="confirm-overwrite"]').trigger('click');
    expect(store.overwriteManualSave).toHaveBeenLastCalledWith('manual-1', true);
    wrapper.unmount();
  });

  it('selects, focuses, and scrolls a newly imported row after resorting', async () => {
    const store = setup([ready('manual-1', '2026-08-15T13:00:00.000Z')]);
    const scrollIntoView = vi.fn();
    HTMLElement.prototype.scrollIntoView = scrollIntoView;
    const wrapper = mountPane('pause');
    const exposed = wrapper.vm as unknown as { importSerialized: (serialized: string) => Promise<void> };

    await exposed.importSerialized('{"canonical":true}');
    await flushPromises();

    expect(store.importManualSave).toHaveBeenCalledWith('{"canonical":true}');
    const imported = wrapper.get('[data-test="save-row-manual-3"]');
    expect(imported.attributes('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(imported.element);
    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' });
    wrapper.unmount();
  });

  it('shows pause-only save commands and explains ironman-disabled operations', () => {
    setup([ready('auto-1', '2026-08-15T13:00:00.000Z')], true);
    const wrapper = mountPane('pause');

    expect(wrapper.get('[data-test="ironman-reason"]').text()).toContain('latest recovery autosave');
    expect(wrapper.get('[data-test="save-load"]').attributes('disabled')).toBeDefined();
    expect(wrapper.get('[data-test="save-import"]').attributes('disabled')).toBeDefined();
    expect(wrapper.get('[data-test="save-new"]').attributes('disabled')).toBeDefined();
    expect(wrapper.get('[data-test="save-overwrite"]').attributes('disabled')).toBeDefined();
    expect(wrapper.get('[data-test="save-export"]').attributes('disabled')).toBeUndefined();
    wrapper.unmount();
  });
});
