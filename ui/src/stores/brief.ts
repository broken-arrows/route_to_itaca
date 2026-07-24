import { computed, ref } from 'vue';
import { defineStore } from 'pinia';
import { useGameStore } from './game';

/**
 * The Brief's tab state. The tab LIST is game content (the hub scene's
 * options); this store only tracks which one is open and renders it.
 *
 * Rendering is out of band — `adapter.renderView` evaluates the sheet's content
 * tree against current Q and touches nothing. So switching tabs is free, and
 * the sheet re-renders on every Q tick with no cache to invalidate.
 */
export const useBriefStore = defineStore('brief', () => {
  const game = useGameStore();
  const selected = ref<string | null>(null);

  // DEVIATION from the task brief (see task report): `tabScenes()` filters by
  // `view-if` against LIVE, non-reactive Q (a plain object Vue cannot track).
  // Without a tick dependency, this computed's only tracked reactive source
  // was `game.adapter` (the ref itself), which never changes across a
  // session — so in the brief's literal version, `tabs` would silently
  // freeze at whatever it evaluated to on its FIRST access and never
  // re-derive again, even across ordinary gameplay (a card/event that flips
  // a view-if-gated quality). `void game.q` is the same "re-render on every Q
  // tick" idiom `activeHtml` below already uses, for the same reason.
  const tabs = computed(() => {
    void game.q;
    return game.adapter?.tabScenes() ?? [];
  });

  const activeTab = computed<string | null>(() => {
    const list = tabs.value;
    if (!list.length) return null;
    // A selected tab that has since hidden itself (POLLS on entering historical
    // mode) falls back to the first, rather than rendering nothing.
    if (selected.value && list.some((t) => t.id === selected.value)) return selected.value;
    return list[0].id;
  });

  const activeHtml = computed<string>(() => {
    void game.q; // re-render on every Q tick
    const id = activeTab.value;
    return id && game.adapter ? game.adapter.renderView(id) : '';
  });

  function select(id: string): void {
    selected.value = id;
  }

  return { tabs, activeTab, activeHtml, select };
});
