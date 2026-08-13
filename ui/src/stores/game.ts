import { computed, markRaw, ref, shallowRef } from 'vue';
import { defineStore } from 'pinia';
import { createSaveStore, type SaveStore } from 'dendrynexus-ten/lib/persistence.js';
import { DendryAdapter } from '../engine/adapter';
import type {
  AchievementEntry,
  DrawResult,
  EffectiveRole,
  Frame,
  GameInfo,
  SaveMeta,
  SaveSlotEntry,
} from '../engine/types';
import type { GlossaryTerm } from '../glossary/mark';
import { useSettingsStore } from './settings';

export type LoadSlotResult =
  | { status: 'loaded' }
  | { status: 'missing' | 'corrupt' | 'unreadable' | 'unsupported'; error?: { code: string } }
  | { status: 'confirmation-required'; compatibility: 'incompatible' | 'unknown' };

export const useGameStore = defineStore('game', () => {
  const adapter = shallowRef<DendryAdapter | null>(null);
  let saves: SaveStore | null = null;
  const frame = ref<Frame | null>(null);
  const version = ref(0);
  const loadError = ref(false);

  const ready = computed(() => adapter.value !== null);
  const q = computed<Record<string, unknown>>(() => {
    void version.value; // tick dependency: Q is mutated in place by the engine
    return adapter.value ? { ...adapter.value.qualities } : {};
  });
  const info = computed<GameInfo | null>(() => (adapter.value ? adapter.value.info : null));
  const effectiveRole = computed<EffectiveRole>(() => frame.value?.effectiveRole ?? 'page');
  // game.json.data.glossary.terms — static per load, but gated on the
  // adapter existing at all (pre-boot / load-error leaves it empty, never
  // throws). Consumed by Prose.vue/GlossaryTerm.vue via useGlossary().
  const glossary = computed<GlossaryTerm[]>(() => adapter.value?.glossary ?? []);
  // game.json.data.achievements.achievements — same "static per load, empty
  // pre-boot/on load-error" contract as glossary above. Consumed by the
  // desk store (toast metadata) and AchievementGallery.vue.
  const achievements = computed<AchievementEntry[]>(() => adapter.value?.achievements ?? []);

  function apply(f: Frame): void {
    // Order matters: version ticks BEFORE the frame is published, so the
    // flush:'sync' frame watcher (desk.ts's syncFromFrame -> checkAchievements)
    // reads a FRESH `q` snapshot on the SAME transition, not the previous
    // one. That watcher fires synchronously inside the `frame.value = f`
    // assignment below; if `q` (a computed memoized on `version`) was
    // already read-and-cached by some other consumer since the last bump
    // (e.g. DeskView's deskMonth/deskYear, read on every render), reading
    // it again before invalidating would return the stale cached copy,
    // missing whatever this transition's action just mutated on Q — see
    // ui/tests/store.desk.achievement-timing.test.ts. `version.value++`
    // itself fires no synchronous watcher (nothing watches `version`
    // directly; `q` is a lazy computed), so bumping it first is
    // observationally safe.
    version.value++;
    frame.value = f;
  }

  function initFromText(text: string): void {
    adapter.value = markRaw(DendryAdapter.fromJSONText(text));
    const manifest = adapter.value.info;
    if (manifest.storageId) {
      saves = createSaveStore({
        storage: localStorage,
        storageId: manifest.storageId,
        gameVersion: manifest.version,
      });
      useSettingsStore().configure(manifest.storageId);
    } else {
      // Backward-compatible engine games can still run, but durable browser
      // state is deliberately unavailable until they declare a namespace.
      saves = null;
      console.warn('game manifest has no storage-id; saves and settings are disabled');
    }
    loadError.value = false;
  }

  async function initFromUrl(url: string): Promise<void> {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      initFromText(await res.text());
    } catch (err) {
      console.error('game data load failed:', err);
      loadError.value = true;
    }
  }

  // Spec §9: the engine can throw on boot (beginGame -> goToScene(root) asserts
  // the scene exists). It used to be uncaught inside GameView's async
  // onMounted, where the rejection was swallowed: the phase stayed 'boot' and
  // the UI showed "Loading…" forever with loadError still false. Surface it on
  // the same flag the fetch failure uses — both mean "there is no game to
  // play" and both render the boot error state.
  function newGame(): boolean {
    try {
      apply(adapter.value!.beginGame());
      return true;
    } catch (err) {
      console.error('newGame failed:', err);
      loadError.value = true;
      return false;
    }
  }
  function choose(i: number): void {
    apply(adapter.value!.choose(i));
  }
  function draw(deckId: string): DrawResult {
    const { result, frame: f } = adapter.value!.drawCard(deckId);
    apply(f);
    return result;
  }
  function play(cardId: string): void {
    apply(adapter.value!.playCard(cardId));
  }
  function playPinned(cardId: string): void {
    apply(adapter.value!.playPinnedCard(cardId));
  }
  function saveSlot(slot: string) {
    const a = adapter.value!;
    const qs = a.qualities;
    const meta: SaveMeta = {
      savedAt: '', // the persistence clock overwrites this atomically
      year: typeof qs.year === 'number' ? qs.year : null,
      month: typeof qs.month === 'number' ? qs.month : null,
      playerParty: typeof qs.player_party === 'string' ? qs.player_party : null,
      sceneId: frame.value?.sceneId ?? '',
      resources: typeof qs.party_resources === 'number' ? qs.party_resources : null,
    };
    if (!saves) return { ok: false, error: { code: 'persistence-unconfigured' } };
    return saves.write(slot, JSON.parse(a.exportStateJSON()), meta as unknown as Record<string, unknown>);
  }

  // Canonical autosave ordering is positional, not a flip-flop: auto-1 is
  // always newest and auto-2 is always the previous auto-1. Reuse the store's
  // export/import validation instead of reaching around its interface into
  // localStorage. If an existing auto-1 is corrupt, abort rather than destroy
  // the only recoverable raw copy.
  function saveAutosave() {
    if (!saves) return { ok: false, error: { code: 'persistence-unconfigured' } };
    const current = saves.read('auto-1');
    if (current.status !== 'missing') {
      const exported = saves.export('auto-1');
      if (!exported.ok) return exported;
      const shifted = saves.import('auto-2', exported.data);
      if (!shifted.ok || shifted.status !== 'ready') {
        return shifted.ok
          ? { ok: false, error: { code: 'autosave-source-not-loadable' } }
          : shifted;
      }
    }
    return saveSlot('auto-1');
  }

  // A news sequence can span several player decisions after the turn-boundary
  // rotation. Keep auto-2 as the genuine previous-turn rollback while making
  // the current sequence resumable: checkpoints replace auto-1 directly and
  // deliberately do not pass through saveAutosave()'s rotation.
  function checkpointAutosave() {
    return saveSlot('auto-1');
  }

  function loadSlot(slot: string, allowRisk = false): LoadSlotResult {
    if (!saves || !adapter.value) return { status: 'missing' };
    const stored = saves.read(slot);
    if (stored.status !== 'ready') {
      return { status: stored.status, error: 'error' in stored ? stored.error : undefined };
    }
    if (stored.compatibility !== 'compatible' && !allowRisk) {
      return { status: 'confirmation-required', compatibility: stored.compatibility };
    }
    apply(adapter.value.importStateJSON(JSON.stringify(stored.record.state)));
    return { status: 'loaded' };
  }

  function listSlots(): SaveSlotEntry[] {
    if (!saves) return [];
    return saves.list().map((entry) => {
      if (entry.status === 'ready') {
        return {
          slot: entry.slot,
          status: entry.status,
          compatibility: entry.compatibility,
          meta: entry.record.meta as unknown as SaveMeta,
          ...(entry.record.meta as unknown as SaveMeta),
        };
      }
      return { slot: entry.slot, status: entry.status, error: entry.error };
    });
  }

  function removeSlot(slot: string) {
    return saves?.remove(slot) ?? { ok: false, error: { code: 'persistence-unconfigured' } };
  }

  function exportSlot(slot: string) {
    return saves?.export(slot) ?? { ok: false, error: { code: 'persistence-unconfigured' } };
  }

  function importSlot(slot: string, serialized: string) {
    return saves?.import(slot, serialized) ?? {
      ok: false,
      error: { code: 'persistence-unconfigured' },
    };
  }

  return {
    adapter,
    ready,
    frame,
    q,
    info,
    effectiveRole,
    glossary,
    achievements,
    loadError,
    initFromText,
    initFromUrl,
    newGame,
    choose,
    draw,
    play,
    playPinned,
    saveSlot,
    saveAutosave,
    checkpointAutosave,
    loadSlot,
    listSlots,
    removeSlot,
    exportSlot,
    importSlot,
  };
});
