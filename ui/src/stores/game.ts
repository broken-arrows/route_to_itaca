import { computed, markRaw, ref, shallowRef } from 'vue';
import { defineStore } from 'pinia';
import { createSaveStore, type SaveStore } from 'dendrynexus-ten/lib/persistence.js';
import { DendryAdapter } from '../engine/adapter';
import type {
  AchievementEntry,
  AchievementLedger,
  DrawResult,
  EffectiveRole,
  Frame,
  GameInfo,
  SaveMeta,
  SaveSlotEntry,
} from '../engine/types';
import type { GlossaryTerm } from '../glossary/mark';
import { useSettingsStore } from './settings';
import { useShellStore } from './shell';
import { loadContentCatalog } from '../locales/content';
import type { AppLocale } from '../i18n';

export type LoadSlotResult =
  | { status: 'loaded' }
  | { status: 'blocked'; error: { code: 'saves-disabled' } }
  | { status: 'missing' | 'corrupt' | 'unreadable' | 'unsupported'; error?: { code: string } }
  | { status: 'confirmation-required'; compatibility: 'incompatible' | 'unknown' };

export type ManualSlotOperationResult =
  | { ok: true; slot: string; status?: 'ready' | 'unsupported' }
  | { ok: false; error: { code: string; message?: string } };

export type OverwriteManualSlotResult =
  | ManualSlotOperationResult
  | { ok: false; status: 'confirmation-required'; slot: string };

const MANUAL_SLOT = /^manual-([1-9]\d*)$/;

function disabledResult() {
  return { ok: false as const, error: { code: 'saves-disabled' as const } };
}

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
  const achievementLedger = computed<AchievementLedger>(() => {
    void version.value;
    return adapter.value?.achievementLedger ?? {};
  });
  const savesDisabled = computed(() => {
    void version.value;
    return adapter.value?.savesDisabled ?? false;
  });

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
      await setContentLanguage(useSettingsStore().language);
    } catch (err) {
      console.error('game data load failed:', err);
      loadError.value = true;
    }
  }

  async function setContentLanguage(language: AppLocale): Promise<void> {
    if (!adapter.value) return;
    const catalog = await loadContentCatalog(language);
    const refreshed = adapter.value.refreshLocale(language, catalog);
    if (refreshed) apply(refreshed);
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
    if (useShellStore().blocksEngineChoices) return;
    apply(adapter.value!.choose(i));
  }
  function chooseFromShell(i: number): void {
    apply(adapter.value!.choose(i));
  }
  function goToScene(id: string): void {
    apply(adapter.value!.goToScene(id));
  }
  function roleHub(role: 'title-hub') {
    return adapter.value?.roleHub(role) ?? null;
  }
  function captureState(): string {
    return adapter.value!.exportStateJSON();
  }
  function restoreState(serialized: string): void {
    apply(adapter.value!.importStateJSON(serialized));
  }
  function draw(deckId: string): DrawResult {
    if (useShellStore().blocksEngineChoices) return { id: null, title: 'no_card_in_deck' };
    const { result, frame: f } = adapter.value!.drawCard(deckId);
    apply(f);
    return result;
  }
  function play(cardId: string): void {
    if (useShellStore().blocksEngineChoices) return;
    apply(adapter.value!.playCard(cardId));
  }
  function playPinned(cardId: string): void {
    if (useShellStore().blocksEngineChoices) return;
    apply(adapter.value!.playPinnedCard(cardId));
  }
  function saveSlot(slot: string) {
    if (savesDisabled.value && slot !== 'auto-1') return disabledResult();
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
    if (savesDisabled.value) {
      const written = saveSlot('auto-1');
      if (!written.ok) return written;
      const removed = saves.remove('auto-2');
      return removed.ok ? written : removed;
    }
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
    if (!savesDisabled.value) return saveSlot('auto-1');
    const written = saveSlot('auto-1');
    if (!written.ok || !saves) return written;
    const removed = saves.remove('auto-2');
    return removed.ok ? written : removed;
  }

  function loadSlot(slot: string, allowRisk = false): LoadSlotResult {
    if (!saves || !adapter.value) return { status: 'missing' };
    if (savesDisabled.value && slot !== 'auto-1') {
      return { status: 'blocked', error: { code: 'saves-disabled' } };
    }
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
    const entries: SaveSlotEntry[] = saves.list().map((entry) => {
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
    return entries.sort((left, right) => {
      const autoRank = (slot: string) => slot === 'auto-1' ? 0 : slot === 'auto-2' ? 1 : 2;
      const rankDifference = autoRank(left.slot) - autoRank(right.slot);
      if (rankDifference !== 0) return rankDifference;

      const leftManual = MANUAL_SLOT.exec(left.slot);
      const rightManual = MANUAL_SLOT.exec(right.slot);
      if (leftManual && rightManual) {
        const timestamp = (entry: SaveSlotEntry) => {
          const parsed = entry.savedAt ? Date.parse(entry.savedAt) : Number.NaN;
          return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
        };
        const leftTime = timestamp(left);
        const rightTime = timestamp(right);
        if (leftTime !== rightTime) return rightTime - leftTime;
        return Number(leftManual[1]) - Number(rightManual[1]);
      }
      if (leftManual) return -1;
      if (rightManual) return 1;
      return left.slot.localeCompare(right.slot);
    });
  }

  function nextManualSlot(): string {
    const occupied = new Set(saves?.list().map(({ slot }) => slot) ?? []);
    let number = 1;
    while (occupied.has(`manual-${number}`)) number++;
    return `manual-${number}`;
  }

  function createManualSave(): ManualSlotOperationResult {
    if (savesDisabled.value) return disabledResult();
    const slot = nextManualSlot();
    const result = saveSlot(slot);
    return result.ok
      ? { ok: true, slot }
      : { ok: false, error: result.error };
  }

  function overwriteManualSave(slot: string, confirmed = false): OverwriteManualSlotResult {
    if (savesDisabled.value) return disabledResult();
    if (!MANUAL_SLOT.test(slot)) {
      return { ok: false as const, error: { code: 'invalid-manual-slot' as const } };
    }
    if (!saves) return { ok: false as const, error: { code: 'persistence-unconfigured' as const } };
    if (saves.read(slot).status !== 'missing' && !confirmed) {
      return { ok: false, status: 'confirmation-required', slot };
    }
    const result = saveSlot(slot);
    return result.ok
      ? { ok: true, slot }
      : { ok: false, error: result.error };
  }

  function removeSlot(slot: string) {
    return saves?.remove(slot) ?? { ok: false, error: { code: 'persistence-unconfigured' } };
  }

  function exportSlot(slot: string) {
    return saves?.export(slot) ?? { ok: false, error: { code: 'persistence-unconfigured' } };
  }

  function importSlot(slot: string, serialized: string) {
    if (savesDisabled.value) return disabledResult();
    return saves?.import(slot, serialized) ?? {
      ok: false,
      error: { code: 'persistence-unconfigured' },
    };
  }


  function importManualSave(serialized: string): ManualSlotOperationResult {
    if (savesDisabled.value) return disabledResult();
    const slot = nextManualSlot();
    const result = importSlot(slot, serialized);
    return result.ok
      ? { ok: true, status: result.status, slot }
      : { ok: false, error: result.error };
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
    achievementLedger,
    savesDisabled,
    loadError,
    initFromText,
    initFromUrl,
    setContentLanguage,
    newGame,
    choose,
    chooseFromShell,
    goToScene,
    roleHub,
    captureState,
    restoreState,
    draw,
    play,
    playPinned,
    saveSlot,
    createManualSave,
    overwriteManualSave,
    saveAutosave,
    checkpointAutosave,
    loadSlot,
    listSlots,
    removeSlot,
    exportSlot,
    importSlot,
    importManualSave,
  };
});
