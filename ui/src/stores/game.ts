import { computed, markRaw, ref, shallowRef } from 'vue';
import { defineStore } from 'pinia';
import { DendryAdapter } from '../engine/adapter';
import type { AchievementEntry, DrawResult, EffectiveRole, Frame, GameInfo, SaveMeta } from '../engine/types';
import type { GlossaryTerm } from '../glossary/mark';

// `dnt:` prefix (the library, not the game) — see i18n.ts's STORAGE_KEY
// comment for the naming rule and the phase-5 per-game discriminator plan.
const SAVE_PREFIX = 'dnt:save:';
// Pre-rename prefix (phases 1–2.5 shipped with the game-named `rti:`).
const LEGACY_SAVE_PREFIX = 'rti:desk:save:';

interface StoredSave {
  meta: SaveMeta;
  state: unknown;
}

// One-time, idempotent: copy any pre-rename save slots to the new prefix so
// existing beta players keep them. Copy (not move) and never overwrite — if
// both exist the new key already won, and leaving the old blob behind is
// harmless (nothing writes it again; delete-legacy can ride along when phase 5
// reworks the shelf anyway).
function migrateLegacySaves(): void {
  if (typeof localStorage === 'undefined') return;
  const legacyKeys: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith(LEGACY_SAVE_PREFIX)) legacyKeys.push(key);
  }
  for (const key of legacyKeys) {
    const target = SAVE_PREFIX + key.slice(LEGACY_SAVE_PREFIX.length);
    if (localStorage.getItem(target) === null) {
      localStorage.setItem(target, localStorage.getItem(key)!);
    }
  }
}

export const useGameStore = defineStore('game', () => {
  migrateLegacySaves();

  const adapter = shallowRef<DendryAdapter | null>(null);
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
  function saveSlot(slot: string): void {
    const a = adapter.value!;
    const qs = a.qualities;
    const meta: SaveMeta = {
      slot,
      savedAt: new Date().toISOString(),
      year: typeof qs.year === 'number' ? qs.year : null,
      month: typeof qs.month === 'number' ? qs.month : null,
      playerParty: typeof qs.player_party === 'string' ? qs.player_party : null,
      sceneId: frame.value?.sceneId ?? '',
      resources: typeof qs.party_resources === 'number' ? qs.party_resources : null,
    };
    const stored: StoredSave = { meta, state: JSON.parse(a.exportStateJSON()) };
    localStorage.setItem(SAVE_PREFIX + slot, JSON.stringify(stored));
  }

  function loadSlot(slot: string): boolean {
    const raw = localStorage.getItem(SAVE_PREFIX + slot);
    if (!raw || !adapter.value) return false;
    const stored = JSON.parse(raw) as StoredSave;
    apply(adapter.value.importStateJSON(JSON.stringify(stored.state)));
    return true;
  }

  function listSlots(): SaveMeta[] {
    const out: SaveMeta[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(SAVE_PREFIX)) {
        out.push((JSON.parse(localStorage.getItem(key)!) as StoredSave).meta);
      }
    }
    return out.sort((a, b) => a.slot.localeCompare(b.slot));
  }

  return {
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
    loadSlot,
    listSlots,
  };
});
