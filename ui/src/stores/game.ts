import { computed, markRaw, ref, shallowRef } from 'vue';
import { defineStore } from 'pinia';
import { DendryAdapter } from '../engine/adapter';
import type { DrawResult, Frame, SaveMeta } from '../engine/types';

const SAVE_PREFIX = 'rti:desk:save:';

interface StoredSave {
  meta: SaveMeta;
  state: unknown;
}

export const useGameStore = defineStore('game', () => {
  const adapter = shallowRef<DendryAdapter | null>(null);
  const frame = ref<Frame | null>(null);
  const version = ref(0);
  const loadError = ref(false);

  const ready = computed(() => adapter.value !== null);
  const q = computed<Record<string, unknown>>(() => {
    void version.value; // tick dependency: Q is mutated in place by the engine
    return adapter.value ? { ...adapter.value.qualities } : {};
  });

  function apply(f: Frame): void {
    frame.value = f;
    version.value++;
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

  function newGame(): void {
    apply(adapter.value!.beginGame());
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
