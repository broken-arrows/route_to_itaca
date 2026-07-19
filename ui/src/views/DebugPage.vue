<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../stores/game';
import Prose from '../components/Prose.vue';

const { t } = useI18n();
const store = useGameStore();
const qFilter = ref('');
const slotName = ref('debug');

onMounted(async () => {
  if (!store.ready) {
    await store.initFromUrl(`${import.meta.env.BASE_URL}game.en.json`);
    if (store.ready) store.newGame();
  }
});

const qRows = computed(() => {
  const needle = qFilter.value.toLowerCase();
  return Object.entries(store.q)
    .filter(([k]) => !needle || k.toLowerCase().includes(needle))
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(0, 200);
});

function onDraw(deckId: string) {
  const result = store.draw(deckId);
  if (result.id === null) {
    console.warn(result.title === 'no_space_in_hand' ? t('debug.handFull') : t('debug.deckEmpty'));
  }
}
</script>

<template>
  <div class="debug" v-if="store.loadError">
    <p>{{ t('debug.loadError') }}</p>
  </div>
  <div class="debug" v-else-if="!store.ready">
    <p>{{ t('debug.loading') }}</p>
  </div>
  <div class="debug" v-else>
    <section class="controls">
      <button data-test="new-game" @click="store.newGame()">{{ t('debug.newGame') }}</button>
      <input data-test="slot-name" v-model="slotName" :placeholder="t('debug.slotName')" />
      <button data-test="save" @click="store.saveSlot(slotName)">{{ t('debug.save') }}</button>
      <button data-test="load" @click="store.loadSlot(slotName)">{{ t('debug.load') }}</button>
      <span v-if="store.frame?.gameOver">{{ t('debug.gameOver') }}</span>
    </section>

    <section class="scene">
      <!-- Engine-authored prose; trusted content from our own game.json -->
      <Prose class="prose" :html="store.frame?.html ?? ''" />
    </section>

    <section v-if="store.frame && !store.frame.isHand" class="choices">
      <h2>{{ t('debug.choices') }}</h2>
      <button
        v-for="(c, i) in store.frame.choices"
        :key="c.id"
        data-test="choice"
        :disabled="!c.canChoose"
        :title="c.subtitle"
        @click="store.choose(i)"
      >
        <!-- Engine output (convertLine returns HTML: <em>/<strong>/raw magic
             blocks), same trust boundary as the prose above — interpolating it
             showed the player literal tags. -->
        <Prose tag="span" class="choice-title" :html="c.title" />
        <Prose v-if="c.subtitle" tag="span" class="choice-subtitle" :html="c.subtitle" />
      </button>
    </section>

    <section v-if="store.frame?.isHand" class="hand-area">
      <h2>{{ t('debug.decks') }}</h2>
      <!-- d.title/card.title/p.title are all engine output too (same
           CaptureUI normalization as choice titles above) — Prose so a
           deck/hand/pinned title the engine marked as a glossary term (e.g.
           a pinned advisor card whose own name matches one, see
           OpenDossier's cover-title fix) renders as an element rather than
           literal tag text. -->
      <button
        v-for="d in store.frame.decks"
        :key="d.id"
        data-test="deck"
        :disabled="!d.canChoose"
        @click="onDraw(d.id)"
      >
        <Prose tag="span" :html="d.title" /> — {{ t('debug.draw') }}
      </button>

      <h2>{{ t('debug.hand', { n: store.frame.hand.length, max: store.frame.maxCards }) }}</h2>
      <button
        v-for="card in store.frame.hand"
        :key="card.id"
        data-test="hand-card"
        @click="store.play(card.id)"
      >
        <Prose tag="span" :html="card.title" /> — {{ t('debug.play') }}
      </button>

      <h2>{{ t('debug.pinned') }}</h2>
      <button
        v-for="p in store.frame.pinned"
        :key="p.id"
        data-test="pinned"
        @click="store.playPinned(p.id)"
      >
        <Prose tag="span" :html="p.title" />
      </button>
    </section>

    <section class="inspector">
      <h2>{{ t('debug.inspector') }}</h2>
      <input data-test="q-filter" v-model="qFilter" :placeholder="t('debug.filter')" />
      <table>
        <tr v-for="[k, v] in qRows" :key="k" data-test="q-row">
          <td>{{ k }}</td>
          <td>{{ v }}</td>
        </tr>
      </table>
    </section>
  </div>
</template>

<style scoped>
.debug {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
  display: grid;
  gap: 16px;
}
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.choices, .hand-area { display: grid; gap: 6px; justify-items: start; }
.inspector table { font-size: 12px; border-collapse: collapse; }
.inspector td { border-bottom: 1px solid var(--paper-1); padding: 2px 10px 2px 0; }
button:disabled { opacity: 0.5; }
</style>
