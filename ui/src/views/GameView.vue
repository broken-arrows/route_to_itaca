<script setup lang="ts">
// GameView — the player-facing phase router (Desk UI Phase 2, Task 8:
// .superpowers/sdd/p2-task-8-brief.md). Boots the game the same way
// DebugPage does (see DebugPage.vue onMounted, via game.initFromUrl), then
// routes purely on the desk store's phase: the four "at the desk" phases
// render DeskView (Task 6/7's work, untouched here); 'eventPage' and 'page'
// render PaperPage, the neutral full-page paper surface (this task);
// 'boot' shows a loading/error state that reuses DebugPage's own i18n keys
// (debug.loading/debug.loadError) rather than duplicating that copy.
//
// Every routed surface renders INSIDE phase 1's StageScaler (default slot,
// mounted into its 1512x860 .stage element): DeskView and PaperPage are
// authored in design-space absolute pixels, so the scaler is what makes
// them viewport-correct. The boot loading/error state lives inside the
// stage too — one wrapper for all branches keeps the template flat, and a
// centered text line scales harmlessly. DebugPage stays UNWRAPPED (a plain
// document page, as in phase 1).
import { computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../stores/game';
import { useDeskStore } from '../stores/desk';
import StageScaler from '../components/StageScaler.vue';
import DeskView from './DeskView.vue';
import PaperPage from '../components/desk/PaperPage.vue';

const { t } = useI18n();
const game = useGameStore();
const desk = useDeskStore();

onMounted(async () => {
  if (!game.ready) {
    await game.initFromUrl(`${import.meta.env.BASE_URL}game.en.json`);
  }
  // Covers both the fresh-load case above and a game store that was
  // already ready (e.g. re-entering this view) but never started — either
  // way, exactly one newGame() call once ready with no frame yet.
  if (game.ready && !game.frame) {
    game.newGame();
  }
});

const DESK_PHASES = new Set(['idle', 'drawing', 'dossierOpen', 'resolving']);
const showDesk = computed(() => DESK_PHASES.has(desk.phase));
const pageVariant = computed<'page' | 'event' | 'ending'>(() => {
  if (desk.phase === 'eventPage') return 'event';
  return game.effectiveRole === 'ending' ? 'ending' : 'page';
});
</script>

<template>
  <StageScaler>
    <div v-if="desk.phase === 'boot'" class="boot-state" data-test="boot-state">
      <p v-if="game.loadError">{{ t('debug.loadError') }}</p>
      <p v-else>{{ t('debug.loading') }}</p>
    </div>
    <DeskView v-else-if="showDesk" />
    <PaperPage v-else :variant="pageVariant" />
  </StageScaler>
</template>

<style scoped>
.boot-state {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
}
</style>
