<script setup lang="ts">
// GameView — the player-facing phase router (Desk UI Phase 2, Task 8:
// .superpowers/sdd/p2-task-8-brief.md). Boots the game the same way
// DebugPage does (see DebugPage.vue onMounted, via game.initFromUrl), then
// routes purely on the desk store's phase: the four "at the desk" phases
// render DeskView; newspaper/eventPage render their Phase-4A paper surfaces;
// other non-desk phases render PaperPage;
// 'boot' shows a loading/error state that reuses DebugPage's own i18n keys
// (debug.loading/debug.loadError) rather than duplicating that copy.
//
// Every routed player surface renders inside one viewport shell. The shell
// does not scale a design canvas: each surface resolves its own geometry so
// type and controls remain readable across desktop display sizes.
import { computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../stores/game';
import { useDeskStore } from '../stores/desk';
import ResponsiveViewport from '../components/ResponsiveViewport.vue';
import DeskView from './DeskView.vue';
import PaperPage from '../components/desk/PaperPage.vue';
import Newspaper from '../components/desk/Newspaper.vue';
import FrontPage from '../components/desk/FrontPage.vue';
import Toast from '../components/desk/Toast.vue';

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
  <ResponsiveViewport>
    <div v-if="desk.phase === 'boot'" class="boot-state" data-test="boot-state">
      <p v-if="game.loadError">{{ t('debug.loadError') }}</p>
      <p v-else>{{ t('debug.loading') }}</p>
    </div>
    <DeskView v-else-if="showDesk" />
    <Newspaper v-else-if="desk.phase === 'newspaper'" />
    <FrontPage v-else-if="desk.phase === 'eventPage'" />
    <PaperPage v-else :variant="pageVariant" />
    <!-- Mounted at the phase-router level, not inside DeskView: an
         achievement unlock (or an engine-error nudge) must stay visible
         regardless of which surface is currently showing — including the
         ending/page surfaces PaperPage renders (phase 2.5 Task 8; see
         docs/design/LEARNINGS.md). Both channels are no-ops (render
         nothing) while their own state is null, so this is safe during
         'boot' too. -->
    <Toast v-if="desk.phase !== 'boot'" :text-key="desk.toastKey" :achievement="desk.achievementToast" />
  </ResponsiveViewport>
</template>

<style scoped>
.boot-state {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
}
</style>
