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
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../stores/game';
import { useDeskStore } from '../stores/desk';
import ResponsiveViewport from '../components/ResponsiveViewport.vue';
import DeskView from './DeskView.vue';
import PaperPage from '../components/desk/PaperPage.vue';
import Newspaper from '../components/desk/Newspaper.vue';
import FrontPage from '../components/desk/FrontPage.vue';
import Toast from '../components/desk/Toast.vue';
import TitleShell from '../components/shell/TitleShell.vue';
import PauseOverlay from '../components/shell/PauseOverlay.vue';
import { useShellStore } from '../stores/shell';
import LibrarySurface from '../components/library/LibrarySurface.vue';
import { captureLibraryUnderlay } from '../components/library/librarySnapshot';

const { t } = useI18n();
const game = useGameStore();
const desk = useDeskStore();
const shell = useShellStore();
const pauseOverlay = ref<InstanceType<typeof PauseOverlay> | null>(null);
const pauseButton = ref<HTMLButtonElement | null>(null);
const playingSurface = ref<HTMLElement | null>(null);
const pausedSurfaceHtml = ref('');
const libraryUnderlyingHtml = ref('');
let pauseInvoker: HTMLElement | null = null;

function syncApplicationMode(): void {
  const hub = game.roleHub('title-hub');
  if (game.effectiveRole === 'title-hub' && hub) shell.enterTitle(hub);
  else shell.beginPlaying();
}

function openPause(event?: Event): void {
  if (!shell.canPause(desk.phase, game.effectiveRole, game.frame?.gameOver ?? false)) return;
  pauseInvoker = event?.currentTarget instanceof HTMLElement
    ? event.currentTarget
    : document.activeElement instanceof HTMLElement ? document.activeElement : null;
  pausedSurfaceHtml.value = playingSurface.value?.innerHTML ?? '';
  shell.openPause(desk.phase, game.effectiveRole, game.frame?.gameOver ?? false);
}

function restorePauseFocus(): void {
  (pauseInvoker?.isConnected && pauseInvoker !== document.body ? pauseInvoker : pauseButton.value)?.focus();
  pauseInvoker = null;
}

watch(() => shell.paused, async (paused, wasPaused) => {
  if (!paused && wasPaused && shell.mode === 'playing') {
    pausedSurfaceHtml.value = '';
    await nextTick();
    restorePauseFocus();
  }
});

watch(() => game.frame, async (frame, previous) => {
  const enteringLibrary = frame?.effectiveRole === 'library-item'
    && previous?.effectiveRole !== 'library-item';
  const leavingLibrary = frame?.effectiveRole !== 'library-item'
    && previous?.effectiveRole === 'library-item';
  if (enteringLibrary && !libraryUnderlyingHtml.value) {
    libraryUnderlyingHtml.value = captureLibraryUnderlay(playingSurface.value);
  }
  // Pause may temporarily navigate the live engine to Achievements. That is
  // a detour, not a Library close: retain the original underlay until the
  // special scene itself returns through backSpecialScene.
  if (leavingLibrary && !shell.paused) {
    await nextTick();
    libraryUnderlyingHtml.value = '';
  }
}, { flush: 'sync' });

function onKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape' || desk.phase === 'boot') return;
  if (shell.mode === 'title') {
    if (shell.overlay === 'title-pane') {
      event.preventDefault();
      document.querySelector<HTMLElement>('[data-test="pane-close"]')?.click();
    }
    return;
  }
  event.preventDefault();
  if (shell.overlay === 'closed') openPause();
  else if (shell.overlay === 'pause-menu') void pauseOverlay.value?.resume();
  else void pauseOverlay.value?.back();
}

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
  if (game.frame) syncApplicationMode();
  window.addEventListener('keydown', onKeydown);
});

onUnmounted(() => window.removeEventListener('keydown', onKeydown));

const DESK_PHASES = new Set(['idle', 'drawing', 'dossierOpen', 'resolving']);
const PAUSE_PHASES = new Set(['page', 'idle', 'dossierOpen', 'newspaper', 'eventPage']);
const showDesk = computed(() => DESK_PHASES.has(desk.phase));
const showPauseButton = computed(() =>
  shell.mode === 'playing'
  && !(game.frame?.gameOver ?? false)
  && (PAUSE_PHASES.has(desk.phase) || game.effectiveRole === 'library-item'),
);
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
    <TitleShell v-else-if="shell.mode === 'title'" />
    <template v-else>
      <div
        v-if="shell.paused"
        class="playing-surface paused-surface-snapshot fogged"
        data-test="paused-surface-snapshot"
        inert
        aria-hidden="true"
        v-html="pausedSurfaceHtml"
      />
      <div ref="playingSurface" class="playing-surface" :class="{ 'live-under-pause': shell.paused }" :inert="shell.paused || undefined" data-test="playing-surface">
        <LibrarySurface
          v-if="game.effectiveRole === 'library-item'"
          :underlying-html="libraryUnderlyingHtml"
        />
        <DeskView v-else-if="showDesk" />
        <Newspaper v-else-if="desk.phase === 'newspaper'" />
        <FrontPage v-else-if="desk.phase === 'eventPage'" />
        <PaperPage v-else :variant="pageVariant" />
        <button
          v-if="showPauseButton"
          ref="pauseButton"
          type="button"
          class="pause-button"
          data-test="pause-button"
          :aria-label="t('shell.pause.open')"
          @click="openPause($event)"
        >
          <span aria-hidden="true">&#8214;</span>
        </button>
        <Toast :text-key="desk.toastKey" :achievement="desk.achievementToast" />
      </div>
      <PauseOverlay v-if="shell.paused" ref="pauseOverlay" />
    </template>
  </ResponsiveViewport>
</template>

<style scoped>
.boot-state {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
}
.playing-surface { position: absolute; inset: 0; }
.playing-surface.live-under-pause { visibility: hidden; }
.playing-surface.fogged { filter: sepia(.22) contrast(.78); }
.playing-surface.fogged::after { content: ''; position: absolute; inset: 0; z-index: 70; background: rgba(240, 232, 207, .66); backdrop-filter: blur(2px); }
.pause-button { position: absolute; z-index: 60; top: 20px; right: 20px; width: 44px; height: 44px; border: 1px solid #332d24; border-radius: 50%; background: #f3ecd5; color: #332d24; font: 700 1.2rem/1 sans-serif; }
.pause-button:focus-visible { outline: 3px solid #a21f1f; outline-offset: 3px; }
</style>
