import { computed, ref } from 'vue';
import { defineStore } from 'pinia';
import type { ChoiceView, EffectiveRole } from '../engine/types';
import type { DeskPhase } from './desk';

export type ApplicationMode = 'title' | 'playing';
export type ShellOverlay = 'closed' | 'pause-menu' | 'title-pane' | 'pause-pane' | 'exit-confirm';
export type ShellPane = 'load' | 'options' | 'achievements' | 'authored' | null;

const STABLE_PHASES = new Set<DeskPhase>(['page', 'idle', 'dossierOpen', 'newspaper', 'eventPage']);

export const useShellStore = defineStore('shell', () => {
  const mode = ref<ApplicationMode>('playing');
  const overlay = ref<ShellOverlay>('closed');
  const pane = ref<ShellPane>(null);
  const titleHubId = ref<string | null>(null);
  const titleChoices = ref<ChoiceView[]>([]);
  const pauseReturnState = ref<string | null>(null);

  const paused = computed(() =>
    overlay.value === 'pause-menu' || overlay.value === 'pause-pane' || overlay.value === 'exit-confirm',
  );
  const blocksEngineChoices = computed(() => paused.value || mode.value === 'title');

  function enterTitle(hub: { id: string; choices: ChoiceView[] }): void {
    mode.value = 'title';
    overlay.value = 'closed';
    pane.value = null;
    titleHubId.value = hub.id;
    titleChoices.value = hub.choices;
    pauseReturnState.value = null;
  }

  function refreshTitleHub(hub: { id: string; choices: ChoiceView[] }): void {
    titleHubId.value = hub.id;
    titleChoices.value = hub.choices;
  }

  function beginPlaying(): void {
    mode.value = 'playing';
    overlay.value = 'closed';
    pane.value = null;
    pauseReturnState.value = null;
  }

  function canPause(phase: DeskPhase, role: EffectiveRole, gameOver: boolean): boolean {
    if (mode.value !== 'playing' || overlay.value !== 'closed' || gameOver) return false;
    return STABLE_PHASES.has(phase) || role === 'library-item';
  }

  function openPause(phase: DeskPhase, role: EffectiveRole, gameOver: boolean): boolean {
    if (!canPause(phase, role, gameOver)) return false;
    overlay.value = 'pause-menu';
    pane.value = null;
    return true;
  }

  function openTitlePane(next: Exclude<ShellPane, null>): void {
    if (mode.value !== 'title') return;
    pane.value = next;
    overlay.value = 'title-pane';
  }

  function openPausePane(next: Exclude<ShellPane, null>): void {
    if (!paused.value) return;
    pane.value = next;
    overlay.value = 'pause-pane';
  }

  function showExitConfirmation(): void {
    if (!paused.value) return;
    overlay.value = 'exit-confirm';
  }

  function backToPauseMenu(): void {
    if (!paused.value) return;
    overlay.value = 'pause-menu';
    pane.value = null;
  }

  function resume(): void {
    overlay.value = 'closed';
    pane.value = null;
    pauseReturnState.value = null;
  }

  return {
    mode,
    overlay,
    pane,
    titleHubId,
    titleChoices,
    pauseReturnState,
    paused,
    blocksEngineChoices,
    enterTitle,
    refreshTitleHub,
    beginPlaying,
    canPause,
    openPause,
    openTitlePane,
    openPausePane,
    showExitConfirmation,
    backToPauseMenu,
    resume,
  };
});
